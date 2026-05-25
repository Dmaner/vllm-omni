# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS streaming word timestamps.

These tests verify the /v1/audio/speech/stream WebSocket endpoint emits
per-sentence word timestamps for generated streaming PCM audio.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest
import websockets

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
EXPECTED_UNITS = ["Hello", "world"]
RUN_LOCAL_4090_TEST = os.environ.get("VLLM_OMNI_RUN_LOCAL_4090_TESTS") == "1"


def get_prompt(prompt_type="english"):
    prompts = {
        "english": "Hello world.",
    }
    return prompts.get(prompt_type, prompts["english"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=[
                "--trust-remote-code",
                "--forced-aligner",
                FORCED_ALIGNER_MODEL,
            ],
        ),
        id="async_chunk",
    )
]


async def _run_timestamp_ws_session(host: str, port: int, model: str) -> dict[str, Any]:
    prompt = get_prompt()
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"
    starts: list[dict[str, Any]] = []
    dones: list[dict[str, Any]] = []
    timestamp_events: list[dict[str, Any]] = []
    chunk_lengths: dict[int, list[int]] = {}
    session_done: dict[str, Any] | None = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": model,
                    "voice": "Ryan",
                    "language": "English",
                    "response_format": "pcm",
                    "stream_audio": True,
                    "word_timestamps": True,
                    "max_new_tokens": 512,
                }
            )
        )
        await ws.send(json.dumps({"type": "input.text", "text": prompt}))
        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=300)
            if isinstance(message, bytes):
                if not starts:
                    raise AssertionError("Received audio bytes before audio.start")
                sentence_index = starts[-1]["sentence_index"]
                chunk_lengths.setdefault(sentence_index, []).append(len(message))
                continue

            payload = json.loads(message)
            msg_type = payload.get("type")
            if msg_type == "audio.start":
                starts.append(payload)
                chunk_lengths.setdefault(payload["sentence_index"], [])
            elif msg_type == "audio.word_timestamps":
                timestamp_events.append(payload)
            elif msg_type == "audio.done":
                dones.append(payload)
            elif msg_type == "session.done":
                session_done = payload
                break
            elif msg_type == "error":
                raise AssertionError(f"WebSocket error: {payload['message']}")
            else:
                raise AssertionError(f"Unexpected WebSocket message: {payload}")

    return {
        "starts": starts,
        "dones": dones,
        "timestamp_events": timestamp_events,
        "chunk_lengths": chunk_lengths,
        "session_done": session_done,
    }


def _assert_word_timestamps(timestamps: list[dict[str, Any]]) -> None:
    assert [item["word"] for item in timestamps] == EXPECTED_UNITS

    last_end = 0
    for item in timestamps:
        start_ms = item["start_ms"]
        end_ms = item["end_ms"]
        assert isinstance(start_ms, int)
        assert isinstance(end_ms, int)
        assert 0 <= start_ms <= end_ms
        assert start_ms >= last_end
        last_end = end_ms


def _assert_timestamp_ws_session(omni_server) -> None:
    result = asyncio.run(_run_timestamp_ws_session(omni_server.host, omni_server.port, omni_server.model))

    starts = result["starts"]
    dones = result["dones"]
    timestamp_events = result["timestamp_events"]
    chunk_lengths = result["chunk_lengths"]
    session_done = result["session_done"]

    assert session_done is not None
    assert session_done["total_sentences"] == 1
    assert len(starts) == 1
    assert len(dones) == 1
    assert len(timestamp_events) == 1

    start = starts[0]
    assert start["type"] == "audio.start"
    assert start["sentence_index"] == 0
    assert start["format"] == "pcm"
    assert start["sample_rate"] == 24000
    assert start["sentence_text"] == get_prompt()

    done = dones[0]
    assert done["sentence_index"] == 0
    assert done["error"] is False
    assert done["total_bytes"] > 0
    assert chunk_lengths[0]
    assert sum(chunk_lengths[0]) == done["total_bytes"]

    timestamp_event = timestamp_events[0]
    assert timestamp_event["sentence_index"] == 0
    assert "error" not in timestamp_event
    _assert_word_timestamps(timestamp_event["word_timestamps"])


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_word_timestamps_001(omni_server) -> None:
    """
    Test WebSocket streaming TTS with word timestamps.
    Deploy Setting: default async_chunk yaml + forced aligner
    Input Modal: text
    Output Modal: audio + word timestamps
    Input Setting: stream_audio=True, response_format=pcm
    Datasets: single request
    """
    _assert_timestamp_ws_session(omni_server)


@pytest.mark.skipif(
    not RUN_LOCAL_4090_TEST,
    reason="Local RTX 4090 timestamp test. Set VLLM_OMNI_RUN_LOCAL_4090_TESTS=1 to run.",
)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_word_timestamps_4090_001(omni_server) -> None:
    """
    Test WebSocket streaming TTS with word timestamps on RTX 4090.
    Deploy Setting: default async_chunk yaml + forced aligner
    Input Modal: text
    Output Modal: audio + word timestamps
    Input Setting: stream_audio=True, response_format=pcm
    Datasets: single request
    """
    _assert_timestamp_ws_session(omni_server)
