# SPDX-License-Identifier: Apache-2.0

"""Standalone Qwen forced-aligner worker.

This file is intentionally executed by path as a child process. It must not
import vllm_omni, because the parent server patches vLLM modules globally and
native Qwen forced alignment relies on vLLM's original pooling-output schema.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import unicodedata
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

AudioChunk: TypeAlias = bytes


def pcm16_chunk_to_float32(audio_chunk: AudioChunk) -> np.ndarray:
    if not isinstance(audio_chunk, bytes):
        raise TypeError(f"audio_chunk must be bytes. Got {type(audio_chunk)!r}.")
    if not audio_chunk:
        return np.zeros((0,), dtype=np.float32)
    wav = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    return np.clip(wav / 32768.0, -1.0, 1.0)


def resample_linear(
    wav: np.ndarray,
    src_sr: int,
    dst_sr: int = 16000,
) -> np.ndarray:
    if src_sr <= 0:
        raise ValueError(f"sr must be positive. Got {src_sr}.")

    wav = np.asarray(wav, dtype=np.float32)
    if src_sr == dst_sr or wav.size == 0:
        return wav

    duration = wav.size / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=wav.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, wav).astype(np.float32)


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def tokenize_alignment_units(text: str) -> list[str]:
    units: list[str] = []
    buf: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            token = "".join(ch for ch in buf if _is_kept_char(ch))
            if token:
                units.append(token)
            buf = []

    for ch in text:
        if _is_cjk_char(ch):
            flush_buf()
            units.append(ch)
        elif _is_kept_char(ch):
            buf.append(ch)
        else:
            flush_buf()

    flush_buf()
    return units


def build_qwen_aligner_prompt(units: list[str]) -> str:
    body = "<timestamp><timestamp>".join(units)
    if body:
        body += "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def fix_monotonic_timestamps(timestamp_ms: list[int]) -> list[int]:
    fixed: list[int] = []
    last = 0

    for timestamp in timestamp_ms:
        timestamp = max(0, int(round(timestamp)))
        if timestamp < last:
            timestamp = last
        fixed.append(timestamp)
        last = timestamp

    return fixed


def _assert_clean_import_state() -> None:
    if "vllm_omni" in sys.modules:
        raise RuntimeError("Forced aligner worker imported vllm_omni; native vLLM pooling schema isolation is broken.")


class QwenForcedAligner:
    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        gpu_memory_utilization: float,
    ) -> None:
        # These are parent-only knobs. Keep them out of vLLM's environment
        # parser in this clean worker process.
        for key in tuple(os.environ):
            if key.startswith("VLLM_OMNI_FORCED_ALIGNER_"):
                os.environ.pop(key, None)

        _assert_clean_import_state()
        from vllm import LLM

        _assert_clean_import_state()
        self.llm = LLM(
            model=model,
            runner="pooling",
            dtype=dtype,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_overrides={
                "architectures": [
                    "Qwen3ASRForcedAlignerForTokenClassification",
                ],
            },
        )

        config = self.llm.llm_engine.vllm_config.model_config.hf_config
        self.timestamp_token_id = int(config.timestamp_token_id)
        self.timestamp_segment_time = float(config.timestamp_segment_time)

    def align(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[dict[str, Any]]:
        text = text.strip()
        if not text:
            return []

        wav = pcm16_chunk_to_float32(audio_chunk)
        if wav.size == 0:
            return []

        units = tokenize_alignment_units(text)
        if not units:
            return []

        wav_16k = resample_linear(wav, src_sr=sr, dst_sr=16000)
        prompt = build_qwen_aligner_prompt(units)
        outputs = self.llm.encode(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "audio": wav_16k,
                    },
                }
            ],
            pooling_task="token_classify",
        )

        output = outputs[0]
        predictions = output.outputs.data.argmax(dim=-1)
        timestamp_ms = [
            int(round(float(pred.item()) * self.timestamp_segment_time))
            for token_id, pred in zip(output.prompt_token_ids, predictions)
            if int(token_id) == self.timestamp_token_id
        ]

        needed = len(units) * 2
        if len(timestamp_ms) < needed:
            return []

        timestamp_ms = fix_monotonic_timestamps(timestamp_ms[:needed])
        return [
            {
                "word": units[i],
                "start_ms": timestamp_ms[i * 2],
                "end_ms": max(timestamp_ms[i * 2], timestamp_ms[i * 2 + 1]),
            }
            for i in range(len(units))
        ]


def serve(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    listener = Listener(str(socket_path), family="AF_UNIX")
    aligner: QwenForcedAligner | None = None

    try:
        conn = listener.accept()
        try:
            while True:
                try:
                    request = conn.recv()
                except EOFError:
                    break

                if not isinstance(request, dict):
                    conn.send({"ok": False, "error": "request must be a dict"})
                    continue

                request_type = request.get("type")
                if request_type == "shutdown":
                    conn.send({"ok": True})
                    break
                if request_type != "align":
                    conn.send(
                        {
                            "ok": False,
                            "error": f"unknown request type: {request_type!r}",
                        }
                    )
                    continue

                try:
                    if aligner is None:
                        aligner = QwenForcedAligner(
                            model=args.model,
                            dtype=args.dtype,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                        )
                    timestamps = aligner.align(
                        audio_chunk=request["audio_chunk"],
                        text=request["text"],
                        sr=int(request["sample_rate"]),
                    )
                    conn.send({"ok": True, "word_timestamps": timestamps})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    conn.send(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        finally:
            conn.close()
    except KeyboardInterrupt:
        return 0
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen forced aligner worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    return serve(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
