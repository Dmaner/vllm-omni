# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import unicodedata
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

AudioChunk: TypeAlias = bytes


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int


def pcm16_chunk_to_float32(audio_chunk: AudioChunk) -> np.ndarray:
    """
    Convert vLLM-Omni PCM streaming chunk to mono float32 waveform.

    Contract:
        audio_chunk is the raw bytes yielded by
        OmniOpenAIServingSpeech._generate_pcm_chunks(...).

    In the current TTS streaming path this is RAW PCM_16 audio bytes.
    """
    if not isinstance(audio_chunk, bytes):
        raise TypeError(
            "audio_chunk must be bytes produced by "
            "OmniOpenAIServingSpeech._generate_pcm_chunks(). "
            f"Got {type(audio_chunk)!r}."
        )

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

    if src_sr == dst_sr:
        return wav

    if wav.size == 0:
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
    """
    MVP tokenizer:
    - CJK: character-level
    - Latin / space-separated text: word-level
    - Mixed text: CJK chars + Latin word spans
    """
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

    for t in timestamp_ms:
        t = max(0, int(round(t)))
        if t < last:
            t = last
        fixed.append(t)
        last = t

    return fixed


class VllmQwenForcedAligner:
    def __init__(self) -> None:
        from vllm import LLM

        model = os.getenv(
            "VLLM_OMNI_FORCED_ALIGNER_MODEL",
            "Qwen/Qwen3-ForcedAligner-0.6B",
        )
        dtype = os.getenv("VLLM_OMNI_FORCED_ALIGNER_DTYPE", "bfloat16")
        gpu_memory_utilization = float(os.getenv("VLLM_OMNI_FORCED_ALIGNER_GPU_MEMORY_UTILIZATION", "0.25"))

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

        self._infer_lock = asyncio.Lock()

    async def align(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[WordTimestamp]:
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

        async with self._infer_lock:
            outputs = await asyncio.to_thread(
                self.llm.encode,
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
            WordTimestamp(
                word=units[i],
                start_ms=timestamp_ms[i * 2],
                end_ms=max(timestamp_ms[i * 2], timestamp_ms[i * 2 + 1]),
            )
            for i in range(len(units))
        ]


_ALIGNER: VllmQwenForcedAligner | None = None
_ALIGNER_LOCK = asyncio.Lock()


async def get_forced_aligner() -> VllmQwenForcedAligner:
    global _ALIGNER

    if _ALIGNER is not None:
        return _ALIGNER

    async with _ALIGNER_LOCK:
        if _ALIGNER is None:
            _ALIGNER = await asyncio.to_thread(VllmQwenForcedAligner)

    return _ALIGNER


async def align(
    audio_chunk: AudioChunk,
    text: str,
    sr: int,
) -> list[WordTimestamp]:
    """
    Shared forced-alignment API for vLLM-Omni TTS.

    Args:
        audio_chunk:
            Raw PCM bytes produced by vLLM-Omni TTS streaming.
        text:
            Text corresponding to this audio chunk.
        sr:
            Sample rate of audio_chunk.

    Returns:
        Word/character timestamps.
    """
    aligner = await get_forced_aligner()
    return await aligner.align(audio_chunk, text, sr)
