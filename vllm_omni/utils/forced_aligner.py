# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
import os
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any, TypeAlias

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
    Alignment tokenizer:
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
    """Managed client for a native-vLLM Qwen forced-aligner worker.

    The worker is launched as a standalone Python script instead of importing
    vLLM inside the already-patched vllm-omni server process. This keeps native
    pooling outputs on the native vLLM schema and avoids patching the generic
    EngineCoreOutput transport just for forced alignment.
    """

    def __init__(self) -> None:
        self.model = os.getenv(
            "VLLM_OMNI_FORCED_ALIGNER_MODEL",
            "Qwen/Qwen3-ForcedAligner-0.6B",
        )
        self.dtype = os.getenv("VLLM_OMNI_FORCED_ALIGNER_DTYPE", "bfloat16")
        self.gpu_memory_utilization = float(os.getenv("VLLM_OMNI_FORCED_ALIGNER_GPU_MEMORY_UTILIZATION", "0.8"))
        self.startup_timeout = float(os.getenv("VLLM_OMNI_FORCED_ALIGNER_STARTUP_TIMEOUT", "60"))
        self.request_timeout = float(os.getenv("VLLM_OMNI_FORCED_ALIGNER_REQUEST_TIMEOUT", "900"))

        self._infer_lock = asyncio.Lock()
        self._conn: Connection | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._socket_path: str | None = None
        atexit.register(self.shutdown)

    async def align(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[WordTimestamp]:
        text = text.strip()
        if not text:
            return []
        if not audio_chunk:
            return []

        units = tokenize_alignment_units(text)
        if not units:
            return []

        async with self._infer_lock:
            result = await asyncio.to_thread(
                self._align_sync,
                audio_chunk,
                text,
                sr,
            )

        return [
            WordTimestamp(
                word=str(item["word"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
            )
            for item in result
        ]

    def _align_sync(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_connection()
        try:
            conn.send(
                {
                    "type": "align",
                    "audio_chunk": audio_chunk,
                    "text": text,
                    "sample_rate": sr,
                }
            )
            if not conn.poll(self.request_timeout):
                self._close_worker(kill=True)
                raise TimeoutError(f"Forced aligner worker timed out after {self.request_timeout:.1f}s")
            response = conn.recv()
        except (EOFError, OSError) as exc:
            self._close_worker(kill=True)
            raise RuntimeError("Forced aligner worker connection failed") from exc

        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid forced aligner response: {response!r}")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "forced alignment failed")))
        timestamps = response.get("word_timestamps", [])
        if not isinstance(timestamps, list):
            raise RuntimeError("Invalid forced aligner timestamp payload")
        return timestamps

    def _ensure_connection(self) -> Connection:
        if self._conn is not None and self._proc is not None and self._proc.poll() is None:
            return self._conn

        self._close_worker(kill=True)
        self._start_worker()
        if self._conn is None:
            raise RuntimeError("Forced aligner worker did not connect")
        return self._conn

    def _start_worker(self) -> None:
        worker_path = Path(__file__).with_name("_qwen_forced_aligner_worker.py")
        socket_path = Path("/tmp") / f"vllm_omni_forced_aligner_{os.getpid()}_{uuid.uuid4().hex}.sock"
        self._socket_path = str(socket_path)

        cmd = [
            sys.executable,
            str(worker_path),
            "--socket",
            self._socket_path,
            "--model",
            self.model,
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
            env=env,
        )

        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Forced aligner worker exited during startup with code {self._proc.returncode}")
            try:
                self._conn = Client(self._socket_path, family="AF_UNIX")
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)

        self._close_worker(kill=True)
        raise TimeoutError(
            "Timed out waiting for forced aligner worker startup" + (f": {last_error}" if last_error else "")
        )

    def _close_worker(self, kill: bool = False) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            if kill:
                proc.kill()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

        if self._socket_path is not None:
            try:
                Path(self._socket_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._socket_path = None

    def shutdown(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                conn.send({"type": "shutdown"})
                if conn.poll(5):
                    conn.recv()
            except Exception:
                pass
        self._close_worker(kill=False)


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
