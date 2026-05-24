# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import subprocess
import sys
import time
import traceback
import unicodedata
import uuid
from dataclasses import dataclass, fields
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np

AudioChunk: TypeAlias = bytes

QWEN_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_FORCED_ALIGNER_DEPLOY_CONFIG = "qwen3_tts_forced_aligner.yaml"


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ForcedAlignerConfig:
    model: str | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    startup_timeout: float | None = None
    request_timeout: float | None = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        raise ValueError(f"forced_aligner requires a model path/name, such as {QWEN_FORCED_ALIGNER_MODEL}.")

    @classmethod
    def disabled(cls) -> ForcedAlignerConfig:
        return cls()

    @classmethod
    def from_model(cls, model: str | None) -> ForcedAlignerConfig:
        if model is None:
            return cls.disabled()
        value = str(model).strip()
        if not value:
            raise ValueError(f"forced_aligner requires a model path/name, such as {QWEN_FORCED_ALIGNER_MODEL}.")
        return cls(model=value)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ForcedAlignerConfig:
        model = raw.get("model")
        if model is None:
            return cls.disabled()

        model_config = cls.from_model(str(model))
        dtype = raw.get("dtype")
        gpu_memory_utilization = raw.get("gpu_memory_utilization")
        max_model_len = raw.get("max_model_len")
        max_num_batched_tokens = raw.get("max_num_batched_tokens")
        max_num_seqs = raw.get("max_num_seqs")
        startup_timeout = raw.get("startup_timeout")
        request_timeout = raw.get("request_timeout")

        return cls(
            model=model_config.model,
            dtype=None if dtype is None else str(dtype),
            gpu_memory_utilization=(None if gpu_memory_utilization is None else float(gpu_memory_utilization)),
            max_model_len=None if max_model_len is None else int(max_model_len),
            max_num_batched_tokens=(None if max_num_batched_tokens is None else int(max_num_batched_tokens)),
            max_num_seqs=None if max_num_seqs is None else int(max_num_seqs),
            startup_timeout=(None if startup_timeout is None else float(startup_timeout)),
            request_timeout=(None if request_timeout is None else float(request_timeout)),
        )


class ForcedAligner(Protocol):
    async def align(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[WordTimestamp]: ...

    def shutdown(self) -> None: ...


class NoOpForcedAligner:
    async def align(
        self,
        audio_chunk: AudioChunk,
        text: str,
        sr: int,
    ) -> list[WordTimestamp]:
        return []

    def shutdown(self) -> None:
        return None


def _load_forced_aligner_section_from_deploy_config(
    config_path: str | Path | None,
) -> Any | None:
    """Read ``forced_aligner`` from a vLLM-Omni deploy config.

    Deploy YAML parsing is delegated to ``resolve_deploy_yaml`` so
    ``base_config`` inheritance follows the same behavior as the normal
    vLLM-Omni stage config path.
    """
    if not config_path:
        return None

    path = Path(config_path)
    if not path.exists():
        return None

    from vllm_omni.config.stage_config import resolve_deploy_yaml

    data = resolve_deploy_yaml(path)
    if not isinstance(data, dict) or "forced_aligner" not in data:
        return None
    return data.get("forced_aligner")


def _resolve_forced_aligner_deploy_path(
    deploy_config: str | Path,
    base_dir: Path | None = None,
) -> Path:
    path = Path(deploy_config).expanduser()
    if path.is_absolute():
        return path

    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parent.parent / "deploy" / path


def _forced_aligner_mapping_from_raw(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return {"model": raw}
    if not isinstance(raw, dict):
        raise TypeError(f"forced_aligner config must be a model path/name or mapping. Got {type(raw)!r}.")

    return {field.name: raw[field.name] for field in fields(ForcedAlignerConfig) if field.name in raw}


def _read_forced_aligner_deploy_mapping(
    deploy_config: str | Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    path = _resolve_forced_aligner_deploy_path(deploy_config, base_dir=base_dir)
    raw = _load_forced_aligner_section_from_deploy_config(path)
    return _forced_aligner_mapping_from_raw(raw)


def _read_default_forced_aligner_mapping() -> dict[str, Any]:
    return _read_forced_aligner_deploy_mapping(QWEN_FORCED_ALIGNER_DEPLOY_CONFIG)


def _build_forced_aligner_mapping(
    raw: Any,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    mapping = _read_default_forced_aligner_mapping()
    if isinstance(raw, dict) and raw.get("deploy_config") is not None:
        mapping.update(
            _read_forced_aligner_deploy_mapping(
                raw["deploy_config"],
                base_dir=base_dir,
            )
        )
    mapping.update(_forced_aligner_mapping_from_raw(raw))
    return mapping


def build_forced_aligner_config(args: Any | None = None) -> ForcedAlignerConfig:
    if args is None:
        return ForcedAlignerConfig.disabled()

    deploy_config = getattr(args, "deploy_config", None)
    raw = _load_forced_aligner_section_from_deploy_config(deploy_config)
    selector = getattr(args, "forced_aligner", None)

    if raw is None and selector is None:
        return ForcedAlignerConfig.disabled()

    mapping = _build_forced_aligner_mapping(
        raw,
        base_dir=Path(deploy_config).parent if deploy_config else None,
    )
    if selector is not None:
        mapping["model"] = ForcedAlignerConfig.from_model(selector).model

    return ForcedAlignerConfig.from_mapping(mapping)


def create_forced_aligner(config: ForcedAlignerConfig) -> ForcedAligner:
    if not config.enabled:
        return NoOpForcedAligner()
    return VllmQwenForcedAligner(config)


def _require_config_value(config: ForcedAlignerConfig, name: str) -> Any:
    value = getattr(config, name)
    if value is None:
        raise ValueError(
            f"forced_aligner.{name} must be configured in "
            f"{QWEN_FORCED_ALIGNER_DEPLOY_CONFIG} or forced_aligner.deploy_config."
        )
    return value


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

    for timestamp in timestamp_ms:
        timestamp = max(0, int(round(timestamp)))
        if timestamp < last:
            timestamp = last
        fixed.append(timestamp)
        last = timestamp

    return fixed


class VllmQwenForcedAligner:
    """Managed client for a native-vLLM Qwen forced-aligner worker.

    The worker is launched by executing this module file by path. That keeps
    native vLLM pooling outputs isolated from the already-patched vLLM-Omni
    server process while avoiding a second source file.
    """

    def __init__(self, config: ForcedAlignerConfig) -> None:
        self.config = config
        self.model = config.resolved_model
        self.dtype = _require_config_value(config, "dtype")
        self.gpu_memory_utilization = _require_config_value(
            config,
            "gpu_memory_utilization",
        )
        self.max_model_len = _require_config_value(config, "max_model_len")
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = _require_config_value(config, "max_num_seqs")
        self.startup_timeout = _require_config_value(config, "startup_timeout")
        self.request_timeout = _require_config_value(config, "request_timeout")

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
        worker_path = Path(__file__).resolve()
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
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]
        if self.max_num_batched_tokens is not None:
            cmd.extend(
                [
                    "--max-num-batched-tokens",
                    str(self.max_num_batched_tokens),
                ]
            )
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


def _assert_clean_worker_import_state() -> None:
    if "vllm_omni" in sys.modules:
        raise RuntimeError("Forced aligner worker imported vllm_omni; native vLLM pooling schema isolation is broken.")


class QwenForcedAlignerWorker:
    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int | None,
        max_num_seqs: int,
    ) -> None:
        _assert_clean_worker_import_state()
        from vllm import LLM

        _assert_clean_worker_import_state()
        self.llm = LLM(
            model=model,
            runner="pooling",
            dtype=dtype,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens or max_model_len,
            max_num_seqs=max_num_seqs,
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
        audio_duration_ms = int(round(wav.size / float(sr) * 1000))
        return [
            {
                "word": units[i],
                "start_ms": min(timestamp_ms[i * 2], audio_duration_ms),
                "end_ms": min(max(timestamp_ms[i * 2], timestamp_ms[i * 2 + 1]), audio_duration_ms),
            }
            for i in range(len(units))
        ]


def _serve_qwen_worker(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    listener = Listener(str(socket_path), family="AF_UNIX")
    aligner: QwenForcedAlignerWorker | None = None

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
                        aligner = QwenForcedAlignerWorker(
                            model=args.model,
                            dtype=args.dtype,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len,
                            max_num_batched_tokens=args.max_num_batched_tokens,
                            max_num_seqs=args.max_num_seqs,
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


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM-Omni forced aligner worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    return parser.parse_args()


def _worker_main() -> int:
    args = _parse_worker_args()
    return _serve_qwen_worker(args)


_ALIGNER: ForcedAligner | None = None
_ALIGNER_CONFIG = ForcedAlignerConfig.disabled()
_ALIGNER_LOCK = asyncio.Lock()


def configure_forced_aligner(config: ForcedAlignerConfig) -> None:
    global _ALIGNER, _ALIGNER_CONFIG

    if config == _ALIGNER_CONFIG:
        return

    if _ALIGNER is not None:
        try:
            _ALIGNER.shutdown()
        finally:
            _ALIGNER = None
    _ALIGNER_CONFIG = config


async def get_forced_aligner(config: ForcedAlignerConfig | None = None) -> ForcedAligner:
    global _ALIGNER

    if config is not None:
        configure_forced_aligner(config)

    if _ALIGNER is not None:
        return _ALIGNER

    async with _ALIGNER_LOCK:
        if _ALIGNER is None:
            _ALIGNER = await asyncio.to_thread(create_forced_aligner, _ALIGNER_CONFIG)

    return _ALIGNER


async def align(
    audio_chunk: AudioChunk,
    text: str,
    sr: int,
    config: ForcedAlignerConfig | None = None,
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
        config:
            Optional server-level forced aligner configuration.

    Returns:
        Word/character timestamps. Returns an empty list when disabled.
    """
    aligner = await get_forced_aligner(config)
    return await aligner.align(audio_chunk, text, sr)


if __name__ == "__main__":
    raise SystemExit(_worker_main())
