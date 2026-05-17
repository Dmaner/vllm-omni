from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from threading import Lock
from typing import Any, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

_FORCED_ALIGNER_ENV = "VLLM_OMNI_FORCED_ALIGNER"


class ForcedAligner(Protocol):
    """Protocol for plugin-provided TTS forced aligners."""

    def align(
        self,
        audio: Any,
        text: str,
        sample_rate: int,
        audio_offset_ms: int = 0,
    ) -> Sequence[Any]:
        """Return word timestamp objects for one audio chunk."""


_aligner_factories: dict[str, Callable[[], ForcedAligner]] = {}
_loaded_aligners: dict[str, ForcedAligner] = {}
_default_aligner_name: str | None = None
_lock = Lock()


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("forced aligner name must be non-empty")
    return normalized


def register_forced_aligner(
    name: str,
    factory: Callable[[], ForcedAligner],
    *,
    set_default: bool = False,
) -> None:
    """Register a plugin-provided forced aligner factory.

    The factory is stored without being called. The aligner is constructed
    lazily on the first request that asks for word timestamps.
    """

    if not callable(factory):
        raise TypeError("forced aligner factory must be callable")

    normalized = _normalize_name(name)
    global _default_aligner_name
    with _lock:
        _aligner_factories[normalized] = factory
        _loaded_aligners.pop(normalized, None)
        if set_default or _default_aligner_name is None:
            _default_aligner_name = normalized
    logger.info("Registered forced aligner %s", normalized)


def get_forced_aligner(name: str | None = None) -> ForcedAligner | None:
    """Return a registered forced aligner, constructing it on first use."""

    explicit_name = name is not None
    if name is None:
        name = os.environ.get(_FORCED_ALIGNER_ENV) or _default_aligner_name
    if name is None:
        return None

    normalized = _normalize_name(name)
    with _lock:
        aligner = _loaded_aligners.get(normalized)
        if aligner is not None:
            return aligner

        factory = _aligner_factories.get(normalized)
        if factory is None:
            if explicit_name:
                raise ValueError(f"Forced aligner {normalized!r} is not registered.")
            logger.warning("Configured forced aligner %r is not registered.", normalized)
            return None

        aligner = factory()
        _loaded_aligners[normalized] = aligner
        return aligner


def align_words(
    *,
    audio: Any,
    text: str,
    sample_rate: int,
    audio_offset_ms: int = 0,
    aligner_name: str | None = None,
) -> list[dict[str, Any]]:
    """Align words for an audio chunk if a forced aligner is available."""

    aligner = get_forced_aligner(aligner_name)
    if aligner is None:
        return []

    return [_coerce_word_timestamp(item) for item in aligner.align(audio, text, sample_rate, audio_offset_ms)]


def _coerce_word_timestamp(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif is_dataclass(item) and not isinstance(item, type):
        data = asdict(item)
    elif isinstance(item, Mapping):
        data = dict(item)
    else:
        data = {
            "word": getattr(item, "word"),
            "start_ms": getattr(item, "start_ms"),
            "end_ms": getattr(item, "end_ms"),
        }

    return {
        "word": str(data["word"]),
        "start_ms": int(data["start_ms"]),
        "end_ms": int(data["end_ms"]),
    }


def _reset_forced_aligners_for_testing() -> None:
    global _default_aligner_name
    with _lock:
        _aligner_factories.clear()
        _loaded_aligners.clear()
        _default_aligner_name = None


__all__ = [
    "ForcedAligner",
    "align_words",
    "get_forced_aligner",
    "register_forced_aligner",
]
