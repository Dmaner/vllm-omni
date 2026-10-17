# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit, bounded fixed-conditioning state for one JING denoising call."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .attention import AttentionGroups
from .packing import JingPackedInputs

_REQUEST_CACHE_OPTION = "xgen_jing_request_cache"
_FIXED_MODULES = ("condition_proj", "token_refiner", "rope", "control.encoder")


def request_cache_enabled(model_config: Mapping[str, object]) -> bool:
    enabled = model_config.get(_REQUEST_CACHE_OPTION, False)
    if type(enabled) is not bool:
        raise TypeError(f"model_config.{_REQUEST_CACHE_OPTION} must be a boolean")
    return enabled


def _tensor_signature(tensor: torch.Tensor, *, mutable: bool = False) -> tuple[object, ...]:
    # Inference tensors have no version counter. The pipeline owns these fixed
    # inputs and never mutates them; identity/geometry checks still apply. For
    # normal tensors, also reject tracked in-place changes without a GPU read.
    version = None if mutable or tensor.is_inference() else tensor._version
    return (
        id(tensor),
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.dtype,
        tensor.device,
        version,
    )


def _packed_signature(packed: JingPackedInputs) -> tuple[object, ...]:
    fixed = (
        packed.text,
        packed.video_indices,
        packed.audio_indices,
        packed.text_indices,
        packed.position_ids,
        packed.token_tags,
        packed.owners,
        packed.is_media,
        packed.reference_rows,
    )
    return (
        *(_tensor_signature(tensor) for tensor in fixed),
        _tensor_signature(packed.video, mutable=True),
        _tensor_signature(packed.audio, mutable=True),
        tuple(packed.text_lengths),
        tuple(packed.audio_lengths),
        packed.output_video_offset,
        tuple(
            (_tensor_signature(camera), height, width, _tensor_signature(rows))
            for camera, height, width, rows in packed.controls
        ),
    )


def _weight_signature(model: nn.Module) -> tuple[object, ...]:
    entries = []
    for name in _FIXED_MODULES:
        module = model.get_submodule(name)
        entries.append((name, id(module), module.training))
        entries.extend((key, _tensor_signature(value)) for key, value in module.named_parameters())
        entries.extend((key, _tensor_signature(value)) for key, value in module.named_buffers())
    return tuple(entries)


@dataclass(frozen=True)
class JingFixedConditioning:
    refined_text: torch.Tensor
    freqs: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    camera: torch.Tensor
    camera_rows: torch.Tensor
    groups: AttentionGroups

    def tensor_bytes(self) -> int:
        tensors = (self.refined_text, self.freqs, self.rope_cos, self.rope_sin, self.camera, self.camera_rows)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors) + sum(
            tensor.numel() * tensor.element_size() for group in self.groups for tensor in group
        )


class JingRequestCache:
    """One entry, bound to one model and packed object; never stored on a model.

    Fixed tensors are owned and read-only within pipeline.diffuse. Reusing this
    object with another request/layout/model, after close, with gradients, or
    after a tracked fixed-input/weight mutation fails instead of reusing data.
    """

    def __init__(self, model: nn.Module, packed: JingPackedInputs) -> None:
        self._model: nn.Module | None = model
        self._packed: JingPackedInputs | None = packed
        self._packed_signature = _packed_signature(packed)
        self._weight_signature = _weight_signature(model)
        self._layout: tuple[int, ...] | None = None
        self._value: JingFixedConditioning | None = None
        self._hits = 0
        self._misses = 0
        self._peak_bytes = 0
        self._closed = False

    def get(self, model: nn.Module, packed: JingPackedInputs, layout: tuple[int, ...]) -> JingFixedConditioning | None:
        if self._closed or model is not self._model or packed is not self._packed:
            raise ValueError("JING request cache belongs to another request/model or has already closed")
        if torch.is_grad_enabled() or model.training:
            raise ValueError("JING request caching requires an eval model with gradients disabled")
        if self._packed_signature != _packed_signature(packed) or self._weight_signature != _weight_signature(model):
            raise ValueError("JING fixed conditioning or its projection weights changed within the request")
        if self._layout is None:
            self._layout = layout
        elif self._layout != layout:
            raise ValueError("JING request cache TP/SP geometry changed within the request")
        if self._value is None:
            self._misses += 1
        else:
            self._hits += 1
        return self._value

    def store(self, value: JingFixedConditioning) -> None:
        if self._closed or self._layout is None or self._value is not None:
            raise ValueError("JING request cache must store exactly one fixed-conditioning entry after a miss")
        self._value = value
        self._peak_bytes = value.tensor_bytes()

    def close(self) -> None:
        self._value = None
        self._model = None
        self._packed = None
        self._packed_signature = ()
        self._weight_signature = ()
        self._layout = None
        self._closed = True

    def scalar_stats(self) -> dict[str, int | bool]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "peak_cached_tensor_bytes": self._peak_bytes,
            "retained_tensor_bytes": 0 if self._value is None else self._peak_bytes,
            "closed": self._closed,
        }
