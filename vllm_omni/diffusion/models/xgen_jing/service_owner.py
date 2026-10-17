# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch import nn

from vllm_omni.diffusion.cancellation import request_cancellation_scope
from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.platforms import current_omni_platform

_T = TypeVar("_T")
_WIRE_DTYPES = {
    str(dtype): dtype
    for dtype in (torch.bfloat16, torch.float32, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
}


class JingRemoteTextConditioner(nn.Module):
    """Weight-free handle for an encoder executed on another DiT rank.

    The explicit text-encoder offload selector requires a declared component.
    Use the existing unloaded/on-demand component protocol; invoking its local
    forward is a routing error, never a substitute encoder implementation.
    """

    is_loaded = False

    def load_to_device(self) -> None:
        # This rank owns no encoder tensor to transfer.
        return None

    def offload_to_cpu(self) -> None:
        # Generic backend setup/cleanup may call this on the unloaded handle.
        return None

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("A non-owner JING rank cannot execute the text encoder")


class JingServiceGroup:
    """One owner inside the already-validated, single-request TP x SP group.

    B's pipeline `_is_output_rank()` validates the real parallel groups before
    this object is constructed. No new process group is created. Python/GPU
    allocation failures vote on Gloo before peers enter tensor broadcasts.
    A dead process or broken communicator still requires the executor timeout
    and teardown; collective failures are not recoverable local fallbacks.
    """

    def __init__(self, *, is_owner: bool, device: torch.device):
        self.is_owner = is_owner
        self.device = device
        self.group = get_world_group() if torch.distributed.is_initialized() else None
        self.owner_rank = None
        self.owner_index = 0
        self.owner_call_counts: dict[str, int] = {}
        self.conditioning_broadcast_count = 0
        self.last_conditioning: dict[str, Any] | None = None
        if self.group is None:
            if not is_owner:
                raise RuntimeError("A standalone JING service must be its output owner")
            return
        owners: list[bool | None] = [None] * self.group.world_size
        torch.distributed.all_gather_object(owners, is_owner, group=self.group.cpu_group)
        indices = [index for index, value in enumerate(owners) if value is True]
        if len(indices) != 1:
            raise RuntimeError(f"JING requires exactly one service/output owner per TP/SP group: {owners}")
        self.owner_index = indices[0]
        self.owner_rank = self.group.ranks[self.owner_index]

    def vote(self, phase: str, error: BaseException | None) -> None:
        status = {
            "phase": phase,
            "error": None if error is None else str(error),
            "error_type": None if error is None else type(error).__name__,
            "aborted": isinstance(error, DiffusionRequestAbortedError),
        }
        statuses = [status]
        if self.group is not None:
            statuses = [None] * self.group.world_size
            torch.distributed.all_gather_object(statuses, status, group=self.group.cpu_group)
        if any(item is None or item.get("phase") != phase for item in statuses):
            raise RuntimeError(f"JING service collective sequence mismatch at {phase}: {statuses}")
        failures = {index: item for index, item in enumerate(statuses) if item["error"] is not None}
        if failures:
            # Healthy waiting ranks report no exception. Preserve cancellation
            # only when every reported failure is an abort; never hide a mixed
            # allocation/compute failure behind the cancelled-request outcome.
            if all(item["aborted"] for item in failures.values()):
                raise DiffusionRequestAbortedError(f"JING owner service cancelled at {phase}") from error
            raise RuntimeError(f"JING owner service failed at {phase}: {failures}") from error

    def run_owner(self, phase: str, function: Callable[[], _T]) -> _T | None:
        result = None
        error = None
        if self.is_owner:
            try:
                self.owner_call_counts[phase] = self.owner_call_counts.get(phase, 0) + 1
                # The original encode/decode helpers call cancellation checks
                # containing a world collective. Peers wait at `vote` here,
                # so disable only these inner checks and restore the context
                # before the all-rank outer cancellation checkpoint. The public
                # API has no local-only check: cancellation is deliberately
                # deferred until this complete TE/VAE stage returns.
                with request_cancellation_scope((), enabled=False):
                    result = function()
                current_omni_platform.synchronize()
            except BaseException as exc:
                # Every live rank must reach the same error decision, including
                # a partially allocated owner stage; never fabricate outputs.
                error = exc
        self.vote(phase, error)
        return result

    def conditioning(self, case: dict[str, Any], encode: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        value = self.run_owner("encode_conditioning", encode)
        tensors: list[torch.Tensor] = []
        packet = None
        error = None
        try:
            if self.is_owner:
                if value is None:
                    raise RuntimeError("The JING encoder owner returned no conditioning")
                prompts = list(value["text_bank"])
                for prompt in prompts:
                    embeddings, tags = value["text_bank"][prompt]
                    if embeddings.dtype != torch.bfloat16 or tags.dtype not in (
                        torch.int64,
                        torch.int32,
                        torch.int16,
                        torch.int8,
                        torch.uint8,
                    ):
                        raise TypeError("JING conditioning must retain BF16 embeddings and integer token tags")
                    tensors.extend((embeddings, tags))
                references = value["references"]
                if any(tensor.dtype != torch.float32 for tensor in references):
                    raise TypeError("JING reference latents must retain FP32")
                tensors.extend(references)
                if any(tensor.layout != torch.strided for tensor in tensors):
                    raise TypeError("JING conditioning broadcast requires dense tensors")
                tensors = [tensor.detach().to(self.device).contiguous() for tensor in tensors]
                packet = {
                    "prompts": prompts,
                    "reference_count": len(references),
                    "specifications": [(list(tensor.shape), str(tensor.dtype)) for tensor in tensors],
                }
                current_omni_platform.synchronize()
        except BaseException as exc:
            error = exc
        self.vote("conditioning_describe", error)
        if self.group is not None:
            packet = self.group.broadcast_object(packet, src=self.owner_index)
        error = None
        try:
            if not isinstance(packet, dict):
                raise TypeError("Missing JING conditioning metadata")
            prompts = packet["prompts"]
            expected_prompts = list(dict.fromkeys(item["prompt"] for item in case["slices"]))
            if prompts != expected_prompts or packet["reference_count"] != len(case["images"]):
                raise ValueError("JING request metadata differs across service ranks")
            specs = packet["specifications"]
            if len(specs) != 2 * len(prompts) + packet["reference_count"]:
                raise ValueError("Incomplete JING conditioning tensor metadata")
            for shape, dtype_name in specs:
                if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
                    raise ValueError(f"Invalid JING conditioning shape: {shape}")
                if dtype_name not in _WIRE_DTYPES:
                    raise TypeError(f"Unsupported JING conditioning dtype: {dtype_name}")
            if not self.is_owner:
                # Allocate all destinations before any NCCL broadcast. OOM on
                # one receiver must be shared with every peer at the Gloo vote.
                for shape, dtype_name in specs:
                    tensors.append(torch.empty(shape, dtype=_WIRE_DTYPES[dtype_name], device=self.device))
        except BaseException as exc:
            error = exc
        self.vote("conditioning_allocate", error)
        if self.group is not None:
            for tensor in tensors:
                torch.distributed.broadcast(tensor, src=self.owner_rank, group=self.group.device_group)
        result = None
        error = None
        try:
            current_omni_platform.synchronize()
            text_bank = {prompt: (tensors[2 * i], tensors[2 * i + 1]) for i, prompt in enumerate(prompts)}
            result = {**case, "text_bank": text_bank, "references": tensors[2 * len(prompts) :]}
        except BaseException as exc:
            error = exc
        self.vote("conditioning_ready", error)
        if result is None:
            raise RuntimeError("JING conditioning transport produced no result")
        self.conditioning_broadcast_count += 1
        self.last_conditioning = {
            "prompt_count": len(prompts),
            "reference_count": packet["reference_count"],
            "tensor_specifications": packet["specifications"],
            "tensor_payload_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
        }
        return result
