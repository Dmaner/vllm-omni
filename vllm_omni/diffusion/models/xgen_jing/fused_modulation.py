# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Indexed JING pointwise kernels with explicit eager BF16 rounding points."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _affine_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        indices_ptr,
        output_ptr,
        width: tl.constexpr,
        modulation_rows: tl.constexpr,
        x_row: tl.constexpr,
        x_col: tl.constexpr,
        scale_row: tl.constexpr,
        scale_col: tl.constexpr,
        shift_row: tl.constexpr,
        shift_col: tl.constexpr,
        index_stride: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = columns < width
        index = tl.load(indices_ptr + row * index_stride)
        valid_index = (index >= 0) & (index < modulation_rows)
        tl.device_assert(valid_index, "JING modulation index is outside its parameter table")
        # The safety mask also prevents an invalid index from reading memory
        # before the device assertion is reported to the caller.
        x = tl.load(x_ptr + row * x_row + columns * x_col, mask, 0).to(tl.float32)
        scale = tl.load(scale_ptr + index * scale_row + columns * scale_col, mask & valid_index, 0)
        shift = tl.load(shift_ptr + index * shift_row + columns * shift_col, mask & valid_index, 0)
        # Each cast is a required eager BF16 materialization boundary. In
        # particular, neither affine multiplication nor addition uses FMA.
        scale_one = (1.0 + scale.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        product = (x * scale_one).to(tl.bfloat16).to(tl.float32)
        value = (product + shift.to(tl.float32)).to(tl.bfloat16)
        tl.store(output_ptr + row * width + columns, value, mask)

    @triton.jit
    def _residual_kernel(
        residual_ptr,
        branch_ptr,
        gate_ptr,
        indices_ptr,
        output_ptr,
        width: tl.constexpr,
        modulation_rows: tl.constexpr,
        residual_row: tl.constexpr,
        residual_col: tl.constexpr,
        branch_row: tl.constexpr,
        branch_col: tl.constexpr,
        gate_row: tl.constexpr,
        gate_col: tl.constexpr,
        index_stride: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = columns < width
        index = tl.load(indices_ptr + row * index_stride)
        valid_index = (index >= 0) & (index < modulation_rows)
        tl.device_assert(valid_index, "JING modulation index is outside its parameter table")
        residual = tl.load(residual_ptr + row * residual_row + columns * residual_col, mask, 0).to(tl.float32)
        branch = tl.load(branch_ptr + row * branch_row + columns * branch_col, mask, 0).to(tl.float32)
        gate = tl.load(gate_ptr + index * gate_row + columns * gate_col, mask & valid_index, 0).to(tl.float32)
        product = (gate * branch).to(tl.bfloat16).to(tl.float32)
        value = (residual + product).to(tl.bfloat16)
        tl.store(output_ptr + row * width + columns, value, mask)


def _validate(
    x: torch.Tensor, indices: torch.Tensor, parameters: tuple[torch.Tensor, ...], branch: torch.Tensor | None = None
) -> tuple[int, int]:
    if not HAS_TRITON or torch.version.hip is not None or x.device.type != "cuda":
        raise ValueError("JING fused modulation requires CUDA with Triton")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise ValueError("JING fused modulation requires eager inference without gradients")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] == 0:
        raise ValueError("JING fused modulation requires a BF16 [tokens, hidden] tensor")
    if indices.ndim != 1 or indices.shape[0] != x.shape[0] or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("JING fused modulation indices must be one integer per token")
    if indices.device != x.device:
        raise ValueError("JING fused modulation indices must share the hidden-state device")
    if any(stride < 0 for value in (x, indices, *parameters) for stride in value.stride()):
        raise ValueError("JING fused modulation does not support negative tensor strides")
    parameter_shape = parameters[0].shape
    if len(parameter_shape) != 2 or parameter_shape[0] == 0 or parameter_shape[1] != x.shape[1]:
        raise ValueError("JING fused modulation parameters must have shape [modalities, hidden]")
    if any(p.shape != parameter_shape or p.dtype != torch.bfloat16 or p.device != x.device for p in parameters):
        raise ValueError("JING fused modulation parameters must share BF16 dtype, shape and device")
    if branch is not None and (
        branch.shape != x.shape
        or branch.dtype != x.dtype
        or branch.device != x.device
        or any(stride < 0 for stride in branch.stride())
    ):
        raise ValueError("JING fused residual branch must share the residual's shape, BF16 dtype and device")
    return x.shape


def indexed_affine(
    normalized: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Compute ``normalized * (1 + scale[indices]) + shift[indices]``.

    RMSNorm runs separately. Scale/shift remain small, possibly strided AdaLN
    views; no token-sized parameter gather is allocated. Every eager BF16
    rounding boundary is explicit, but bitwise equivalence still requires
    validation on the actual GPU/compiler stack before enabling this option.
    """
    rows, width = _validate(normalized, indices, (scale, shift))
    output = torch.empty((rows, width), dtype=normalized.dtype, device=normalized.device)
    if rows:
        _affine_kernel[(rows, triton.cdiv(width, 256))](
            normalized,
            scale,
            shift,
            indices,
            output,
            width,
            scale.shape[0],
            *normalized.stride(),
            *scale.stride(),
            *shift.stride(),
            indices.stride(0),
            256,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return output


def indexed_residual(
    residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Compute ``residual + gate[indices] * branch`` with two BF16 roundings."""
    rows, width = _validate(residual, indices, (gate,), branch)
    output = torch.empty((rows, width), dtype=residual.dtype, device=residual.device)
    if rows:
        _residual_kernel[(rows, triton.cdiv(width, 256))](
            residual,
            branch,
            gate,
            indices,
            output,
            width,
            gate.shape[0],
            *residual.stride(),
            *branch.stride(),
            *gate.stride(),
            indices.stride(0),
            256,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return output
