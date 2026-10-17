# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit CuTe FA4 for one already-selected JING visibility group."""

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

if not hasattr(torch.ops.vllm_omni, "xgen_jing_fa4_dense_attention"):

    @torch.library.custom_op("vllm_omni::xgen_jing_fa4_dense_attention", mutates_args=())
    def _jing_fa4_dense_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
        deterministic: bool,
    ) -> torch.Tensor:
        # Do not use utils.fa's platform-selected function: Hopper may resolve
        # that symbol to FA3. This model-specific route promises CuTe FA4.
        from flash_attn.cute import flash_attn_func

        kwargs = {"causal": False, "softmax_scale": softmax_scale}
        if deterministic:
            kwargs["deterministic"] = True
        result = flash_attn_func(query, key, value, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError("CuTe FA4 returned an invalid JING attention output")
        return output

    @_jing_fa4_dense_op.register_fake
    def _jing_fa4_dense_fake(query, key, value, softmax_scale, deterministic):
        return query.new_empty(query.shape)


_jing_fa4_dense_op = torch.ops.vllm_omni.xgen_jing_fa4_dense_attention


class JingFA4Attention(nn.Module):
    """BF16 BSND attention with no mask, dropout, padding or SP inside.

    JING selects each group's exact Q/K/V indices before this call. Grouping
    remains outside the kernel; concatenating groups as one dense sequence
    would violate slice-local text visibility. The opaque operation follows
    the shared FA4 backend's compile boundary, while pinning the CuTe API.
    """

    def __init__(self, num_heads: int, head_size: int, *, deterministic: bool) -> None:
        super().__init__()
        if not current_omni_platform.is_cuda():
            raise ValueError("JING CuTe FA4 requires CUDA")
        capability = current_omni_platform.get_device_capability()
        if capability is None or capability.major not in (9, 10, 11):
            raise ValueError("JING CuTe FA4 currently requires SM90, SM100 or SM110")
        if head_size != 128 or num_heads < 1:
            raise ValueError("JING CuTe FA4 currently requires positive local head count and head dimension 128")
        try:
            from flash_attn.cute import flash_attn_func
        except ImportError as exc:
            raise ImportError(
                "JING FLASH_ATTN requires CuTe FA4; no other attention kernel will be substituted"
            ) from exc
        if not callable(flash_attn_func):
            raise TypeError("flash_attn.cute.flash_attn_func is not callable")
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = head_size**-0.5
        self.deterministic = deterministic

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        for tensor in (query, key, value):
            if (
                tensor.ndim != 4
                or tensor.shape[0] != 1
                or tensor.shape[1] < 1
                or tensor.shape[2:] != (self.num_heads, self.head_size)
                or tensor.dtype != torch.bfloat16
                or tensor.device.type != "cuda"
                or tensor.device != query.device
            ):
                raise ValueError("JING CuTe FA4 expects nonempty BF16 [1, sequence, local_heads, 128] tensors")
        if key.shape != value.shape:
            raise ValueError("JING CuTe FA4 requires equal key/value geometry")
        return _jing_fa4_dense_op(query, key, value, self.softmax_scale, self.deterministic)
