# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Slice-local text visibility without allocating a quadratic attention mask."""

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention, MiniMaxH3DiTArchConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

AttentionGroups = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def build_attention_groups(
    owners: torch.Tensor,
    is_media: torch.Tensor,
    *,
    local_start: int = 0,
    local_length: int | None = None,
) -> AttentionGroups:
    """Queries see all media rows plus text rows with their own slice owner.

    Reference images have owner -1, so their queries see media only. Aligned
    padding has no owner and is excluded from both query and key indices.
    """
    if owners.ndim != 1 or owners.shape != is_media.shape or is_media.dtype != torch.bool:
        raise ValueError("JING attention owners and media flags must be equal-length one-dimensional tensors")
    end = len(owners) if local_length is None else min(local_start + local_length, len(owners))
    local_owners = owners[local_start:end]
    groups = []
    for owner in local_owners.unique().tolist():
        query_rows = torch.nonzero(local_owners == owner, as_tuple=True)[0]
        key_rows = torch.nonzero(is_media | (owners == owner), as_tuple=True)[0]
        groups.append((query_rows, key_rows))
    return tuple(groups)


def gather_sequence_rows(rows: torch.Tensor, world_size: int, total: int) -> torch.Tensor:
    if world_size == 1:
        return rows[:total]
    group = get_sp_group()
    if group.world_size != world_size:
        raise RuntimeError("JING sequence-parallel group does not match the configured AllGather degree")
    return group.all_gather(rows.contiguous(), dim=0)[:total]


class JingAttention(MiniMaxH3Attention):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: "QuantizationConfig | None",
        *,
        prefix: str,
        skip_sequence_parallel: bool = False,
    ) -> None:
        super().__init__(arch, quant_config, prefix=prefix, skip_sequence_parallel=skip_sequence_parallel)
        backend = self.attention.attn_backend.get_name()
        self.jing_fa4 = None
        if backend == "FLASH_ATTN":
            from .fa4 import JingFA4Attention

            config = get_current_diffusion_config_or_none()
            deterministic = bool(getattr(config, "fa_deterministic", False))
            self.jing_fa4 = JingFA4Attention(self.num_heads, self.head_dim, deterministic=deterministic)
        elif backend not in {"TORCH_SDPA", "SDPA"}:
            raise ValueError(f"Unsupported JING attention backend: {backend}")
        self.jing_attention_kernel = "torch_sdpa" if self.jing_fa4 is None else "flash_attn.cute.flash_attn_func"

    def _apply_cached_rope(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cos, sin = rotary
        expected = (x.shape[0], self.rot_dim)
        if cos.shape != expected or sin.shape != expected or cos.dtype != x.dtype or sin.dtype != x.dtype:
            raise ValueError("JING cached RoPE must match the original activation dtype and local row geometry")
        if cos.device != x.device or sin.device != x.device:
            raise ValueError("JING cached RoPE and activations must share a device")
        # Same native RotaryEmbedding dispatch and concat as H3 _apply_rope;
        # only the immutable cos/sin construction has moved out of the blocks.
        rotated = self.rope(x[..., : self.rot_dim], cos, sin)
        return torch.cat((rotated, x[..., self.rot_dim :]), dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        freqs: torch.Tensor,
        groups: AttentionGroups,
        world_size: int,
        total_rows: int,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected, _ = self.qkv_proj(hidden)
        width = self.num_heads * self.head_dim
        q, k, v = (part.reshape(-1, self.num_heads, self.head_dim) for part in projected.split(width, dim=-1))
        if rotary is None:
            q = self._apply_rope(self.q_norm(q), freqs)
            k = self._apply_rope(self.k_norm(k), freqs)
        else:
            q = self._apply_cached_rope(self.q_norm(q), rotary)
            k = self._apply_cached_rope(self.k_norm(k), rotary)
        # Q stays local while each rank needs K/V for global media and the
        # selected slice text. TP already partitions heads independently.
        k = gather_sequence_rows(k, world_size, total_rows)
        v = gather_sequence_rows(v, world_size, total_rows)
        output = torch.zeros_like(q)
        for query_rows, key_rows in groups:
            if self.jing_fa4 is None:
                query = q.index_select(0, query_rows).transpose(0, 1).unsqueeze(0)
                key = k.index_select(0, key_rows).transpose(0, 1).unsqueeze(0)
                value = v.index_select(0, key_rows).transpose(0, 1).unsqueeze(0)
                attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                output.index_copy_(0, query_rows, attended.squeeze(0).transpose(0, 1))
            else:
                # Each visibility group is one dense rectangular attention:
                # local Q rows, global media plus that slice's text K/V rows.
                # Keep BSND and run no extra SP operation inside the kernel.
                attended = self.jing_fa4(
                    q.index_select(0, query_rows).unsqueeze(0),
                    k.index_select(0, key_rows).unsqueeze(0),
                    v.index_select(0, key_rows).unsqueeze(0),
                )
                output.index_copy_(0, query_rows, attended.squeeze(0))
        result, _ = self.out_proj(output.flatten(1))
        return result
