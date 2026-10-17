# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Slice-local text visibility without allocating a quadratic attention mask."""

import torch
from torch.nn import functional as F

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

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
    def forward(
        self,
        hidden: torch.Tensor,
        *,
        freqs: torch.Tensor,
        groups: AttentionGroups,
        world_size: int,
        total_rows: int,
    ) -> torch.Tensor:
        projected, _ = self.qkv_proj(hidden)
        width = self.num_heads * self.head_dim
        q, k, v = (part.reshape(-1, self.num_heads, self.head_dim) for part in projected.split(width, dim=-1))
        q = self._apply_rope(self.q_norm(q), freqs)
        k = self._apply_rope(self.k_norm(k), freqs)
        # Q stays local while each rank needs K/V for global media and the
        # selected slice text. TP already partitions heads independently.
        k = gather_sequence_rows(k, world_size, total_rows)
        v = gather_sequence_rows(v, world_size, total_rows)
        output = torch.zeros_like(q)
        for query_rows, key_rows in groups:
            query = q.index_select(0, query_rows).transpose(0, 1).unsqueeze(0)
            key = k.index_select(0, key_rows).transpose(0, 1).unsqueeze(0)
            value = v.index_select(0, key_rows).transpose(0, 1).unsqueeze(0)
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            output.index_copy_(0, query_rows, attended.squeeze(0).transpose(0, 1))
        result, _ = self.out_proj(output.flatten(1))
        return result
