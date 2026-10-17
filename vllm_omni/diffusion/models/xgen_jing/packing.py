# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Materialize a single JING request on the joint text/audio/video timeline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import _axis_from_sqrt_area
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

from .control import control_vector


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def chunk_lengths(num_frames: int, first_chunk_size: int = 2) -> tuple[int, list[int]]:
    """Align pixel frames to the H3 VAE and select the JING chunk boundaries."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, video_latent_num_frames

    _integer(num_frames, "num_frames", 1)
    _integer(first_chunk_size, "first_chunk_size")
    if first_chunk_size not in (0, 2):
        raise ValueError("first_chunk_size must be 0 or 2")
    aligned = align_num_frames(num_frames, frames_per_chunk=17, latents_per_chunk=5)
    latent_frames = video_latent_num_frames(aligned, frames_per_chunk=17, latents_per_chunk=5)
    groups, remainder = divmod(latent_frames, 5)
    if groups == 0 or remainder != 2:
        raise ValueError("JING needs at least 22 aligned pixel frames and 5*n+2 latent frames")
    lengths = [5] * groups
    if first_chunk_size:
        lengths[0] += 2
    else:
        lengths.append(2)
    return aligned, lengths


def _control_keys(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.split(",") if "," in value else list(value.strip())
    if not isinstance(value, (list, tuple, set)) or any(not isinstance(key, str) for key in value):
        raise ValueError("control must contain key names or a comma-separated key string")
    return sorted({key.strip().lower() for key in value if key.strip()})


def expand_slices(case: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    """Resolve per-chunk prompts and keyboard controls before encoding text."""
    _integer(count, "slice count", 1)
    if not isinstance(case, Mapping):
        raise ValueError("case must be a mapping")
    if "trajectory" in case:
        raise ValueError("trajectory is unsupported; specify chunks or slices")
    specifications = case.get("slices")
    if "chunks" in case:
        if "slices" in case or "control" in case:
            raise ValueError("chunks cannot coexist with slices or top-level control")
        chunks = case["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a nonempty list")
        specifications = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping) or set(chunk).difference(("prompt", "repeat", "control")):
                raise ValueError(f"chunks[{index}] accepts only prompt, repeat and control")
            repeats = _integer(chunk.get("repeat", 1), f"chunks[{index}].repeat", 1)
            commands = chunk.get("control", [""] * repeats)
            if not isinstance(commands, list) or len(commands) != repeats:
                raise ValueError(f"chunks[{index}].control needs {repeats} entries")
            if any(not isinstance(command, str) for command in commands):
                raise ValueError(f"chunks[{index}].control entries must be strings")
            specifications.extend(
                {"prompt": chunk.get("prompt", case.get("prompt")), "control": command} for command in commands
            )
    if specifications is None:
        specifications = [{"prompt": case.get("prompt"), "control": case.get("control", [])}] * count
    if not isinstance(specifications, list) or len(specifications) != count:
        raise ValueError(f"Expected {count} slice specifications")
    expanded = []
    for index, specification in enumerate(specifications):
        if not isinstance(specification, Mapping) or set(specification).difference(("prompt", "control")):
            raise ValueError(f"slices[{index}] accepts only prompt and control")
        prompt = specification.get("prompt", case.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"slices[{index}].prompt must be nonempty text")
        keys = _control_keys(specification.get("control", []))
        expanded.append({"prompt": prompt, "control": keys, "vector": control_vector(keys)})
    return expanded


def audio_endpoint(video_index: int) -> int:
    """Map a valid latent-video boundary to a per-channel audio boundary."""
    groups, remainder = divmod(_integer(video_index, "video_index"), 5)
    if remainder not in (0, 2):
        raise ValueError("A video boundary must have remainder 0 or 2 modulo 5")
    return 28 * groups + 4 * remainder


def _spatial_grid(height: int, width: int, patch: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    scale = math.sqrt(height * width)
    return _axis_from_sqrt_area(height, patch[1], scale), _axis_from_sqrt_area(width, patch[2], scale)


def _media_positions(
    frames: int, height: int, width: int, patch: tuple[int, int, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    h_axis, w_axis = _spatial_grid(height, width, patch)
    ticks = torch.arange(frames + 1)
    # Compute integer pixel starts first, avoiding cumulative floating-point drift.
    offsets = torch.tensor((0, 1, 5, 9, 13))
    clock = (17 * ticks.div(5, rounding_mode="floor") + offsets[ticks.remainder(5)]).double() * (5.0 / 3.0)
    video_positions = torch.cartesian_prod(clock[:-1], h_axis, w_axis)
    boundaries = [*range(0, frames - 1, 5), frames]
    audio_clock = torch.empty(audio_endpoint(frames), dtype=torch.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        first, last = audio_endpoint(left), audio_endpoint(right)
        fraction = torch.arange(last - first, dtype=torch.float64) * ((clock[right] - clock[left]) / (last - first))
        audio_clock[first:last] = clock[left] + fraction
    audio_positions = torch.zeros((2, audio_clock.numel(), 3), dtype=torch.float64)
    audio_positions[:, :, 0] = audio_clock
    audio_positions[0, :, 2] = w_axis[0]
    audio_positions[1, :, 2] = w_axis[-1]
    return video_positions.to(device), audio_positions.to(device)


@dataclass
class JingPackedInputs:
    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    owners: torch.Tensor
    is_media: torch.Tensor
    reference_rows: torch.Tensor
    text_lengths: tuple[int, ...]
    controls: list[tuple[torch.Tensor, int, int, torch.Tensor]]
    output_video_offset: int
    audio_lengths: tuple[int, ...]

    def timesteps(self, video_t: float, audio_t: float, reference_t: float) -> tuple[torch.Tensor, torch.Tensor]:
        times = torch.full_like(self.token_tags, float(video_t), dtype=torch.float32)
        times[self.audio_indices] = float(audio_t)
        times[self.reference_rows] = 1.0 - float(reference_t)
        return times.unique(sorted=True, return_inverse=True)

    def audio_channel_major(self) -> torch.Tensor:
        chunks = self.audio.squeeze(0).split(tuple(length * 2 for length in self.audio_lengths))
        channels = [chunk.reshape(2, length, chunk.shape[-1]) for chunk, length in zip(chunks, self.audio_lengths)]
        return torch.cat(channels, dim=1).flatten(0, 1)


@dataclass
class _Block:
    modality: str
    rows: torch.Tensor
    positions: torch.Tensor
    owner: int
    tags: torch.Tensor


def _latent(value: Any, rank: int, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != rank or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must be a nonempty rank-{rank} tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    return value.to(device)


def pack(
    payload: Mapping[str, Any], model_config: Mapping[str, Any], generation: Mapping[str, Any], device: torch.device
) -> JingPackedInputs:
    """Build compact modality tensors and their indices in the joint sequence."""
    device = torch.device(device)
    patch_values = model_config["patch_size"]
    if not isinstance(patch_values, Sequence) or len(patch_values) != 3:
        raise ValueError("patch_size must have three dimensions")
    patch = tuple(_integer(value, "patch_size", 1) for value in patch_values)
    if patch[0] != 1:
        raise ValueError("JING packing requires temporal patch size 1")
    video = _latent(payload["video"], 4, "video", device)
    audio = _latent(payload["audio"], 3, "audio", device)
    channels, frames, height, width = video.shape
    lengths = tuple(_integer(value, "chunk length", 1) for value in payload["lengths"])
    slices = payload["slices"]
    if not lengths or sum(lengths) != frames or len(slices) != len(lengths):
        raise ValueError("Chunk lengths and slices must cover the video exactly")
    if height % patch[1] or width % patch[2]:
        raise ValueError("Video dimensions must be divisible by the spatial patch size")
    if tuple(audio.shape) != (2, audio_endpoint(frames), model_config["audio_in_channels"]):
        raise ValueError("Audio shape must cover the complete stereo timeline with the configured latent width")
    reserved = _integer(generation["reserved_slots"], "reserved_slots")
    references = payload["references"]
    if len(references) > reserved:
        raise ValueError("Reference images exceed the number of reserved time slots")

    video_positions, audio_positions = _media_positions(frames, height, width, patch, device)
    blocks: list[_Block] = []
    control_blocks: list[tuple[int, torch.Tensor]] = []
    text_lengths: list[int] = []
    audio_lengths: list[int] = []
    spatial_rows = (height // patch[1]) * (width // patch[2])
    reference_count = 0

    def media_block(modality: str, rows: torch.Tensor, positions: torch.Tensor, owner: int) -> None:
        tags = torch.full((len(rows),), 2 if modality == "audio" else 0, device=device, dtype=torch.long)
        blocks.append(_Block(modality, rows, positions, owner, tags))

    for index, reference in enumerate(references):
        latent = _latent(reference, 4, f"references[{index}]", device)
        if latent.shape[:2] != (channels, 1) or latent.shape[2] % patch[1] or latent.shape[3] % patch[2]:
            raise ValueError("Each reference must be one video-latent frame with patch-aligned dimensions")
        rows = minimax_h3_patchify_video_latent(latent.unsqueeze(0), patch_size=patch)
        h_axis, w_axis = _spatial_grid(latent.shape[2], latent.shape[3], patch)
        reference_positions = torch.cartesian_prod(
            torch.tensor([index - reserved], dtype=torch.float64), h_axis, w_axis
        )
        media_block("video", rows, reference_positions.to(device), -1)
        reference_count += len(rows)

    start = 0
    text_width = None
    for owner, (length, specification) in enumerate(zip(lengths, slices)):
        stop = start + length
        audio_start, audio_stop = audio_endpoint(start), audio_endpoint(stop)
        prompt = specification["prompt"]
        if prompt not in payload["text_bank"]:
            raise ValueError(f"Missing text embeddings for slice {owner}")
        embeddings, tags = payload["text_bank"][prompt]
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim < 2 or not embeddings.is_floating_point():
            raise ValueError("Text embeddings must be a floating-point tensor with a feature dimension")
        embeddings = embeddings.to(device).reshape(-1, embeddings.shape[-1])
        if not len(embeddings) or (text_width is not None and embeddings.shape[-1] != text_width):
            raise ValueError("Text embeddings must be nonempty and share their feature width")
        text_width = embeddings.shape[-1]
        if not isinstance(tags, torch.Tensor) or tags.numel() != len(embeddings) or tags.is_floating_point():
            raise ValueError("Text tags must contain one integer tag per embedding row")
        text_positions = torch.zeros((len(embeddings), 3), device=device, dtype=torch.float64)
        text_positions[:, 0] = torch.arange(len(embeddings), device=device) - reserved - len(embeddings)
        blocks.append(
            _Block("text", embeddings, text_positions, owner, tags.to(device=device, dtype=torch.long).flatten())
        )
        text_lengths.append(len(embeddings))
        media_block(
            "audio",
            audio[:, audio_start:audio_stop].flatten(0, 1),
            audio_positions[:, audio_start:audio_stop].flatten(0, 1),
            owner,
        )
        audio_lengths.append(audio_stop - audio_start)
        rows = minimax_h3_patchify_video_latent(video[:, start:stop].unsqueeze(0), patch_size=patch)
        vector = torch.as_tensor(specification["vector"], device=device, dtype=video.dtype)
        if vector.shape != (6,) or not bool(torch.isfinite(vector).all()):
            raise ValueError("Every slice requires a finite six-component control vector")
        control_blocks.append((len(blocks), vector.unsqueeze(0).expand(length, 6)))
        media_block("video", rows, video_positions[start * spatial_rows : stop * spatial_rows], owner)
        start = stop

    indices: list[torch.Tensor] = []
    cursor = 0
    for block in blocks:
        indices.append(torch.arange(cursor, cursor + len(block.rows), device=device))
        cursor += len(block.rows)

    def modality_rows(name: str) -> torch.Tensor:
        return torch.cat([block.rows for block in blocks if block.modality == name]).unsqueeze(0)

    def modality_indices(name: str) -> torch.Tensor:
        return torch.cat([index for block, index in zip(blocks, indices) if block.modality == name])

    return JingPackedInputs(
        video=modality_rows("video"),
        audio=modality_rows("audio"),
        text=modality_rows("text"),
        video_indices=modality_indices("video"),
        audio_indices=modality_indices("audio"),
        text_indices=modality_indices("text"),
        position_ids=torch.cat([block.positions for block in blocks]),
        token_tags=torch.cat([block.tags for block in blocks]),
        owners=torch.cat([torch.full_like(index, block.owner) for block, index in zip(blocks, indices)]),
        is_media=torch.cat(
            [
                torch.full_like(index, block.modality != "text", dtype=torch.bool)
                for block, index in zip(blocks, indices)
            ]
        ),
        reference_rows=torch.arange(reference_count, device=device),
        text_lengths=tuple(text_lengths),
        controls=[(raw, height, width, indices[index]) for index, raw in control_blocks],
        output_video_offset=reference_count,
        audio_lengths=tuple(audio_lengths),
    )


Packed = JingPackedInputs
