# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera-command embeddings and FiLM with the released JING tensor names."""

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

ControlSegment = tuple[torch.Tensor, int, int, torch.Tensor]


def control_vector(keys: str | Sequence[str] | set[str]) -> list[float]:
    """Return translation xyz and rotation pitch/yaw/roll for WASD/IJKL."""
    if isinstance(keys, str):
        commands = keys.lower().strip().split(",") if "," in keys else list(keys.lower().strip())
        commands = [command.strip() for command in commands]
    elif isinstance(keys, (list, tuple, set)):
        commands = list(keys)
    else:
        raise TypeError("JING camera control must be a key string or a sequence of keys")
    if any(
        not isinstance(command, str) or command.lower() not in "wasdijkl" or len(command) != 1 for command in commands
    ):
        raise ValueError("JING camera keys must be individual w/a/s/d/i/j/k/l keys")
    active = {command.lower() for command in commands}
    return [
        float("d" in active) - float("a" in active),
        0.0,
        float("w" in active) - float("s" in active),
        float("i" in active) - float("k" in active),
        float("l" in active) - float("j" in active),
        0.0,
    ]


class JingControlEncoder(nn.Module):
    def __init__(self, width: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.input_proj = nn.Linear(6 * pt, width, dtype=torch.bfloat16)
        # The released state includes the dense-control projection even though
        # the public key-command path supplies one vector per video frame.
        self.dense_input_proj = nn.Linear(6 * pt * ph * pw, width, dtype=torch.bfloat16)
        self.mlp1 = nn.Linear(width, width, dtype=torch.bfloat16)
        self.mlp2 = nn.Linear(width, width, dtype=torch.bfloat16)

    def forward(self, camera: torch.Tensor, height: int, width: int) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        if camera.ndim != 2 or camera.shape[-1] != 6 or camera.shape[0] % pt:
            raise ValueError("Camera controls must have shape [latent_frames, 6] aligned to temporal patches")
        if height % ph or width % pw:
            raise ValueError("Camera spatial dimensions must align to video patches")
        frames = self.input_proj(camera.reshape(-1, 6 * pt).to(self.input_proj.weight.dtype))
        frames = frames + self.mlp2(F.silu(self.mlp1(frames)))
        return frames.repeat_interleave((height // ph) * (width // pw), dim=0)


class JingCameraFiLM(nn.Module):
    def __init__(self, control_width: int, hidden_size: int) -> None:
        super().__init__()
        self.mlp1 = nn.Linear(control_width, control_width, dtype=torch.bfloat16)
        self.mlp2 = nn.Linear(control_width, control_width, dtype=torch.bfloat16)
        self.scale = nn.Linear(control_width, hidden_size, dtype=torch.bfloat16)
        self.shift = nn.Linear(control_width, hidden_size, dtype=torch.bfloat16)

    def forward(self, hidden: torch.Tensor, camera: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0:
            return hidden
        encoded = camera.to(self.mlp1.weight.dtype)
        encoded = encoded + self.mlp2(F.silu(self.mlp1(encoded)))
        selected = hidden.index_select(0, rows)
        modulated = selected * (1 + self.scale(encoded).to(hidden.dtype)) + self.shift(encoded).to(hidden.dtype)
        return hidden.index_copy(0, rows, modulated)


class JingControl(nn.Module):
    def __init__(
        self, config: Mapping[str, object], *, hidden_size: int, num_layers: int, patch_size: tuple[int, int, int]
    ) -> None:
        super().__init__()
        if config.get("enable") is not True or config.get("dim_in") != 6:
            raise ValueError("JING requires the released enabled six-dimensional camera-control configuration")
        width = config.get("dim_out")
        if type(width) is not int or width <= 0:
            raise ValueError("JING control dim_out must be a positive integer")
        self.encoder = JingControlEncoder(width, patch_size)
        self.block_injectors = nn.ModuleList([JingCameraFiLM(width, hidden_size) for _ in range(num_layers)])

    def forward(self, segments: Sequence[ControlSegment]) -> tuple[torch.Tensor, torch.Tensor]:
        if not segments:
            raise ValueError("JING requires camera controls for every generated video slice, including zero commands")
        features, positions = [], []
        for camera, height, width, rows in segments:
            encoded = self.encoder(camera, height, width)
            if len(encoded) != rows.numel():
                raise ValueError("Camera embeddings and generated video token indices must have equal lengths")
            features.append(encoded)
            positions.append(rows)
        return torch.cat(features), torch.cat(positions)
