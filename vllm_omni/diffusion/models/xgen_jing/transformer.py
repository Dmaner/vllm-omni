# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native H3 modules with JING's segmented text and camera-conditioned FFNs."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_rank
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import MiniMaxH3RuntimeAdalnCache
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
    MiniMaxH3FinalLayer,
    MiniMaxH3MLP,
    MiniMaxH3Rope,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenRefiner,
)
from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, OffloadStrategy, resolve_offload
from vllm_omni.platforms import current_omni_platform

from .attention import AttentionGroups, JingAttention, build_attention_groups, gather_sequence_rows
from .control import JingCameraFiLM, JingControl
from .fused_modulation import indexed_affine, indexed_residual
from .packing import JingPackedInputs
from .request_cache import JingFixedConditioning, JingRequestCache

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig


class JingDiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: "QuantizationConfig | None",
        *,
        prefix: str,
        adaln_cache: MiniMaxH3RuntimeAdalnCache | None = None,
        fused_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.fused_modulation = fused_modulation
        self.norm1 = RMSNorm(arch.hidden_size, eps=arch.norm_eps, dtype=torch.bfloat16)
        self.norm2 = RMSNorm(arch.hidden_size, eps=arch.norm_eps, dtype=torch.bfloat16)
        self.attn = JingAttention(arch, quant_config, prefix=f"{prefix}.attn", skip_sequence_parallel=True)
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")
        # Shift/scale/gate projections retain the checkpoint precision.
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            None,
            expand_ratio=6,
            modality_num=3,
            prefix=f"{prefix}.adaln_proj",
            adaln_cache=adaln_cache,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        time_embedding: torch.Tensor,
        modality_indices: torch.Tensor,
        freqs: torch.Tensor,
        groups: AttentionGroups,
        world_size: int,
        total_rows: int,
        camera: torch.Tensor,
        camera_rows: torch.Tensor,
        injector: JingCameraFiLM,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.fused_modulation:
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_proj(time_embedding)
            attended = self.attn(
                indexed_affine(self.norm1(hidden), scale_a, shift_a, modality_indices),
                freqs=freqs,
                groups=groups,
                world_size=world_size,
                total_rows=total_rows,
                rotary=rotary,
            )
            hidden = indexed_residual(hidden, attended, gate_a, modality_indices)
            # Preserve camera FiLM between the attention residual and norm2;
            # the residual itself must continue to use the pre-FiLM hidden.
            conditioned = injector(hidden, camera, camera_rows)
            branch = self.mlp(indexed_affine(self.norm2(conditioned), scale_f, shift_f, modality_indices))
            return indexed_residual(hidden, branch, gate_f, modality_indices)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            values.index_select(0, modality_indices) for values in self.adaln_proj(time_embedding)
        )
        attended = self.attn(
            self.norm1(hidden) * (1 + scale_a) + shift_a,
            freqs=freqs,
            groups=groups,
            world_size=world_size,
            total_rows=total_rows,
            rotary=rotary,
        )
        hidden = hidden + gate_a * attended
        # FiLM conditions the FFN input; the residual retains the attention
        # result, rather than accumulating the camera shift as a residual.
        conditioned = injector(hidden, camera, camera_rows)
        return hidden + gate_f * self.mlp(self.norm2(conditioned) * (1 + scale_f) + shift_f)


class XGenJingDiTModel(MiniMaxH3DiTModel):
    _repeated_blocks = ["JingDiTBlock"]
    _sp_plan = {}  # JING owns its row slicing and AllGather-KV collectives.
    _cache_dit_adapter_config = None
    _hsdp_ignored_modules = [*MiniMaxH3DiTModel._hsdp_ignored_modules, "control"]
    host_weight_restore_contract = FinalLayoutModelContract(implementation_id="xgen-jing-dit", version="1")

    def __init__(
        self,
        od_config: "OmniDiffusionConfig",
        quant_config: "QuantizationConfig | None" = None,
        *,
        diffusers_weights: bool = True,
    ) -> None:
        # Construct the native H3 primitives once with our block type. Calling
        # the H3 constructor and replacing its blocks would allocate a second
        # full transformer while the first one is still resident.
        nn.Module.__init__(self)
        self._validate_quant_config(quant_config)
        if not diffusers_weights:
            raise ValueError("JING requires its released Diffusers-format transformer checkpoint")
        parallel = od_config.parallel_config
        if parallel.ulysses_degree != 1 or parallel.ring_degree != 1:
            raise ValueError("JING supports TP and pure AllGather-KV SP; set ulysses_degree=ring_degree=1")
        if getattr(parallel, "cfg_parallel_size", 1) != 1:
            raise ValueError("The four-step JING checkpoint does not support CFG parallelism")
        tf_config = od_config.tf_model_config
        config = tf_config.to_dict() if hasattr(tf_config, "to_dict") else dict(tf_config)
        self.arch = arch = MiniMaxH3DiTArchConfig.from_mapping(config)
        self._validate_tp_config(arch=arch, tp_size=get_tensor_model_parallel_world_size())
        attention_config = od_config.diffusion_attention_config
        for role, category in (("self", None), ("minimax_h3.token_refiner", "self")):
            spec, source = attention_config.resolve_with_source(role=role, role_category=category)
            allowed = {"TORCH_SDPA", "FLASH_ATTN"} if role == "self" else {"TORCH_SDPA"}
            if spec is None or spec.backend.upper() not in allowed or spec.quant is not None:
                raise ValueError(
                    f"XGEN-JING requires an explicit backend in {sorted(allowed)} without attention quantization "
                    f"for role {role!r} (resolved from {source or 'platform default'}). "
                    "FLASH_ATTN on JING's main blocks means CuTe FA4; the token refiner must remain TORCH_SDPA."
                )
            if spec.backend.upper() == "FLASH_ATTN" and spec.backend_kwargs():
                raise ValueError("JING CuTe FA4 does not accept backend-specific options")
        self.od_config, self.parallel_config = od_config, parallel
        self._diffusers_weights = True
        self._rope_theta = float(config.get("rope_theta", 10000.0))
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self.fused_modulation = od_config.model_config.get("xgen_jing_fused_modulation", False)
        if type(self.fused_modulation) is not bool:
            raise TypeError("model_config.xgen_jing_fused_modulation must be a boolean")
        if self.fused_modulation:
            if not HAS_TRITON or not current_omni_platform.is_cuda() or not od_config.enforce_eager:
                raise ValueError("XGEN-JING fused modulation requires eager CUDA inference with Triton")
            if od_config.dtype != torch.bfloat16:
                raise ValueError("XGEN-JING fused modulation requires BF16 activations and modulation parameters")
        cache_enabled = od_config.model_config.get("xgen_jing_adaln_cache", False)
        cache_bytes = od_config.model_config.get("xgen_jing_adaln_cache_max_bytes", 256 * 1024**2)
        if type(cache_enabled) is not bool:
            raise TypeError("model_config.xgen_jing_adaln_cache must be a boolean")
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("model_config.xgen_jing_adaln_cache_max_bytes must be a nonnegative integer")
        offload_adaln = od_config.model_config.get("xgen_jing_adaln_cpu_offload", False)
        if type(offload_adaln) is not bool:
            raise TypeError("model_config.xgen_jing_adaln_cpu_offload must be a boolean")
        if offload_adaln:
            offload = resolve_offload(od_config)
            if not current_omni_platform.is_cuda() or not od_config.enforce_eager:
                raise ValueError("XGEN-JING AdaLN CPU offload requires eager CUDA inference")
            if od_config.dtype != torch.bfloat16:
                raise ValueError("XGEN-JING AdaLN CPU offload requires BF16 modulation weights")
            if parallel.use_hsdp or (offload.strategy is not OffloadStrategy.NONE and offload.offloads(DIT_COMPONENT)):
                raise ValueError("XGEN-JING AdaLN CPU offload cannot be combined with HSDP or DiT offload")
            if od_config.diffusion_load_format != "default":
                raise ValueError("XGEN-JING AdaLN CPU offload requires the native default load format")
            if getattr(od_config, "host_weight_runtime_mode", "disabled") != "disabled":
                raise ValueError("XGEN-JING AdaLN CPU offload does not yet support Host Weight Runtime restores")
        # Reuse the exact H3 cache: actual embedding bytes, weight/storage
        # versions and numerical settings identify entries. TP ranks vote on
        # hits before any rank skips the projection's gather collective.
        # All JING AdaLN constructors explicitly receive quant_config=None;
        # main-block attention/MLP online FP8 does not quantize these weights.
        # The two switches are independent: offload without caching stages the
        # original projection on every call and may trade latency for memory.
        self.adaln_cache = (
            MiniMaxH3RuntimeAdalnCache(max_bytes=cache_bytes if cache_enabled else 0, offload_weights=offload_adaln)
            if cache_enabled or offload_adaln
            else None
        )
        self.vsa_gates_enabled = False
        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="video_patch_proj",
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="audio_patch_proj",
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="condition_proj",
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch, None, prefix="token_refiner")
        self.blocks = nn.ModuleList(
            [
                JingDiTBlock(
                    arch,
                    quant_config,
                    prefix=f"blocks.{index}",
                    adaln_cache=self.adaln_cache,
                    fused_modulation=self.fused_modulation,
                )
                for index in range(arch.num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(arch, None, prefix="final_layer", adaln_cache=self.adaln_cache)
        control_config = config.get("control_config")
        if not isinstance(control_config, Mapping):
            raise ValueError("Missing JING control_config; use the JING transformer config rather than base H3")
        self.control = JingControl(
            control_config, hidden_size=arch.hidden_size, num_layers=arch.num_layers, patch_size=arch.patch_size
        )
        self._online_fp8_shapes: dict[str, tuple[int, ...]] = {}
        if quant_config is not None:
            from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod

            for index, block in enumerate(self.blocks):
                for suffix, layer in (
                    ("attn.qkv_proj", block.attn.qkv_proj),
                    ("attn.out_proj", block.attn.out_proj),
                    ("mlp.fc1", block.mlp.fc1),
                    ("mlp.fc2", block.mlp.fc2),
                ):
                    method = layer.quant_method
                    if isinstance(method, Fp8PerTensorOnlineLinearMethod):
                        if method.use_marlin:
                            raise ValueError(
                                "JING online FP8 requires a native FP8 linear kernel, not Marlin repacking"
                            )
                        self._online_fp8_shapes[f"blocks.{index}.{suffix}"] = tuple(layer.weight.shape)
        # Online FP8 adds scales while consuming the checkpoint stream. Keep
        # the original targets so generated parameters cannot hide a missing
        # source tensor or become mistaken for checkpoint requirements.
        self._checkpoint_parameter_names = frozenset(name for name, _ in self.named_parameters())
        self._mark_missing_params_required()

    @staticmethod
    def _validate_quant_config(quant_config: "QuantizationConfig | None") -> None:
        if quant_config is None:
            return
        from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
        from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

        if (
            not isinstance(quant_config, OnlineQuantizationConfig)
            or quant_config.args.targets is not None
            or quant_config.args.linear is None
            or quant_config.args.linear.weight != kFp8StaticTensorSym
            or quant_config.args.linear.activation is not None
        ):
            raise ValueError(
                "JING supports only online per-tensor FP8 with dynamic activations for main-block attention/MLP; "
                "use quantization='fp8' with optional ignored_layers"
            )

    def _validate_online_fp8_weights(self) -> set[str]:
        """Return only scales generated by a completed, supported conversion."""
        generated: set[str] = set()
        for name, checkpoint_shape in self._online_fp8_shapes.items():
            layer = self.get_submodule(name)
            weight = layer.weight
            scale = getattr(layer, "weight_scale", None)
            if (
                not getattr(layer, "_already_called_process_weights_after_loading", False)
                or weight.is_meta
                or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                or tuple(weight.shape) != checkpoint_shape[::-1]
                or not isinstance(scale, nn.Parameter)
                or scale.is_meta
                or scale.dtype != torch.float32
                or scale.numel() != 1
                or not bool((torch.isfinite(scale) & (scale > 0)).all().item())
                or getattr(layer, "input_scale", None) is not None
            ):
                raise ValueError(f"Incomplete or invalid JING online FP8 conversion: {name}")
            generated.add(f"{name}.weight_scale")
        return generated

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        camera_parameters = dict(self.control.named_parameters(prefix="control"))
        loaded_camera: set[str] = set()
        source_names: set[str] = set()

        def base_weights():
            for name, weight in weights:
                if name in source_names:
                    raise ValueError(f"Duplicate JING checkpoint tensor: {name}")
                source_names.add(name)
                if name.startswith("control."):
                    if name not in camera_parameters:
                        raise ValueError(f"Unknown JING camera parameter: {name}")
                    default_weight_loader(camera_parameters[name], weight)
                    loaded_camera.add(name)
                else:
                    yield name, weight

        loaded = super().load_weights(base_weights()) | loaded_camera
        missing = self._checkpoint_parameter_names - loaded
        if missing:
            raise ValueError(f"Incomplete JING checkpoint; missing {len(missing)} tensors: {sorted(missing)[:8]}")
        if self._online_fp8_shapes:
            from vllm.model_executor.model_loader.reload.layerwise import finalize_layerwise_processing

            # Most linears finish in their weight_loader. Finalize stragglers
            # only after verifying the real checkpoint is complete. The outer
            # diffusion loader can safely repeat this native, idempotent step.
            finalize_layerwise_processing(self, model_config=None)
        generated = self._validate_online_fp8_weights()
        unexpected = set(dict(self.named_parameters())) - loaded - generated
        if unexpected:
            raise ValueError(f"Uninitialized JING parameters after loading: {sorted(unexpected)[:8]}")
        self.post_load_weights()
        return loaded

    def post_load_weights(self) -> None:
        super().post_load_weights()
        if self.adaln_cache is not None and self.adaln_cache.offload_weights:
            projections = [*(block.adaln_proj for block in self.blocks), self.final_layer.adaln_proj]
            for projection in projections:
                for name, parameter in projection.linear.named_parameters():
                    if parameter.device.type != "cpu" or parameter.dtype != torch.bfloat16:
                        raise ValueError(f"{projection._cache_name}.{name} must remain CPU BF16 with AdaLN offload")

    def _prepare_fixed_conditioning(
        self,
        packed: JingPackedInputs,
        *,
        start: int,
        end: int,
        padding: int,
        local_length: int,
    ) -> JingFixedConditioning:
        # Keep the full packed GEMM and original per-slice refiner batch shapes.
        projected_text, _ = self.condition_proj(packed.text[0].to(torch.bfloat16))
        refined = []
        for text in projected_text.split(packed.text_lengths):
            boundaries = torch.tensor([0, len(text)], dtype=torch.int32, device=text.device)
            refined.append(self.token_refiner(text, cu_seqlens=boundaries, max_seqlen=len(text)))
        positions = F.pad(packed.position_ids, (0, 0, 0, padding))[start:end]
        freqs = self.rope(positions.unsqueeze(0))
        camera, camera_rows = self.control(packed.controls)
        local_camera = (camera_rows >= start) & (camera_rows < end)
        camera, camera_rows = camera[local_camera], camera_rows[local_camera] - start
        groups = build_attention_groups(packed.owners, packed.is_media, local_start=start, local_length=local_length)
        # H3 casts both trigonometric outputs to q/k's BF16 dtype before the
        # same native rotary operator. Do not change the dispatch or math.
        return JingFixedConditioning(
            refined_text=torch.cat(refined),
            freqs=freqs,
            rope_cos=torch.cos(freqs).to(projected_text.dtype),
            rope_sin=torch.sin(freqs).to(projected_text.dtype),
            camera=camera,
            camera_rows=camera_rows,
            groups=groups,
        )

    def forward(
        self,
        packed: JingPackedInputs,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        attention_groups: AttentionGroups | None = None,
        *,
        request_cache: JingRequestCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_rows = packed.token_tags.numel()
        if timestep_indices.shape != (total_rows,):
            raise ValueError("JING timestep indices must cover every packed row")
        if sum(packed.text_lengths) != packed.text.shape[1] or any(n <= 0 for n in packed.text_lengths):
            raise ValueError("JING text_lengths must partition the nonempty slice texts")
        world_size = int(self.parallel_config.allgather_degree)
        rank = get_sequence_parallel_rank() if world_size > 1 else 0
        local_length = (total_rows + world_size - 1) // world_size
        start, end = rank * local_length, (rank + 1) * local_length
        padding = local_length * world_size - total_rows
        video, _ = self.video_patch_proj(packed.video[0].float())
        audio, _ = self.audio_patch_proj(packed.audio[0].float())
        if request_cache is None:
            projected_text, _ = self.condition_proj(packed.text[0].to(torch.bfloat16))
            refined = []
            for text in projected_text.split(packed.text_lengths):
                boundaries = torch.tensor([0, len(text)], dtype=torch.int32, device=text.device)
                refined.append(self.token_refiner(text, cu_seqlens=boundaries, max_seqlen=len(text)))
            hidden = projected_text.new_zeros((total_rows + padding, self.hidden_size))
            for indices, values in (
                (packed.video_indices, video),
                (packed.audio_indices, audio),
                (packed.text_indices, torch.cat(refined)),
            ):
                hidden.index_copy_(0, indices, values.to(hidden.dtype))
            hidden = hidden[start:end].contiguous()
            positions = F.pad(packed.position_ids, (0, 0, 0, padding))[start:end]
            freqs = self.rope(positions.unsqueeze(0))
            local_times = F.pad(timestep_indices, (0, padding))[start:end]
            modality_indices = F.pad(timestep_indices * 3 + packed.token_tags, (0, padding))[start:end]
            time_embedding = self.time_embedder(timestep.flatten())
            camera, camera_rows = self.control(packed.controls)
            local_camera = (camera_rows >= start) & (camera_rows < end)
            camera, camera_rows = camera[local_camera], camera_rows[local_camera] - start
            if attention_groups is not None and world_size != 1:
                raise ValueError("For JING SP, omit attention_groups so each rank derives its local query groups")
            groups = (
                attention_groups
                if attention_groups is not None
                else build_attention_groups(
                    packed.owners, packed.is_media, local_start=start, local_length=local_length
                )
            )
            rotary = None
        else:
            if attention_groups is not None:
                raise ValueError("JING request caching derives its attention groups from the bound packed request")
            geometry = (
                total_rows,
                world_size,
                rank,
                local_length,
                start,
                end,
                padding,
                get_tensor_model_parallel_world_size(),
            )
            fixed = request_cache.get(self, packed, geometry)
            if fixed is None:
                fixed = self._prepare_fixed_conditioning(
                    packed, start=start, end=end, padding=padding, local_length=local_length
                )
                request_cache.store(fixed)
            hidden = fixed.refined_text.new_zeros((total_rows + padding, self.hidden_size))
            for indices, values in (
                (packed.video_indices, video),
                (packed.audio_indices, audio),
                (packed.text_indices, fixed.refined_text),
            ):
                hidden.index_copy_(0, indices, values.to(hidden.dtype))
            hidden = hidden[start:end].contiguous()
            freqs = fixed.freqs
            local_times = F.pad(timestep_indices, (0, padding))[start:end]
            modality_indices = F.pad(timestep_indices * 3 + packed.token_tags, (0, padding))[start:end]
            time_embedding = self.time_embedder(timestep.flatten())
            camera, camera_rows, groups = fixed.camera, fixed.camera_rows, fixed.groups
            rotary = (fixed.rope_cos, fixed.rope_sin)
        if self.adaln_cache is not None:
            self.adaln_cache.prepare(time_embedding)
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                time_embedding=time_embedding,
                modality_indices=modality_indices,
                freqs=freqs,
                groups=groups,
                world_size=world_size,
                total_rows=total_rows,
                camera=camera,
                camera_rows=camera_rows,
                injector=self.control.block_injectors[index],
                rotary=rotary,
            )
        video_pred, audio_pred = self.final_layer(hidden, t_emb=time_embedding, inverse_indices=local_times)
        predictions = gather_sequence_rows(torch.cat((video_pred, audio_pred), dim=-1), world_size, total_rows)
        video_width = video_pred.shape[-1]
        return (
            predictions[packed.video_indices, :video_width].unsqueeze(0),
            predictions[packed.audio_indices, video_width:].unsqueeze(0),
        )


EntryClass = XGenJingDiTModel
