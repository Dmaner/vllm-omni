# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import PIL.Image
import torch
from torch import nn
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput, SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import get_minimax_h3_post_process_func
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader import OffloadConfig, OffloadPlan, OffloadStrategy
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.quantization.component_config import ComponentQuantizationConfig, resolve_component_quant_config

from .packing import JingPackedInputs, audio_endpoint, chunk_lengths, expand_slices, pack
from .transformer import XGenJingDiTModel

_H3_COMPONENTS = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler")
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
_GENERATION = {
    "first_chunk_size": 2,
    "ref_image_slots": 5,
    "reserved_slots": 15,
    "reference_timestep": 0.0,
    "text_encoder_reference_images": True,
}


def get_xgen_jing_post_process_func(od_config: OmniDiffusionConfig):
    return get_minimax_h3_post_process_func(od_config)


def _h3_api() -> SimpleNamespace:
    # Keep an optional model dependency out of imports of the shared registry.
    try:
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
        from diffusers.modular_pipelines import PipelineState
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AfterDenoiseStep,
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3FL2VATextEncoderStep,
            MiniMaxH3KeyframeVaeEncoderStep,
        )
        from diffusers.video_processor import VideoProcessor
    except ImportError as exc:
        raise ImportError(
            "XGEN-JING requires the MiniMax-H3 modular encoders, decoders and scheduler from Diffusers 0.40.0."
        ) from exc
    return SimpleNamespace(
        video_vae=AutoencoderKLMiniMaxH3,
        audio_vae=AutoencoderKLMiniMaxH3Audio,
        scheduler=MiniMaxH3Scheduler,
        state=PipelineState,
        after_denoise=MiniMaxH3AfterDenoiseStep,
        video_decoder=MiniMaxH3VideoDecodeStep,
        audio_decoder=MiniMaxH3AudioDecodeStep,
        reference_encoder=MiniMaxH3KeyframeVaeEncoderStep,
        text_encoder=MiniMaxH3FL2VATextEncoderStep,
        video_processor=VideoProcessor,
    )


def _prepare_image(image: PIL.Image.Image, height: int, width: int, *, stretch: bool) -> PIL.Image.Image:
    image = image.convert("RGB")
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    # Keep the released reference geometry: cover the canvas using rounded
    # dimensions, then center-crop. ImageOps.fit uses different rounding.
    scale = max(width / image.width, height / image.height)
    resized_width = max(width, round(image.width * scale))
    resized_height = max(height, round(image.height * scale))
    image = image.resize((resized_width, resized_height), PIL.Image.Resampling.LANCZOS)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _component_root(od_config: OmniDiffusionConfig) -> Path:
    model = str(od_config.model_config.get("xgen_jing_h3_model", "MiniMaxAI/MiniMax-H3"))
    root = Path(model).expanduser()
    if not root.is_dir():
        if root.is_absolute() or model.startswith("."):
            raise ValueError(f"XGEN-JING H3 component directory does not exist: {root}")
        root = Path(
            download_weights_from_hf_specific(
                model_name_or_path=model,
                cache_dir=None,
                allow_patterns=[f"{name}/**" for name in _H3_COMPONENTS],
                revision=od_config.model_config.get("xgen_jing_h3_revision"),
                require_all=True,
            )
        )
    missing = [name for name in _H3_COMPONENTS if not (root / name).is_dir()]
    if missing:
        raise ValueError(
            f"XGEN-JING requires the Diffusers MiniMax-H3 component layout at {root}; missing {missing}. "
            "Set model_config.xgen_jing_h3_model to that root, not an FL2VA/Ref2VA partition."
        )
    return root


class _JingTextConditioner(nn.Module):
    """Give the H3 conditioning helper a forward boundary for Omni offload hooks."""

    def __init__(self, model: nn.Module, tokenizer: Any, processor: Any, encode_step: Any, state_type: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.encode_step = encode_step()
        self.state_type = state_type

    def forward(self, prompt: str, images: list[PIL.Image.Image], device: torch.device, dtype: torch.dtype):
        state = self.state_type()
        state.set("prompt", prompt)
        state.set("keyframes", images)
        components = SimpleNamespace(
            text_encoder=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            _execution_device=device,
            text_encoder_layer=50,
            text_tag=1,
            video_tag=0,
        )
        _, state = self.encode_step(components, state)
        return state.get("prompt_embeds").to(device=device, dtype=dtype), state.get("text_token_tags")


class XGenJingPipeline(
    nn.Module,
    DenoiseProgressMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
):
    """The released bidirectional JING checkpoint, with four joint audio/video steps."""

    supports_request_cancellation: ClassVar[bool] = True
    supports_step_execution: ClassVar[bool] = False
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"token_refiner": "blocks"},
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={
            "text_encoder": ("model.model.visual.blocks", "model.model.language_model.layers"),
        },
        encoder_dlo_weight_replication=frozenset({"text_encoder"}),
    )
    _PROFILER_TARGETS: ClassVar[list[str]] = ["encode_conditioning", "diffuse", "decode"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.od_config = od_config
        # The registry reapplies this flag after construction; tiled VAE
        # execution is part of JING's released recipe, not only a memory option.
        if od_config.vae_use_tiling is not True:
            raise ValueError(
                "XGEN-JING requires vae_use_tiling=True (CLI: --vae-use-tiling). "
                "Tiled VAE encoding and decoding are part of its released generation recipe."
            )
        self.device = get_local_device()
        self.parallel_config = od_config.parallel_config
        self._vae_cpu_offload = od_config.model_config.get("xgen_jing_vae_cpu_offload", False)
        if type(self._vae_cpu_offload) is not bool:
            raise TypeError("model_config.xgen_jing_vae_cpu_offload must be a boolean")
        if self._vae_cpu_offload:
            if self.device.type != "cuda":
                raise ValueError("XGEN-JING VAE CPU offload currently requires a CUDA execution device")
            if self.parallel_config.use_hsdp or od_config.enable_cuda_graph_decode:
                raise ValueError("XGEN-JING VAE CPU offload requires HSDP and CUDA graph decode to be disabled")
            if od_config.diffusion_load_format != "default":
                raise ValueError("XGEN-JING VAE CPU offload requires the native default load format")
            # These registered VAEs are staged explicitly below. Exclude them
            # from discovery's always-resident VAE list so a shared offloader
            # cannot move them back to the accelerator after construction.
            self._vae_modules = []
        if od_config.dtype != torch.bfloat16:
            raise ValueError("XGEN-JING currently requires dtype=bfloat16")
        quant_config = od_config.quantization_config
        if isinstance(quant_config, ComponentQuantizationConfig) and any(
            quant_config.resolve(name) is not None for name in ("text_encoder", "vae", "video_vae", "audio_vae")
        ):
            raise ValueError("XGEN-JING online FP8 requires text_encoder and both VAEs to remain unquantized")
        transformer_quant_config = resolve_component_quant_config(quant_config, "transformer")
        if od_config.cache_backend not in (None, "none"):
            raise ValueError("XGEN-JING does not yet support diffusion cache backends")
        if od_config.lora_path:
            raise ValueError("XGEN-JING does not yet support LoRA adapters")
        if self.parallel_config.cfg_parallel_size != 1:
            raise ValueError("XGEN-JING is guidance-distilled; cfg_parallel_size must be 1")
        if self.parallel_config.pipeline_parallel_size != 1 or self.parallel_config.vae_patch_parallel_size != 1:
            raise ValueError("XGEN-JING does not yet support pipeline parallelism or VAE patch parallelism")
        if self.parallel_config.use_hsdp and od_config.hsdp_weight_load_strategy == "pre_sharded":
            raise ValueError("XGEN-JING's checkpoint mapping requires hsdp_weight_load_strategy=full")
        if any(not od_config.model_loaded.get(name, True) for name in ("text_encoder", "vae_encoder", "vae_decoder")):
            raise ValueError("XGEN-JING currently requires local text encoding and local VAE encoding/decoding")
        self.api = _h3_api()
        self.architecture = od_config.tf_model_config.to_dict()
        self.patch_size = tuple(self.architecture["patch_size"])
        self.transformer = XGenJingDiTModel(od_config, quant_config=transformer_quant_config, diffusers_weights=True)
        model_path = Path(str(od_config.model)).expanduser()
        # Both the Hub repository root and a directly selected jing_flash_v1
        # directory use the same native checkpoint stream.
        subfolder = None if (model_path / "config.json").is_file() else "jing_flash_v1"
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(model_path) if model_path.is_dir() else str(od_config.model),
                subfolder=subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        root = _component_root(od_config)
        # Honor the loader's CPU/offload/HSDP placement context, while avoiding
        # safetensors' direct-to-device allocations for pretrained components.
        load_device = torch.empty(0).device
        offload_config = OffloadConfig.from_od_config(od_config)
        offload_encoder = offload_config.strategy is not OffloadStrategy.NONE and offload_config.should_offload_encoder(
            "text_encoder", self._offload_plan
        )
        with torch.device("cpu"):
            tokenizer = Qwen2TokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True)
            processor = Qwen3VLProcessor.from_pretrained(root / "processor", local_files_only=True)
            encoder = (
                Qwen3VLForConditionalGeneration.from_pretrained(
                    root / "text_encoder",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    local_files_only=True,
                )
                .eval()
                .requires_grad_(False)
            )
            self.text_encoder = _JingTextConditioner(
                encoder, tokenizer, processor, self.api.text_encoder, self.api.state
            )
            self.video_vae = (
                self.api.video_vae.from_pretrained(root / "vae", torch_dtype=torch.float32, local_files_only=True)
                .eval()
                .requires_grad_(False)
            )
            self.audio_vae = (
                self.api.audio_vae.from_pretrained(root / "audio_vae", torch_dtype=torch.float32, local_files_only=True)
                .eval()
                .requires_grad_(False)
            )
        # Online quantization can select a CUDA construction device even when
        # offload is enabled. Leave a selected encoder on CPU for the backend
        # to stage; moving all its weights first defeats layerwise offload.
        if not offload_encoder:
            self.text_encoder.to(load_device)
        if not self._vae_cpu_offload:
            self.video_vae.to(load_device)
            self.audio_vae.to(load_device)
        self.video_vae.enable_tiling()
        self.scheduler = self.api.scheduler.from_pretrained(root / "scheduler", local_files_only=True)
        self.audio_scheduler = self.api.scheduler.from_pretrained(root / "audio_scheduler", local_files_only=True)
        self.video_processor = self.api.video_processor(vae_scale_factor=16, do_normalize=False)
        if self.audio_vae.config.sampling_rate != 32000:
            raise ValueError("XGEN-JING requires the H3 audio VAE with a 32000 Hz sample rate")
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def vae(self):
        return self.video_vae

    @contextmanager
    def _vae_on_device(self, vae: nn.Module) -> Iterator[None]:
        if not self._vae_cpu_offload:
            yield
            return
        # Diffusers helpers call encode/decode directly, bypassing forward
        # hooks. Keep the released FP32 weights and tile geometry unchanged.
        try:
            vae.to(self.device)
            yield
        finally:
            # Also release partially moved weights if transfer or execution
            # fails. A cleanup failure propagates rather than running on CPU.
            vae.to("cpu")

    def _offload_vaes(self) -> None:
        if self._vae_cpu_offload:
            self.video_vae.to("cpu")
            self.audio_vae.to("cpu")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def transformer_weights():
            for name, tensor in weights:
                if not name.startswith("transformer."):
                    raise ValueError(f"Unexpected XGEN-JING weight source: {name}")
                yield name.removeprefix("transformer."), tensor

        loaded = {f"transformer.{name}" for name in self.transformer.load_weights(transformer_weights())}
        # These components were loaded with their own pretrained loaders above.
        loaded.update(name for name, _ in self.named_parameters() if not name.startswith("transformer."))
        return loaded

    def _prepare_request(self, prompt: Any, params: OmniDiffusionSamplingParams) -> dict[str, Any]:
        if not isinstance(prompt, (str, Mapping)):
            raise OmniClientError("XGEN-JING expects one text prompt, optionally with multi_modal_data.image")
        prompt_data = {"prompt": prompt} if isinstance(prompt, str) else prompt
        options = (params.extra_args or {}).get("xgen_jing", {})
        if not isinstance(options, Mapping):
            raise OmniClientError("extra_args.xgen_jing must be a dictionary")
        unknown = set(options) - {"chunks", "slices", "control", "reference_images"}
        if unknown:
            raise OmniClientError(f"Unknown extra_args.xgen_jing fields: {sorted(unknown)}")
        if params.num_inference_steps not in (None, 4) or params.timesteps is not None or params.sigmas is not None:
            raise OmniClientError("jing_flash_v1 uses its fixed four-step schedule; custom schedules are unsupported")
        if (
            params.do_classifier_free_guidance
            or (params.guidance_scale_provided and params.guidance_scale not in (None, 1, 1.0))
            or (params.guidance_scale_2_provided and params.guidance_scale_2 not in (None, 1, 1.0))
        ):
            raise OmniClientError("XGEN-JING is guidance-distilled and does not accept classifier-free guidance")
        if prompt_data.get("negative_prompt"):
            raise OmniClientError("XGEN-JING does not use a negative prompt")
        if params.resolved_frame_rate not in (None, 24, 24.0):
            raise OmniClientError("XGEN-JING generates synchronized video/audio at 24 fps")
        if params.output_type == "latent":
            raise OmniClientError("XGEN-JING currently returns decoded video/audio only")
        if params.lora_request is not None:
            raise OmniClientError("XGEN-JING does not yet support request-scoped LoRA adapters")
        if type(params.num_outputs_per_prompt) is not int or params.num_outputs_per_prompt < 1:
            raise OmniClientError("num_outputs_per_prompt must be a positive integer")
        if isinstance(params.generator, list):
            raise OmniClientError("XGEN-JING accepts one generator or seed per request")
        settings = dict(_GENERATION)
        settings.update(
            height=544 if params.height is None else params.height,
            width=960 if params.width is None else params.width,
        )
        # The generic sampling dataclass defaults to one image frame.
        frames = 124 if params.num_frames in (None, 1) else params.num_frames
        for name, value in {**settings, "num_frames": frames}.items():
            if name in {"height", "width", "num_frames"} and (type(value) is not int or value <= 0):
                raise OmniClientError(f"{name} must be a positive integer")
        if settings["height"] % 32 or settings["width"] % 32:
            raise OmniClientError("XGEN-JING height and width must be divisible by 32")
        try:
            settings["num_frames"], lengths = chunk_lengths(frames, settings["first_chunk_size"])
            slices = expand_slices({"prompt": prompt_data.get("prompt", ""), **options}, len(lengths))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(str(exc)) from exc
        media = prompt_data.get("multi_modal_data") or {}
        if not isinstance(media, Mapping) or set(media) - {"image"}:
            raise OmniClientError("XGEN-JING accepts reference images; audio/video input is unsupported")
        images = media.get("image")
        if images is not None and "reference_images" in options:
            raise OmniClientError(
                "Supply reference images through either multi_modal_data.image or xgen_jing, not both"
            )
        if images is None:
            images = options.get("reference_images", [])
        if isinstance(images, PIL.Image.Image):
            images = [images]
        if not isinstance(images, (list, tuple)) or any(not isinstance(image, PIL.Image.Image) for image in images):
            raise OmniClientError(
                "Reference images must be decoded PIL images; use Omni's image input route for URLs or local files"
            )
        if len(images) > settings["ref_image_slots"]:
            raise OmniClientError("XGEN-JING accepts at most five reference images")
        seed = params.seed
        if seed is None:
            seed = params.generator.initial_seed() if params.generator is not None else 42
        if type(seed) is not int or seed < 0 or seed >= 2**63 - params.num_outputs_per_prompt:
            raise OmniClientError("seed must be a nonnegative integer below 2**63 - num_outputs_per_prompt")
        return {"settings": settings, "lengths": lengths, "slices": slices, "images": images, "seed": seed}

    @torch.no_grad()
    def encode_conditioning(self, case: dict[str, Any]) -> dict[str, Any]:
        settings = case["settings"]
        images = [
            _prepare_image(image, settings["height"], settings["width"], stretch=index == 0)
            for index, image in enumerate(case["images"])
        ]
        text_bank = {}
        for prompt in dict.fromkeys(item["prompt"] for item in case["slices"]):
            check_request_cancellation(synchronize=True)
            text_bank[prompt] = self.text_encoder(prompt, images, self.device, torch.bfloat16)
        references = []
        components = SimpleNamespace(
            vae=self.video_vae,
            _execution_device=self.device,
            pixel_mean=_PIXEL_MEAN,
            pixel_std=_PIXEL_STD,
            keyframe_encode_seed=42,
        )
        reference_encoder = self.api.reference_encoder()
        if images:
            with self._vae_on_device(self.video_vae):
                for image in images:
                    check_request_cancellation(synchronize=True)
                    state = self.api.state()
                    state.set("keyframes", [image])
                    _, state = reference_encoder(components, state)
                    references.append(state.get("condition_latents")[0][0])
        return {**case, "text_bank": text_bank, "references": references}

    def _initial_inputs(self, conditioning: dict[str, Any], seed: int) -> JingPackedInputs:
        frames = sum(conditioning["lengths"])
        settings = conditioning["settings"]
        # Draw on CPU in temporal order so seeds retain the released recipe's
        # sequence independently of device count and parallel partitioning.
        video = (
            torch.randn(
                frames,
                self.architecture["in_channels"],
                settings["height"] // 16,
                settings["width"] // 16,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device="cpu",
                dtype=torch.float32,
            )
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        audio = (
            torch.randn(
                audio_endpoint(frames),
                2,
                self.architecture["audio_in_channels"],
                generator=torch.Generator(device="cpu").manual_seed(seed + 1),
                device="cpu",
                dtype=torch.float32,
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        return pack({**conditioning, "video": video, "audio": audio}, self.architecture, _GENERATION, self.device)

    @torch.no_grad()
    def diffuse(self, packed: JingPackedInputs) -> tuple[torch.Tensor, torch.Tensor]:
        self._offload_vaes()
        # Explicit sigmas include the clean endpoint. H3's timesteps are 1-sigma
        # and its velocity points toward data, so a generic flow scheduler is
        # not interchangeable with this scheduler.
        base = torch.tensor((1.0, 0.7, 0.4, 0.15, 0.0), dtype=torch.float32, device="cpu")
        for scheduler in (self.scheduler, self.audio_scheduler):
            shifted = scheduler.shift * base / (1 + (scheduler.shift - 1) * base)
            scheduler.set_timesteps(sigmas=shifted, device=self.device)
            if len(scheduler.timesteps) != 4:
                raise RuntimeError("The H3 scheduler must produce four timesteps for jing_flash_v1")
        try:
            with self.progress_bar(total=4) as progress:
                for index, (video_time, audio_time) in enumerate(
                    zip(self.scheduler.timesteps, self.audio_scheduler.timesteps, strict=True)
                ):
                    check_request_cancellation(synchronize=True)
                    self.record_denoise_step(index, normalized_timestep=video_time.item(), total_steps=4)
                    timesteps, timestep_indices = packed.timesteps(video_time, audio_time, 0.0)
                    video_prediction, audio_prediction = self.transformer(packed, timesteps, timestep_indices)
                    generated = slice(packed.output_video_offset, None)
                    packed.video[:, generated] = self.scheduler.step(
                        video_prediction[:, generated].float(),
                        video_time,
                        packed.video[:, generated],
                        return_dict=False,
                    )[0]
                    packed.audio[:] = self.audio_scheduler.step(
                        audio_prediction.float(), audio_time, packed.audio, return_dict=False
                    )[0]
                    progress.update()
        finally:
            self.record_denoise_step(None, total_steps=None)
        if not torch.isfinite(packed.video).all() or not torch.isfinite(packed.audio).all():
            raise FloatingPointError("XGEN-JING produced non-finite latents")
        return packed.video[0, packed.output_video_offset :], packed.audio_channel_major()

    @torch.no_grad()
    def decode(self, video: torch.Tensor, audio: torch.Tensor, case: dict[str, Any]):
        settings = case["settings"]
        frames = sum(case["lengths"])
        state = self.api.state()
        decode_inputs = {
            "latents": video,
            "audio_latents": audio,
            "num_condition_video_rows": 0,
            "num_condition_audio_rows": 0,
            "num_latent_frames": frames,
            "num_audio_latents": audio_endpoint(frames),
            "latent_height": settings["height"] // 16,
            "latent_width": settings["width"] // 16,
            "output_type": "pt",
        }
        for name, value in decode_inputs.items():
            state.set(name, value)
        components = SimpleNamespace(
            _execution_device=self.device,
            vae=self.video_vae,
            audio_vae=self.audio_vae,
            vae_latent_channels=self.architecture["in_channels"],
            patch_size=self.patch_size,
            audio_channels=2,
            audio_sampling_rate=self.audio_vae.config.sampling_rate,
            video_processor=self.video_processor,
            pixel_mean=_PIXEL_MEAN,
            pixel_std=_PIXEL_STD,
        )
        check_request_cancellation(synchronize=True)
        _, state = self.api.after_denoise()(components, state)
        with self._vae_on_device(self.video_vae):
            _, state = self.api.video_decoder()(components, state)
        check_request_cancellation(synchronize=True)
        with self._vae_on_device(self.audio_vae):
            _, state = self.api.audio_decoder()(components, state)
        video, audio = state.get("videos"), state.get("audio")
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise FloatingPointError("XGEN-JING decoded non-finite media")
        # Diffusers returns B,T,C,H,W; Omni transports B,T,H,W,C uint8.
        video = video.detach().float()
        video.clamp_(0, 1).mul_(255).round_()
        frames = video.permute(0, 1, 3, 4, 2)
        quantized = torch.empty(frames.shape, device=frames.device, dtype=torch.uint8)
        quantized.copy_(frames)
        return quantized, audio

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("XGEN-JING supports one request at a time")
        check_request_cancellation(synchronize=True)
        params = request.sampling_params
        case = self._prepare_request(request.prompts[0], params)
        conditioning = self.encode_conditioning(case)
        videos, audios = [], []
        for output_index in range(params.num_outputs_per_prompt):
            check_request_cancellation(synchronize=True)
            packed = self._initial_inputs(conditioning, case["seed"] + output_index)
            video_latents, audio_latents = self.diffuse(packed)
            video, audio = self.decode(video_latents, audio_latents, case)
            videos.append(video.cpu())
            audios.append(audio.cpu())
            del packed, video_latents, audio_latents, video, audio
        return DiffusionOutput(
            output=(torch.cat(videos), torch.cat(audios)),
            video_output_index=0,
            post_process_func=get_xgen_jing_post_process_func(self.od_config),
            stage_durations=self.stage_durations if self.enable_diffusion_pipeline_profiler else {},
        )
