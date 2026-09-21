# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression tests for the pipeline's online FP8 configuration routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipeline_module
from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline_dependencies(monkeypatch):
    """Keep the real pipeline constructor without loading a checkpoint or GPU."""
    model_config = SimpleNamespace(
        llm_config=SimpleNamespace(hidden_size=8),
        vision_config=SimpleNamespace(patch_size=2),
        downsample_ratio=0.5,
        use_pixel_head=True,
        add_noise_scale_embedding=False,
    )
    language_model = nn.Module()
    language_model.model = nn.Module()
    language_model_class = Mock(return_value=language_model)

    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "_resolve_model_path", lambda path: path)
    monkeypatch.setattr(pipeline_module.SenseNovaU1Config, "from_pretrained", Mock(return_value=model_config))
    monkeypatch.setattr(pipeline_module.AutoTokenizer, "from_pretrained", Mock(return_value=Mock()))
    monkeypatch.setattr(pipeline_module, "SenseNovaU1ForCausalLM", language_model_class)
    monkeypatch.setattr(pipeline_module, "NEOVisionModel", Mock(side_effect=lambda config: nn.Identity()))
    monkeypatch.setattr(pipeline_module, "ConvDecoder", Mock(side_effect=lambda hidden_size: nn.Identity()))

    return model_config, language_model_class


@pytest.mark.parametrize(
    "routing",
    ["disabled", "global", "language_model", "unmatched", "default", "component_disabled"],
)
def test_pipeline_routes_online_fp8_config(pipeline_dependencies, routing):
    model_config, language_model_class = pipeline_dependencies
    fp8_config = build_quant_config("fp8")
    quant_config, expected_config = {
        "disabled": (None, None),
        "global": (fp8_config, fp8_config),
        "language_model": (
            ComponentQuantizationConfig({"language_model": fp8_config}),
            fp8_config,
        ),
        "unmatched": (ComponentQuantizationConfig({"transformer": fp8_config}), None),
        "default": (ComponentQuantizationConfig({}, default_config=fp8_config), fp8_config),
        "component_disabled": (
            ComponentQuantizationConfig({"language_model": None}, default_config=fp8_config),
            None,
        ),
    }[routing]
    od_config = SimpleNamespace(
        model="sensenova-test-model",
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        revision=None,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline_module.SenseNovaU1Pipeline(od_config=od_config)

    language_model_class.assert_called_once_with(
        model_config.llm_config,
        quant_config=expected_config,
        prefix="language_model",
    )
