# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

XGEN_JING_EXTRA_BODY_PARAMS = frozenset({"xgen_jing"})


def get_xgen_jing_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    del extra_body
    return VideoGenerationDefaults(
        width=960,
        height=544,
        num_frames=124,
        num_inference_steps=4,
        guidance_scale=1.0,
        fps=24.0,
        dimension_multiple=32,
        default_negative_prompt=None,
        output="xgen_jing_output.mp4",
    )
