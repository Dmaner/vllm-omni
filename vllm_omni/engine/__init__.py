"""
Engine components for vLLM-Omni.
"""

from typing import Any

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Three supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
      - scalar: a Python scalar (msgspec-serializable)
    Exactly one of (tensor_data, list_data, scalar_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None

    # Scalar form
    scalar_data: Any | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for additional
    information payloads, enabling direct transfer of pre-computed data
    between pipeline stages.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.

    Attributes:
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None

    @classmethod
    def from_request(
        cls,
        request: EngineCoreRequest,
        *,
        prompt_embeds: torch.Tensor | None = None,
        additional_information: AdditionalInformationPayload | None = None,
    ) -> "OmniEngineCoreRequest":
        """Clone an EngineCoreRequest into an OmniEngineCoreRequest with optional payload overrides."""

        if prompt_embeds is None:
            prompt_embeds = request.prompt_embeds
        if additional_information is None:
            additional_information = getattr(request, "additional_information", None)

        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            data_parallel_rank=request.data_parallel_rank,
            prompt_embeds=prompt_embeds,
            client_index=request.client_index,
            current_wave=request.current_wave,
            priority=request.priority,
            trace_headers=request.trace_headers,
            resumable=request.resumable,
            external_req_id=request.external_req_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
            additional_information=additional_information,
        )


class OmniPoolingOutput:
    """Msgpack marker for pooling outputs that may be tensor or omni payload."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)


def _is_serialized_tensor(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and isinstance(value[0], str) and isinstance(value[1], list)


def _decode_pooling_value(decoder: Any, value: Any) -> Any:
    if _is_serialized_tensor(value):
        return decoder._decode_tensor(value)
    if isinstance(value, dict):
        return {key: _decode_pooling_value(decoder, val) for key, val in value.items()}
    if isinstance(value, list):
        return [_decode_pooling_value(decoder, val) for val in value]
    return value


def decode_omni_pooling_output(decoder: Any, value: Any) -> OmniPoolingOutput:
    return OmniPoolingOutput(_decode_pooling_value(decoder, value))


def unwrap_omni_pooling_output(value: Any) -> Any:
    return value.value if isinstance(value, OmniPoolingOutput) else value


class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: OmniPoolingOutput | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
