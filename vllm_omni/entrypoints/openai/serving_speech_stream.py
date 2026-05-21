"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json
from contextlib import aclosing
from dataclasses import asdict
from numbers import Real
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    SentenceSplitter,
)
from vllm_omni.utils.forced_aligner import align as forced_align

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_DEFAULT_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_SAMPLE_RATE_CONFIG_ATTRS = (
    "audio_tokenizer_sample_rate",
    "output_sample_rate",
    "output_sampling_rate",
    "audio_sample_rate",
    "sample_rate",
    "sampling_rate",
)
_SAMPLE_RATE_NESTED_CONFIG_ATTRS = (
    "talker_config",
    "speech_tokenizer_config",
    "tokenizer_config",
    "audio_config",
    "vocoder",
    "audio_vae",
    "codec_config",
)


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    @classmethod
    def _resolve_pcm_sample_rate(cls, speech_service: OmniOpenAIServingSpeech) -> int:
        """Best-effort output PCM sample-rate lookup for stream metadata and alignment."""
        model_config = getattr(getattr(speech_service, "engine_client", None), "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        return cls._extract_sample_rate_from_config(hf_config) or _DEFAULT_PCM_SAMPLE_RATE

    @classmethod
    def _extract_sample_rate_from_config(cls, config: Any, seen: set[int] | None = None) -> int | None:
        if config is None or cls._is_mock_value(config):
            return None

        if seen is None:
            seen = set()
        config_id = id(config)
        if config_id in seen:
            return None
        seen.add(config_id)

        for attr_name in _SAMPLE_RATE_CONFIG_ATTRS:
            raw_value = config.get(attr_name) if isinstance(config, dict) else getattr(config, attr_name, None)
            sample_rate = cls._coerce_sample_rate(raw_value)
            if sample_rate is not None:
                return sample_rate

        for attr_name in _SAMPLE_RATE_NESTED_CONFIG_ATTRS:
            nested = config.get(attr_name) if isinstance(config, dict) else getattr(config, attr_name, None)
            sample_rate = cls._extract_sample_rate_from_config(nested, seen)
            if sample_rate is not None:
                return sample_rate

        return None

    @staticmethod
    def _coerce_sample_rate(value: Any) -> int | None:
        if value is None or isinstance(value, bool) or OmniStreamingSpeechHandler._is_mock_value(value):
            return None

        try:
            if hasattr(value, "item") and not isinstance(value, Real):
                value = value.item()
            if not isinstance(value, Real) or isinstance(value, bool):
                return None
            sample_rate = int(value)
        except (TypeError, ValueError):
            return None

        return sample_rate if sample_rate > 0 else None

    @staticmethod
    def _is_mock_value(value: Any) -> bool:
        return type(value).__module__.startswith("unittest.mock")

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(boundary_re=boundary_re)
            sentence_index = 0

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                    await self._send_error(websocket, "input.text message too large")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await self._generate_and_send(websocket, config, sentence, sentence_index)
                        sentence_index += 1

                elif msg_type == "input.done":
                    # Flush remaining buffer
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(websocket, config, remaining, sentence_index)
                        sentence_index += 1

                    # Send session.done
                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "total_sentences": sentence_index,
                        }
                    )
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
            await self._send_error(websocket, "session.config message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "session.config must be a JSON object")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"
        pcm_sample_rate = self._resolve_pcm_sample_rate(self._speech_service)

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            start_payload["sample_rate"] = pcm_sample_rate
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        pcm_buffer = bytearray()

        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                    async for chunk in stream:
                        total_bytes += len(chunk)
                        if config.word_timestamps:
                            pcm_buffer.extend(chunk)
                        await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)

            if config.word_timestamps and config.stream_audio and not generation_failed:
                audio_chunk = bytes(pcm_buffer)

                await self._align_and_send_word_timestamps(
                    websocket=websocket,
                    sentence_text=sentence_text,
                    sentence_index=sentence_index,
                    audio_chunk=audio_chunk,
                    sample_rate=pcm_sample_rate,
                )
        except WebSocketDisconnect:
            generation_failed = True
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(websocket, f"Generation failed for sentence {sentence_index}: {e}")
        finally:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore

    async def _align_and_send_word_timestamps(
        self,
        websocket: WebSocket,
        sentence_text: str,
        sentence_index: int,
        audio_chunk: bytes,
        sample_rate: int,
    ) -> None:
        try:
            word_timestamps = await forced_align(
                audio_chunk=audio_chunk,
                text=sentence_text,
                sr=sample_rate,
            )

            await websocket.send_json(
                {
                    "type": "word_timestamps",
                    "sentence_index": sentence_index,
                    "word_timestamps": [asdict(item) for item in word_timestamps],
                }
            )

        except Exception as e:
            logger.warning(
                "Forced alignment failed for sentence %d: %s",
                sentence_index,
                e,
                exc_info=True,
            )
            await websocket.send_json(
                {
                    "type": "word_timestamps",
                    "sentence_index": sentence_index,
                    "word_timestamps": [],
                    "error": str(e),
                }
            )
