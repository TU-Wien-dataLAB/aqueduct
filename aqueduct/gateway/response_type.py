import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from functools import reduce
from typing import Any, Literal, TypeVar

import litellm
from asgiref.sync import sync_to_async
from django.core.serializers.json import DjangoJSONEncoder
from django.db import connection
from django.http.response import HttpResponseBase, StreamingHttpResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)
from litellm.types.utils import Usage as UsageModel
from mcp.types import JSONRPCMessage
from pydantic import BaseModel

from gateway.rate_limiting import record_token_usage
from management.models import Request, Usage

log = logging.getLogger("aqueduct")

T = TypeVar("T", bound=ModelResponseStream | JSONRPCMessage)

# Sent instead of `data: [DONE]` when an OpenAI stream fails. The exception text
# is only logged, never sent to the client.
STREAM_ERROR_EVENT = (
    "data: "
    + json.dumps(
        {"error": {"message": "Upstream stream failed", "type": "server_error", "code": 500}}
    )
    + "\n\n"
).encode()


def release_db_connection() -> None:
    """Return this thread's database connection to the pool.

    Django only returns it on `request_finished`, which never runs when an
    exception escapes a streaming response, so the pool slot leaks (#246).
    Must run in the thread that uses the connection (i.e. via `sync_to_async`).
    A connection inside an atomic block is kept: closing it would abort the
    transaction (e.g. the one wrapping each Django `TestCase`).
    """
    if not connection.in_atomic_block:
        connection.close()


def _save_and_release(request_log: Request) -> None:
    try:
        request_log.save()
    finally:
        release_db_connection()


class RawJsonResponse(HttpResponseBase):
    """Mimics JSONResponse behaviour, but dumps data to JSON lazily - only on access."""

    streaming = False

    def __init__(self, data: dict[str, Any] | list[Any] | BaseModel, **kwargs: Any) -> None:
        if not isinstance(data, (dict, list, BaseModel)):
            raise TypeError("RawJsonResponse data has to be a dict, list, or a pydantic BaseModel")

        # ``data`` is the original object passed when creating the response instance
        self.data = data
        # ``_content`` stores the data dumped to a string
        self._content: bytes | None = None
        kwargs.setdefault("content_type", "application/json")
        super().__init__(**kwargs)

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = self._dump_data()
        return self._content

    def _dump_data(self) -> bytes:
        """Serialize ``self.data`` to JSON and return the result as bytes."""
        content: Any
        if isinstance(self.data, dict):
            # Data can be a dict with models / lists of models as values
            content = {k: _dump_value(v) for k, v in self.data.items()}
        else:
            # Data can be a pydantic model, or a list of models / plain values
            # (e.g. /model_group/info), matching JsonResponse(data, safe=False)
            content = _dump_value(self.data)
        return self.make_bytes(json.dumps(content, cls=DjangoJSONEncoder))

    @property
    def text(self) -> str:
        return self.content.decode(self.charset or "utf-8")


def _dump_value(value: Any) -> Any:
    """Return a JSON-serializable version of ``value``.

    Pydantic models are dumped to dicts (dropping None and unset fields),
    lists and tuples are traversed recursively; anything else is returned
    unchanged for ``DjangoJSONEncoder`` to handle.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True, exclude_unset=True, mode="json")
    if isinstance(value, (list, tuple)):
        return [_dump_value(item) for item in value]
    return value


def _apply_transforms(chunk: T, transforms: list[Callable[[T], T]]) -> T:
    return reduce(lambda obj, tr: tr(obj), transforms, chunk)


def _count_text_tokens(model: str, text: str) -> int:
    """Estimate BPE token count for ``text`` using LiteLLM's local tokenizer."""
    if not text:
        return 0
    try:
        return max(0, int(litellm.token_counter(model=model or "gpt-4o", text=text)))
    except Exception:
        log.debug("Failed to count text tokens for model %r", model, exc_info=True)
        return 0


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for sub in content:
        if isinstance(sub, str):
            out.append(sub)
        elif isinstance(sub, Mapping):
            text = sub.get("text")
            if isinstance(text, str) and text:
                out.append(text)
    return "\n".join(out)


def _responses_input_to_text(resp_input: object) -> str:
    """Extract plain text from a Responses API ``input`` payload for token estimation."""
    if isinstance(resp_input, str):
        return resp_input
    if not isinstance(resp_input, list):
        return ""
    parts: list[str] = []
    for item in resp_input:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping):
            text = _content_to_text(item.get("content"))
            if text:
                parts.append(text)
            for field in ("text", "arguments", "output"):
                val = item.get(field)
                if isinstance(val, str) and val:
                    parts.append(val)
    return "\n".join(parts)


def _estimate_prompt_tokens(model: str, prompt: object) -> int:
    if isinstance(prompt, str):
        return _count_text_tokens(model, prompt)
    if not isinstance(prompt, list) or not prompt:
        return 0
    if isinstance(prompt[0], int):
        return len(prompt)
    if isinstance(prompt[0], list):
        return sum(len(item) for item in prompt if isinstance(item, list))
    return sum(_count_text_tokens(model, item) for item in prompt if isinstance(item, str))


def _estimate_input_tokens(model: str, request_body: Mapping[str, object] | None) -> int:
    """Estimate request input tokens when a stream ends before upstream reports usage."""
    if not request_body:
        return 0
    raw_model = request_body.get("model")
    effective_model = model or (raw_model if isinstance(raw_model, str) and raw_model else "gpt-4o")
    try:
        messages = request_body.get("messages")
        if isinstance(messages, list) and messages:
            tools = request_body.get("tools")
            kwargs: dict[str, Any] = {"model": effective_model, "messages": messages}
            if isinstance(tools, list) and tools:
                kwargs["tools"] = tools
            return max(0, int(litellm.token_counter(**kwargs)))

        if "prompt" in request_body:
            return _estimate_prompt_tokens(effective_model, request_body.get("prompt"))

        instructions = request_body.get("instructions")
        instr_text = instructions if isinstance(instructions, str) else ""
        input_text = (
            _responses_input_to_text(request_body.get("input")) if "input" in request_body else ""
        )
        parts = [s for s in (instr_text, input_text) if s]
        if parts:
            return _count_text_tokens(effective_model, "\n".join(parts))
    except Exception:
        log.debug("Failed to estimate input tokens for model %r", effective_model, exc_info=True)
    return 0


def _get_field(obj: object, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_call_fields(call_obj: object) -> str:
    if call_obj is None:
        return ""
    return "".join(
        val
        for attr in ("name", "arguments")
        if isinstance(val := _get_field(call_obj, attr), str) and val
    )


def _extract_delta_text(delta: object) -> tuple[str, str]:
    visible: list[str] = []
    content = _get_field(delta, "content")
    if isinstance(content, str) and content:
        visible.append(content)
    # normalize_reasoning_fields populates both keys with the same string;
    # read only one to avoid double-counting reasoning tokens.
    reas = _get_field(delta, "reasoning_content")
    if reas is None:
        reas = _get_field(delta, "reasoning")
    reasoning = reas if isinstance(reas, str) else ""

    tool_calls = _get_field(delta, "tool_calls")
    if isinstance(tool_calls, list):
        visible.extend(_extract_call_fields(_get_field(tc, "function")) for tc in tool_calls)
    visible.append(_extract_call_fields(_get_field(delta, "function_call")))
    return "".join(visible), reasoning


def _extract_stream_chunk_text(chunk: object) -> tuple[str, str]:
    """Extract ``(visible_text, reasoning_text)`` from a single OpenAI stream chunk."""
    choices = _get_field(chunk, "choices")
    if isinstance(choices, list):
        visible: list[str] = []
        reasoning: list[str] = []
        for choice in choices:
            delta = _get_field(choice, "delta")
            if delta is not None:
                vis, reas = _extract_delta_text(delta)
                if vis:
                    visible.append(vis)
                if reas:
                    reasoning.append(reas)
            elif isinstance(text := _get_field(choice, "text"), str) and text:
                visible.append(text)
        return "".join(visible), "".join(reasoning)

    event_type = _get_field(chunk, "type")
    if isinstance(event_type, str) and event_type.endswith(".delta"):
        delta_text = _get_field(chunk, "delta")
        if isinstance(delta_text, str) and delta_text:
            return ("", delta_text) if "reasoning" in event_type else (delta_text, "")

    return "", ""


def _build_incomplete_usage(
    token_usage: Usage,
    visible_parts: list[str],
    reasoning_parts: list[str],
    model_name: str,
    request_body: Mapping[str, object] | None,
) -> Usage:
    in_tokens = (
        token_usage.input_tokens
        if token_usage.input_tokens > 0
        else _estimate_input_tokens(model_name, request_body)
    )
    if token_usage.output_tokens > 0:
        out_tokens = token_usage.output_tokens
        reas_tokens = token_usage.reasoning_tokens
    else:
        vis_tokens = _count_text_tokens(model_name, "".join(visible_parts))
        reas_tokens = _count_text_tokens(model_name, "".join(reasoning_parts))
        out_tokens = vis_tokens + reas_tokens
    return Usage(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cached_input_tokens=token_usage.cached_input_tokens,
        reasoning_tokens=reas_tokens,
    )


def _finalize_stream_log(
    request_log: Request,
    status: int,
    start_time: float,
    usage: Usage | None,
    *,
    record_limits: bool,
) -> None:
    if usage is not None:
        request_log.token_usage = usage
    request_log.status_code = status
    request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
    # Save and release in one call, so no connection is held if an
    # exception or cancellation escapes afterwards.
    _save_and_release(request_log)
    if record_limits:
        record_token_usage(request_log.token_id, request_log.token_usage)


class RawStreamingResponse(StreamingHttpResponse):
    """A wrapper for streaming data that can be turned into a StreamingHttpResponse."""

    def __init__(
        self,
        streaming_content: AsyncIterator[Any],
        request_log: Request | None,
        transforms: list[Callable[[T], T]] | None = None,
        *,
        mode: Literal["openai", "mcp"] = "openai",
        request_body: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(streaming_content, AsyncIterator):
            raise TypeError("RawStreamResponse streaming_content has to be async iterable")

        self.request_log = request_log
        self.transforms = transforms or []
        self.mode = mode
        self.request_body = request_body
        kwargs.setdefault("content_type", "text/event-stream")
        super().__init__(streaming_content=(), **kwargs)
        self.streaming_content = self._iter_stream(streaming_content)

    async def _record_incomplete(
        self,
        status: int,
        start_time: float,
        token_usage: Usage,
        visible_parts: list[str],
        reasoning_parts: list[str],
    ) -> None:
        if self.request_log is None:
            return
        is_openai = self.mode == "openai"
        usage = (
            _build_incomplete_usage(
                token_usage,
                visible_parts,
                reasoning_parts,
                self.request_log.model or "",
                self.request_body,
            )
            if is_openai
            else None
        )
        await sync_to_async(_finalize_stream_log)(
            self.request_log, status, start_time, usage, record_limits=is_openai
        )

    async def _iter_stream(self, streaming_content: AsyncIterator[Any]) -> AsyncGenerator[bytes]:
        """Post-process streaming response chunks with transforms, and log OpenAI responses.

        Note: MCP streaming responses do not have `request_log` attached;
        usage tokens and response time are only logged for OpenAI responses.
        Also, OpenAI responses require the last yielded chunk to be b"data: [DONE]\n\n",
        which however is not MCP-compliant.
        """
        token_usage = Usage(0, 0)
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        start_time = time.monotonic()
        is_openai = self.mode == "openai"
        log.debug(
            "%r stream. Applying the following transforms to each chunk: %s",
            self.mode,
            self.transforms,
        )

        if is_openai and self.request_log is None:
            raise ValueError(f"Missing request_log for an OpenAI streaming response: {self}!")

        try:
            async for raw_chunk in streaming_content:
                chunk = _apply_transforms(raw_chunk, self.transforms)

                if is_openai:
                    chunk_usage = get_token_usage(chunk)
                    if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                        token_usage = chunk_usage
                    vis_delta, reas_delta = _extract_stream_chunk_text(chunk)
                    if vis_delta:
                        visible_parts.append(vis_delta)
                    if reas_delta:
                        reasoning_parts.append(reas_delta)
                    chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
                else:
                    chunk_str = chunk.model_dump_json(exclude_none=True)

                yield f"data: {chunk_str}\n\n".encode()

            if is_openai and self.request_log is not None:
                await sync_to_async(_finalize_stream_log)(
                    self.request_log, 200, start_time, token_usage, record_limits=True
                )
                yield b"data: [DONE]\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            await self._record_incomplete(
                499, start_time, token_usage, visible_parts, reasoning_parts
            )
            raise
        except Exception:
            log.exception("Error while streaming response: %r", streaming_content)
            await self._record_incomplete(
                500, start_time, token_usage, visible_parts, reasoning_parts
            )
            if not is_openai:
                raise
            # An exception escaping the stream would leave the client hanging,
            # so end the stream with an error event instead (without [DONE]).
            yield STREAM_ERROR_EVENT


def get_token_usage(data: dict[str, Any] | BaseModel) -> Usage:
    """Retrieves token usage information from the raw response data.

    Note that if the response data does not match the expected format, or does
    not contain the usage information, the returned token usage will be wrong,
    i.e. set to 0.

    Args:
        data: The raw response data (or content's chunk for streaming responses),
          expected to be a dict or BaseModel subclass.
    Returns:
        The :class:`Usage` object with the used input and output token counts.
    """
    if isinstance(
        data, (dict, ModelResponse, ModelResponseStream, EmbeddingResponse, TextCompletionResponse)
    ):
        # LiteLLM models implement `.get()` method, but the OpenAI ones - don't.
        usage = data.get("usage")
        if isinstance(usage, (dict, UsageModel)):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            cached_input_tokens = 0
            if isinstance(details, dict):
                cached_input_tokens = details.get("cached_tokens") or 0
            elif details is not None:
                cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
            output_details = usage.get("completion_tokens_details") or usage.get(
                "output_tokens_details"
            )
            reasoning_tokens = 0
            if isinstance(output_details, dict):
                reasoning_tokens = output_details.get("reasoning_tokens") or 0
            elif output_details is not None:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
            return Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        # Cohere-style responses (e.g. /rerank with a Cohere-format backend)
        # carry the token counts in meta.tokens instead of a top-level usage
        # object.
        meta = data.get("meta") if isinstance(data, dict) else None
        if isinstance(meta, dict):
            tokens = meta.get("tokens")
            if isinstance(tokens, dict):
                return Usage(
                    input_tokens=tokens.get("input_tokens") or 0,
                    output_tokens=tokens.get("output_tokens") or 0,
                    cached_input_tokens=0,
                    reasoning_tokens=0,
                )

    else:
        # Handle responses API format (top-level usage or in response field)
        usage = getattr(data, "usage", None)
        if not usage and hasattr(data, "response"):
            usage = getattr(data.response, "usage", None)
        if usage:
            try:
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
            except AttributeError:
                input_tokens = output_tokens = 0
            details = getattr(usage, "input_tokens_details", None)
            cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
            output_details = getattr(usage, "output_tokens_details", None)
            reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
            return Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
            )

    return Usage(input_tokens=0, output_tokens=0)
