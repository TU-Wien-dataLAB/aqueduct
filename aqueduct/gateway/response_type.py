import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from functools import reduce
from typing import Any, Literal, TypeVar

from asgiref.sync import sync_to_async
from django.core.serializers.json import DjangoJSONEncoder
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


class RawStreamingResponse(StreamingHttpResponse):
    """A wrapper for streaming data that can be turned into a StreamingHttpResponse."""

    def __init__(
        self,
        streaming_content: AsyncIterator[Any],
        request_log: Request | None,
        transforms: list[Callable[[T], T]] | None = None,
        *,
        mode: Literal["openai", "mcp"] = "openai",
        **kwargs: Any,
    ) -> None:
        if not isinstance(streaming_content, AsyncIterator):
            raise TypeError("RawStreamResponse streaming_content has to be async iterable")

        self.streaming_content = streaming_content
        self.request_log = request_log
        self.transforms = transforms or []
        self.mode = mode
        kwargs.setdefault("content_type", "text/event-stream")
        super().__init__(streaming_content=(), **kwargs)
        self.streaming_content = self._iter_stream(streaming_content)

    async def _iter_stream(self, streaming_content: AsyncIterator[Any]) -> AsyncGenerator[bytes]:
        """Post-process streaming response chunks with transforms, and log OpenAI responses.

        Note: MCP streaming responses do not have `request_log` attached;
        usage tokens and response time are only logged for OpenAI responses.
        Also, OpenAI responses require the last yielded chunk to be b"data: [DONE]\n\n",
        which however is not MCP-compliant.
        """
        token_usage = Usage(0, 0)
        start_time = time.monotonic()
        is_openai = self.mode == "openai"
        log.debug(
            "%r stream. Applying the following transforms to each chunk: %s",
            self.mode,
            self.transforms,
        )

        if is_openai and self.request_log is None:
            raise ValueError(f"Missing request_log for an OpenAI streaming response: {self}!")

        async for raw_chunk in streaming_content:
            chunk = _apply_transforms(raw_chunk, self.transforms)

            if is_openai:
                chunk_usage = get_token_usage(chunk)
                if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                    token_usage = chunk_usage
                chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            else:
                chunk_str = chunk.model_dump_json(exclude_none=True)

            yield f"data: {chunk_str}\n\n".encode()

        if is_openai:
            if self.request_log is None:
                raise ValueError(f"Missing request_log for an OpenAI streaming response: {self}!")
            self.request_log.token_usage = token_usage
            self.request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
            await self.request_log.asave()
            # Record token usage into the rate-limit buckets. Streaming requests
            # defer recording to here (stream end) since token usage is only known
            # once the upstream stream completes.
            await sync_to_async(record_token_usage)(
                self.request_log.token_id, self.request_log.token_usage
            )

            yield b"data: [DONE]\n\n"


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
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
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
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    return Usage(input_tokens=0, output_tokens=0)
