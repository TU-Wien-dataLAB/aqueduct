import base64
import logging
from functools import wraps
from typing import Any, cast

import httpx
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from litellm.types.utils import ModelResponse, ModelResponseStream
from openai.types.chat.chat_completion_content_part_param import FileFile

from gateway.config import get_files_api_client
from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import RawJsonResponse, RawStreamingResponse, error_response
from management.models import FileObject, Token

log = logging.getLogger("aqueduct")


async def extract_text_with_tika(file_bytes: bytes) -> str:
    """Extract text from file bytes using Tika API."""
    tika_url = f"{getattr(settings, 'TIKA_SERVER_URL', 'http://localhost:9998')}/tika"

    async with httpx.AsyncClient() as client:
        response = await client.put(tika_url, content=file_bytes, headers={}, timeout=30.0)
        response.raise_for_status()
        return response.text


async def file_to_bytes(token: Token | None, file: FileFile) -> bytes:
    """Convert file description to bytes and content type."""
    file_id = file.get("file_id", None)
    file_data = file.get("file_data", None)

    if file_data:
        # file data contains b64 encoded files -> decode as bytes
        try:
            header, file_b64 = file_data.split(
                ",", maxsplit=1
            )  # removes data uri (data:application/pdf;base64,<b64>)
            if not header.startswith("data:"):
                raise ValueError("Incorrect data URI for base64 encoded file.")
            return base64.b64decode(file_b64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 file data: {e}") from e

    elif file_id:
        # file is given as an id of a file object
        try:
            if token and token.service_account:
                file_obj = await FileObject.objects.select_related("token__user").aget(
                    id=file_id, token__service_account__team=token.service_account.team
                )
            elif token:
                file_obj = await FileObject.objects.select_related("token__user").aget(
                    id=file_id, token__user=token.user
                )
            else:
                file_obj = await FileObject.objects.select_related("token__user").aget(id=file_id)

            try:
                client = get_files_api_client()
            except ValueError as e:
                raise ValueError(f"Files API not configured: {e}") from e
        except FileObject.DoesNotExist:
            raise
        except Exception as e:
            raise ValueError(f"Failed to read file with id {file_id}: {e}") from e
        else:
            response = await client.files.content(file_obj.id)
            return response.content
    else:
        raise RuntimeError("Neither 'file_data' nor 'file_id' are given.")


def process_file_content(view_func: AsyncView) -> AsyncView:
    """Decorator to process file content in chat completions using Tika."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not pydantic_model:
            log.error("Invalid request - missing request body")
            return error_response("Invalid request: missing request body", status=400)

        messages = pydantic_model.get("messages", [])
        if not messages:
            return await view_func(request, *args, **kwargs)

        # Process messages to extract text from file content
        total_file_size_bytes = 0
        max_total_size_mb = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB
        max_total_size_bytes = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB * 1024 * 1024
        max_file_mb = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB
        max_file_bytes = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB * 1024 * 1024
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "file":
                    file = FileFile(**content_item.get("file", {}))  # type: ignore[typeddict-item]
                    try:
                        file_bytes = await file_to_bytes(token, file)
                    except FileObject.DoesNotExist:
                        log.exception("File not found")
                        return error_response("File not found", status=404)
                    except Exception as e:
                        # return json response here if there was an error
                        log.exception("Error processing file - %s", e)
                        return error_response(f"Error processing file: {e!s}", status=400)

                    if len(file_bytes) > max_file_bytes:
                        log.error(
                            "File processing error - File too large "
                            "(individual file must be <= %sMB)",
                            max_file_mb,
                        )
                        return error_response(
                            f"Error processing file content: File too large. "
                            f"Individual file must be <= {max_file_mb}MB.",
                            status=400,
                        )
                    total_file_size_bytes += len(file_bytes)
                    if total_file_size_bytes > max_total_size_bytes:
                        log.error(
                            "File processing error - Files too large in total "
                            "(all files must be <= %sMB)",
                            max_total_size_mb,
                        )
                        return error_response(
                            f"Error processing file content: Files too large in total. "
                            f"All files must be <= {max_total_size_mb}MB.",
                            status=400,
                        )

                    # Extract text using Tika
                    try:
                        extracted_text = await extract_text_with_tika(file_bytes)
                    except httpx.HTTPStatusError as e:
                        # return json response here if there was a tika request error
                        log.exception("Tika error extracting text from file - %s", e)
                        return error_response(
                            f"Tika error extracting text from file: {e!s}", status=400
                        )

                    extracted_text = (
                        f"Content of user-uploaded file "
                        f"'{file.get('filename', 'unknown filename')}':"
                        f"\n---\n{extracted_text}\n---"
                    )

                    # Replace file content with extracted text
                    content_item["type"] = "text"
                    content_item["text"] = extracted_text
                    del content_item["file"]

        return await view_func(request, *args, **kwargs)

    return wrapper


def _normalize_reasoning_in_message(message: dict[str, Any]) -> bool:
    """Ensure both 'reasoning' and 'reasoning_content' are present if either exists.

    Returns True if the message was modified.
    """
    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")

    if reasoning_content is not None and reasoning is None:
        message["reasoning"] = reasoning_content
        return True
    if reasoning is not None and reasoning_content is None:
        message["reasoning_content"] = reasoning
        return True
    return False


def normalize_reasoning_fields(view_func: AsyncView) -> AsyncView:
    """Normalize reasoning/reasoning_content fields in chat completion responses.

    Ensures that if a response message contains either 'reasoning' or
    'reasoning_content', both fields are present with the same value.
    This provides compatibility for clients that expect one field name
    or the other.

    Handles both streaming and non-streaming responses.
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        result = await view_func(request, *args, **kwargs)

        if isinstance(result, RawStreamingResponse):

            def _normalized_stream(chunk: ModelResponseStream) -> ModelResponseStream:
                choices = chunk.get("choices", [])
                for choice in choices:
                    # Streaming chunks use "delta"; final chunks use "message"
                    message = choice.get("delta") or {}
                    if message:
                        _normalize_reasoning_in_message(message)
                return chunk

            result.transforms.append(_normalized_stream)

        elif isinstance(result, RawJsonResponse):
            content = cast("dict[str, Any] | ModelResponse", result.content)
            choices = content.get("choices", [])
            for choice in choices:
                message = choice.get("message", {})
                if message:
                    _normalize_reasoning_in_message(message)

        return result

    return wrapper
