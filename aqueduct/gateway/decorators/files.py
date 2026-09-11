import json
import logging
import sys
from functools import wraps
from typing import Any

import litellm
from django.core.handlers.asgi import ASGIRequest

from gateway.config import get_files_api_client, get_router, resolve_model_alias
from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import error_response

log = logging.getLogger("aqueduct")


def require_files_api_client(view_func: AsyncView) -> AsyncView:
    """Decorator that injects a files API client into the view kwargs, or returns 503.

    Uses late-bound import of get_files_api_client so that tests can
    patch it on the calling module (e.g. gateway.views.vector_stores).
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # Look up get_files_api_client from the module where view_func is defined,
        # so tests patching that module's reference will be respected.
        view_module = sys.modules.get(view_func.__module__)
        _get_client = getattr(view_module, "get_files_api_client", get_files_api_client)

        try:
            client = _get_client()
        except ValueError:
            return error_response("Vector Store API not configured", status=503)
        kwargs["client"] = client
        return await view_func(request, *args, **kwargs)

    return wrapper


def _lookup_relay_model_name(requested_model: str) -> str | None:
    """Return upstream relay model for a configured deployment, else None."""
    router = get_router()
    if not router:
        return requested_model

    requested_model = resolve_model_alias(requested_model)
    deployment: litellm.Deployment | None = router.get_deployment(model_id=requested_model)
    if not deployment:
        return None

    litellm_params = getattr(deployment, "litellm_params", None)
    deployment_model = getattr(litellm_params, "model", None)
    if not deployment_model:
        return None

    relay_model: str
    relay_model, _, _, _ = litellm.get_llm_provider(deployment_model)
    return relay_model


def rewrite_batch_file_models(content: bytes) -> bytes:
    """
    Rewrite model names in a batch input JSONL file.

    Each line is a JSON object with structure:
    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "relay-model-name", ...}}

    This function:
    1. Ensures custom_id is a string type (required by OpenAI Batch API)
    2. Replaces the model name in each request body with the actual upstream model name
    3. Rejects unknown models with a ValueError that includes custom_id
    """
    lines = content.decode("utf-8").splitlines()
    rewritten_lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            body = request.get("body", {})
            if "model" in body:
                original_model = body["model"]
                relay_model = _lookup_relay_model_name(original_model)
                if relay_model is None:
                    custom_id = request.get("custom_id", "<missing custom_id>")
                    raise ValueError(
                        f"Unknown model '{original_model}' for custom_id '{custom_id}'"
                    )
                body["model"] = relay_model
                request["body"] = body
            if "custom_id" in request:
                request["custom_id"] = str(request["custom_id"])
            rewritten_lines.append(json.dumps(request, separators=(",", ":")))
        except json.JSONDecodeError:
            # Keep invalid lines as-is (will fail validation anyway)
            rewritten_lines.append(line)

    return ("\n".join(rewritten_lines) + "\n").encode("utf-8")


def extract_preview(content: bytes, num_lines: int = 10) -> str:
    """Extract first 10 lines from file content for preview."""
    try:
        lines = content.decode("utf-8").splitlines()[:num_lines]
        return "\n".join(lines)
    except UnicodeDecodeError:
        return "[Binary content - no preview available]"


def process_batch_file(view_func: AsyncView) -> AsyncView:
    """
    Decorator for batch file upload that processes file content before proxying.

    This decorator (applied after @parse_body):
    1. Reads the file content ONCE from the uploaded file
    2. Extracts a preview (first 10 lines) of the original content
    3. For batch files: rewrites model names using router config
    4. Passes processed data to the view via kwargs:
       - file_content: bytes - the raw file content (or rewritten for batch)
       - file_preview: str | None - first 10 lines for batch files

    This follows the existing decorator pattern (like @parse_body) of reading
    data once and passing it via kwargs to avoid multiple reads.
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        if request.method != "POST":
            return await view_func(request, *args, **kwargs)

        pydantic_model = kwargs.get("pydantic_model")
        if not pydantic_model:
            return await view_func(request, *args, **kwargs)

        uploaded = pydantic_model.get("file")
        purpose = pydantic_model.get("purpose")

        if not uploaded:
            return await view_func(request, *args, **kwargs)

        # Read file content ONCE
        content = uploaded.read()

        if purpose == "batch":
            kwargs["file_preview"] = extract_preview(content)
            try:
                kwargs["file_content"] = rewrite_batch_file_models(content)
            except ValueError as e:
                return error_response(f"Batch file validation failed: {e!s}", status=400)
        else:
            kwargs["file_content"] = content
            kwargs["file_preview"] = ""

        return await view_func(request, *args, **kwargs)

    return wrapper
