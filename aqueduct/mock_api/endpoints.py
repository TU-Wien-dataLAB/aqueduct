import asyncio
import logging.config
import os
import secrets
from json import JSONDecodeError

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.status import HTTP_404_NOT_FOUND

from mock_api.mock_configs import (
    MockConfig,
    MockPlainTextConfig,
    MockStreamingConfig,
    default_delete_configs,
    default_get_configs,
    default_post_configs,
    default_post_stream_configs,
    special_configs,
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler", "formatter": "standard",
        }
    },
    "loggers": {
        "fastapi": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        }
    },
}  # fmt: skip
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fastapi")


delays_enabled = os.getenv("MOCK_API_DELAYS", "false").lower() == "true"


class MockException(HTTPException):
    """Raised to simulate an exception while calling the external API."""


app = FastAPI(debug=True)


@app.get("/health")
async def health_check() -> JSONResponse:
    logger.debug("health check successful")
    return JSONResponse({"status": "ok"})


@app.post("/configure/{path:path}")
async def configure_endpoint(
    path: str, config: MockConfig | MockStreamingConfig | MockPlainTextConfig
) -> dict[str, str]:
    """
    Configure a special mock response for a specific endpoint.

    This endpoint allows to dynamically configure a mock response for any endpoint,
    regardless of the request method (GET, POST, DELETE). It can be used when the default
    response from `default_{method}_configs` is not what one needs.
    """
    special_configs[path] = config
    logger.debug("Configured a special mock response for %s", path)
    return {"message": f"Configured a special mock response for {path}"}


@app.post("/reset/{path:path}")
async def reset_endpoint(path: str) -> dict[str, str]:
    """
    Reset the special mock response for a specific endpoint to its default behavior.

    A request to this endpoint should be sent after a test with a special mock response
    finishes, to prevent tests from interfering with one another.
    """
    del special_configs[path]
    logger.debug("Reset the special mock response for %s", path)
    return {"message": f"Reset the special mock response for {path}"}


@app.delete("/{path:path}")
@app.get("/{path:path}")
@app.post("/{path:path}")
async def mock_endpoint(path: str, request: Request) -> JSONResponse | PlainTextResponse | StreamingResponse:
    """
    The endpoint that mocks responses from the external OpenAI API.

    If `MOCK_API_DELAYS` env var is set to true, a random delay of 0.1-1 s is added
    before returning a response.
    """
    path = path.strip("/").removeprefix("v1/")

    try:
        if path in special_configs:
            # Special config can be a streaming, a JSON, or a plain text one.
            config = special_configs[path]
        elif await _should_stream(request):
            # Some streaming configs are stored in a separate dict, because some
            # endpoints can return both a streaming or a JSON response.
            # Note: streaming responses can only be returned for POST requests.
            config = default_post_stream_configs[path]
        else:
            if request.method == "POST":
                config_dict = default_post_configs
            elif request.method == "GET":
                config_dict = default_get_configs
            elif request.method == "DELETE":
                config_dict = default_delete_configs
            else:
                raise ValueError(f"No mocks configured for request method {request.method}")
            config = _find_config_for_path(path, config_dict)
    except KeyError as err:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No mock configured for this endpoint: {path}"
        ) from err

    logger.debug("Got a request to %s", path)

    if delays_enabled:
        delay = 0.1 + (secrets.randbelow(900) / 1000)
        logger.debug("Adding %.2fs delay to request", delay)
        await asyncio.sleep(delay)

    if isinstance(config, MockStreamingConfig):
        return StreamingResponse(content=config.response_data, status_code=config.status_code, headers=config.headers)
    if isinstance(config, MockPlainTextConfig):
        return PlainTextResponse(content=config.response_data, status_code=config.status_code, headers=config.headers)
    return JSONResponse(content=config.response_data, status_code=config.status_code, headers=config.headers)


async def _should_stream(request: Request) -> bool:
    try:
        request_data = await request.json()
        should_stream = request_data.get("stream") is True
    except (AttributeError, JSONDecodeError, UnicodeDecodeError):
        should_stream = False
    return should_stream


def _find_config_for_path(path: str, config_dict: dict[str, MockConfig]) -> MockConfig:
    """Find the matching config template and its value for a given path."""
    # Try exact match first (for paths without IDs)
    if path in config_dict:
        return config_dict[path]

    # Try template matching
    for template, config in config_dict.items():
        if _path_matches_template(path, template):
            return config

    raise KeyError(f"No config found for path: {path}")


def _path_matches_template(path: str, template: str) -> bool:
    """
    Check if a path matches a template where the string "id" is a wildcard.

    Examples:
        "batches/batch-abc123/cancel" matches "batches/id/cancel"
        "vector_stores/vs-mock-123/files" matches "vector_stores/id/files"
    """
    path_segments = path.split("/")
    template_segments = template.split("/")

    if len(path_segments) != len(template_segments):
        return False

    for path_seg, template_seg in zip(path_segments, template_segments, strict=True):
        if template_seg == "id":
            continue  # 'id' matches anything
        if path_seg != template_seg:
            return False

    return True
