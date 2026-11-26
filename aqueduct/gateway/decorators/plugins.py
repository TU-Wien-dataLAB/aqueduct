import types
import typing
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, cast, overload

from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, Decorator, ViewResult
from gateway.response_error import error_response
from gateway.response_type import RawStreamingResponse
from management.plugins import (
    BlockedByPlugin,
    PluginSnippet,
    after_hook,
    after_streaming_hook,
    before_hook,
)

ResponseKind = Literal["json", "streaming", "json_or_streaming"]


def _type_name(t: Any) -> str:
    """A short, readable name for a request-body type.

    OpenAI ``CreateParams`` are often ``NonStreaming | Streaming`` unions, which
    report ``__name__ == "Union"``. Strip those suffixes to produce a stable base
    name (e.g. ``CompletionCreateParams``).
    """
    name = getattr(t, "__name__", None)
    if name and name != "Union":
        return str(name)
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if args and origin in (typing.Union, types.UnionType):
        bases = set()
        for arg in args:
            member = _type_name(arg)
            for suffix in ("NonStreaming", "Streaming"):
                if member.endswith(suffix):
                    member = member[: -len(suffix)]
                    break
            bases.add(member)
        if len(bases) == 1:
            return bases.pop()
        return " | ".join(sorted(bases))
    return str(t)


@dataclass(frozen=True)
class PluginEndpoint:
    """Metadata for a plugin-capable endpoint, used by the admin registry."""

    key: str
    pydantic_type: Any
    response_kind: ResponseKind

    @property
    def type_name(self) -> str:
        """A short, readable name for the request-body type."""
        return _type_name(self.pydantic_type)


#: Registry of plugin-capable endpoints, keyed by normalized path (no ``/v1/``).
#: Populated at import time by ``@run_plugins(endpoint_key=...)``.
_ENDPOINT_REGISTRY: dict[str, PluginEndpoint] = {}


def register_plugin_endpoint(
    endpoint_key: str, pydantic_type: Any = None, response_kind: ResponseKind = "json"
) -> None:
    """Register a plugin-capable endpoint in the admin registry."""
    _ENDPOINT_REGISTRY[endpoint_key] = PluginEndpoint(
        key=endpoint_key, pydantic_type=pydantic_type, response_kind=response_kind
    )


def get_plugin_endpoint_registry() -> dict[str, PluginEndpoint]:
    """Return the registry of plugin-capable endpoints.

    The registry is populated when the gateway views are imported. If it is still
    empty (e.g. the admin is rendered before the urlconf is loaded), importing the
    urlconf here triggers the view decorators to run and populate it.
    """
    if not _ENDPOINT_REGISTRY:
        from gateway import urls as _urls  # noqa: F401  (imports views -> registers endpoints)

    return dict(_ENDPOINT_REGISTRY)


def generate_plugin_stub(endpoints: list[dict[str, Any]], class_name: str = "MyPlugin") -> str:
    """Build a runnable ``PluginSnippet`` stub for the given endpoints.

    ``endpoints`` is a list of dicts with ``key``, ``type_name`` and
    ``response_kind`` (the shape produced by ``get_plugin_endpoint_registry()``).
    """
    keys = sorted({e["key"] for e in endpoints})
    type_names = sorted({e["type_name"] for e in endpoints if e.get("type_name")})
    has_json = any(e.get("response_kind") in ("json", "json_or_streaming") for e in endpoints)
    has_streaming = any(
        e.get("response_kind") in ("streaming", "json_or_streaming") for e in endpoints
    )

    # When a single endpoint is chosen, type ``body`` with its request-body model so
    # the generated hook documents the fields available. Multiple types (or none) fall
    # back to ``dict``; ``from __future__ import annotations`` keeps the annotation a
    # string so it never needs the model imported at runtime.
    body_type = type_names[0] if len(type_names) == 1 else "dict"

    lines = ["from __future__ import annotations", "", f"class {class_name}(PluginSnippet):"]
    if keys:
        lines.append(f"    # Runs on: {', '.join(keys)}")
        lines.append(f"    endpoints = [{', '.join(repr(k) for k in keys)}]")
    else:
        lines.append("    # No endpoints selected: the plugin runs on every endpoint.")
        lines.append("    endpoints = []")
    lines.append("")
    lines.append(f"    def before_request(self, request, token, body: {body_type}) -> dict | None:")
    if type_names:
        lines.append(f"        # Request body is validated against: {', '.join(type_names)}")
    lines.append(
        '        """Optionally transform/replace the request body.\n'
        "\n"
        "        Return the (possibly mutated) body to pass downstream, or ``None`` to\n"
        "        leave the body unchanged. Raise ``BlockedByPlugin`` to stop the request.\n"
        '        """'
    )
    lines.append("        return body")
    lines.append("")
    if has_json:
        lines.append("    def after_response(self, response) -> None:")
        lines.append(
            '        """Non-streaming only. Return a replacement response, or ``None`` '
            'to keep unchanged."""'
        )
        lines.append("        return None")
        lines.append("")
    if has_streaming:
        lines.append("    def after_streaming_response(self, response) -> None:")
        lines.append('        """Streaming wrapper only (headers/status, not content)."""')
        lines.append("        return None")
    return "\n".join(lines).rstrip() + "\n"


@overload
def run_plugins(
    view_func: AsyncView,
    *,
    endpoint_key: str | None = None,
    pydantic_type: Any = None,
    response_kind: ResponseKind = "json",
) -> AsyncView: ...


@overload
def run_plugins(
    view_func: None = None,
    *,
    endpoint_key: str | None = None,
    pydantic_type: Any = None,
    response_kind: ResponseKind = "json",
) -> Decorator: ...


def run_plugins(
    view_func: AsyncView | None = None,
    *,
    endpoint_key: str | None = None,
    pydantic_type: Any = None,
    response_kind: ResponseKind = "json",
) -> AsyncView | Decorator:
    """Wrap a view to run plugins and register it as plugin-capable.

    Use as ``@run_plugins(endpoint_key=..., pydantic_type=..., response_kind=...)``
    to register the endpoint in the admin registry, or as a bare ``@run_plugins``
    decorator for backwards compatibility.
    """
    if endpoint_key is not None:
        register_plugin_endpoint(endpoint_key, pydantic_type, response_kind)

    def decorator(fn: AsyncView) -> AsyncView:
        @wraps(fn)
        async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
            plugins: list[PluginSnippet] = getattr(request, "active_plugins", [])
            if not plugins:
                return await fn(request, *args, **kwargs)

            token = kwargs.get("token")
            body = kwargs.get("pydantic_model")

            try:
                final_body = before_hook(plugins, request, token, body)
            except BlockedByPlugin as e:
                return error_response(e.reason, status=e.status)

            if final_body is not None:
                kwargs["pydantic_model"] = final_body

            response = await fn(request, *args, **kwargs)

            if isinstance(response, RawStreamingResponse):
                return cast("ViewResult", after_streaming_hook(plugins, request, response))
            return cast("ViewResult", after_hook(plugins, request, response))

        return wrapper

    if view_func is not None:
        return decorator(view_func)
    return decorator
