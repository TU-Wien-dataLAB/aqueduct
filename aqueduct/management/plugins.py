from __future__ import annotations

import inspect
import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ValidationError

from management.models import Snippet, SnippetType

if TYPE_CHECKING:
    from gateway.response_type import RawJsonResponse, RawStreamingResponse

log = logging.getLogger("aqueduct")


class PluginSnippet:
    #: Normalized endpoint paths this plugin applies to (e.g. ``"chat/completions"``).
    #: Empty means the plugin applies to every endpoint.
    #: Different API URLs have different request types, so pick the matching hook:
    #: JSON-only endpoints (e.g. ``embeddings``, ``vector_stores/...``) are non-streaming;
    #: streaming-capable endpoints (e.g. ``chat/completions``, ``responses``) can be streaming.
    endpoints: ClassVar[list[str]] = []

    #: Internal bookkeeping for per-hook timing (not part of the plugin API).
    processing_time_ms: int = 0

    def before_request(self, *args: Any, **kwargs: Any) -> Any | None:
        """Optionally transform/replace the request body.

        The gateway calls this hook as ``before_request(request, token, body)``; the
        ``*args``/``**kwargs`` signature lets a subclass name only the parameters it
        needs. Return the (possibly mutated) body to pass downstream, or ``None``
        to leave the body unchanged. Raise :class:`BlockedByPlugin` to stop the request.
        """

    def after_response(self, response: RawJsonResponse) -> RawJsonResponse | None:
        """Run for non-streaming responses only (``RawJsonResponse``).

        Requires a non-streaming request type — i.e. a JSON-only API URL.
        Return a replacement response, or ``None`` to keep the response unchanged.
        """

    def after_streaming_response(
        self, response: RawStreamingResponse
    ) -> RawStreamingResponse | None:
        """Run for streaming responses only (``RawStreamingResponse``) — headers/status,
        not content.

        Requires a streaming request type — i.e. a streaming-capable API URL
        (e.g. ``chat/completions``). Return a replacement response, or ``None``
        to keep the response unchanged.
        """


class BlockedByPlugin(Exception):  # noqa: N818
    def __init__(self, reason: str, status: int = 403) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _record_call(plugin: PluginSnippet, start: float) -> None:
    duration_ms = int((time.monotonic() - start) * 1000)
    plugin.processing_time_ms = duration_ms
    log.debug("Plugin %s hook took %d ms", type(plugin).__name__, duration_ms)


def _endpoint(request: Any) -> str:
    """Return the normalized endpoint path for ``request`` (no leading/v1 prefix)."""
    return str(request.path.strip("/").removeprefix("v1/"))


def _plugin_applies(plugin: PluginSnippet, request: Any) -> bool:
    """Whether ``plugin`` should run for this request's endpoint.

    A plugin with an empty ``endpoints`` list applies to every endpoint;
    otherwise the normalized request path must be listed in ``endpoints``.
    """
    endpoints = plugin.endpoints
    if not endpoints:
        return True
    return _endpoint(request) in endpoints


def before_hook(plugins: list[PluginSnippet], request: Any, token: Any, body: Any) -> Any:
    for plugin in plugins:
        if not _plugin_applies(plugin, request):
            continue
        start = time.monotonic()
        result = plugin.before_request(request, token, body)
        _record_call(plugin, start)
        if result is not None:
            body = result
    return body


def after_hook(plugins: list[PluginSnippet], request: Any, response: Any) -> Any:
    """Run non-streaming ``after_response`` hooks in order.

    Each plugin may return a replacement response, which becomes the input to
    the next plugin; the final response is returned.
    """
    for plugin in plugins:
        if not _plugin_applies(plugin, request):
            continue
        start = time.monotonic()
        result = plugin.after_response(response)
        _record_call(plugin, start)
        if result is not None:
            response = result
    return response


def after_streaming_hook(plugins: list[PluginSnippet], request: Any, response: Any) -> Any:
    """Run streaming ``after_streaming_response`` hooks in order (headers/status only).

    Each plugin may return a replacement response, which becomes the input to
    the next plugin; the final response is returned.
    """
    for plugin in plugins:
        if not _plugin_applies(plugin, request):
            continue
        start = time.monotonic()
        result = plugin.after_streaming_response(response)
        _record_call(plugin, start)
        if result is not None:
            response = result
    return response


def compile_plugin_class(code: str) -> type[PluginSnippet]:
    try:
        source = compile(code, "<plugin>", "exec")
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}") from e

    namespace: dict[str, Any] = {"PluginSnippet": PluginSnippet, "BlockedByPlugin": BlockedByPlugin}
    try:
        exec(source, namespace)  # noqa: S102
    except Exception as e:
        raise ValidationError(f"Error executing plugin code: {e}") from e

    subclasses = [
        value
        for value in namespace.values()
        if inspect.isclass(value)
        and issubclass(value, PluginSnippet)
        and value is not PluginSnippet
    ]
    if not subclasses:
        raise ValidationError(
            "Snippet must define exactly one class that subclasses 'PluginSnippet'."
        )
    if len(subclasses) > 1:
        raise ValidationError(
            "Snippet must define exactly one class that subclasses 'PluginSnippet' "
            f"(found {len(subclasses)})."
        )
    return subclasses[0]


@lru_cache(maxsize=128)
def _plugin_class(pk: int) -> type[PluginSnippet]:
    snippet = Snippet.objects.get(pk=pk)
    return compile_plugin_class(snippet.code)


def resolve_active_plugins() -> list[PluginSnippet]:
    ids = list(
        Snippet.objects.filter(type=SnippetType.PLUGIN, active=True)
        .order_by("order", "id")
        .values_list("pk", flat=True)
    )
    return [_plugin_class(pk)() for pk in ids]
