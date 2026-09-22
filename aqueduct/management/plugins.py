import inspect
import logging
import time
from functools import lru_cache
from typing import Any

from django.core.exceptions import ValidationError

from management.models import Snippet, SnippetType

log = logging.getLogger("aqueduct")


class PluginSnippet:
    processing_time_ms: int = 0

    def before_request(self, request: Any, token: Any, body: Any) -> Any:
        pass

    def after_response(self, request: Any, token: Any, response: Any) -> None:
        pass

    def on_error(self, request: Any, exc: Exception) -> None:
        pass


class BlockedByPlugin(Exception):  # noqa: N818
    def __init__(self, reason: str, status: int = 403) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _record_call(plugin: PluginSnippet, start: float) -> None:
    duration_ms = int((time.monotonic() - start) * 1000)
    plugin.processing_time_ms = duration_ms
    log.debug("Plugin %s hook took %d ms", type(plugin).__name__, duration_ms)


def before_hook(plugins: list[PluginSnippet], request: Any, token: Any, body: Any) -> Any:
    for plugin in plugins:
        call = getattr(plugin, "before_request", None)
        if call is None:
            continue
        start = time.monotonic()
        result = call(request, token, body)
        _record_call(plugin, start)
        if result is not None:
            body = result
    return body


def after_hook(plugins: list[PluginSnippet], request: Any, token: Any, response: Any) -> None:
    for plugin in plugins:
        call = getattr(plugin, "after_response", None)
        if call is None:
            continue
        start = time.monotonic()
        call(request, token, response)
        _record_call(plugin, start)


def _error_hook(plugins: list[PluginSnippet], request: Any, exc: Exception) -> None:
    for plugin in plugins:
        call = getattr(plugin, "on_error", None)
        if call is None:
            continue
        start = time.monotonic()
        call(request, exc)
        _record_call(plugin, start)


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
