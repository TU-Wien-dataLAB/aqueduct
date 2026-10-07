"""Runtime resolution of database-stored configuration snippets.
When no active config snippet exists, ConfigSnippet is used.

Snippet code is a Python source string that subclasses :class: ConfigSnippet.
Admins override only the methods they need; the base class provides the rest.
"""

import inspect
import logging
from typing import Any

from django.core.exceptions import ValidationError

from management.models import Snippet, SnippetType, UserGroup

log = logging.getLogger("aqueduct")

DEFAULT_ORG_NAME = "default"
DEFAULT_USER_GROUP = UserGroup.USER
METHOD_SELF_PLUS_ARG_COUNT = 2

CONFIG_METHODS = ("org_name", "team_names", "display_team_names", "user_group")


class ConfigSnippet:
    def org_name(self, claims: dict[str, Any]) -> str | None:
        return DEFAULT_ORG_NAME

    def team_names(self, claims: dict[str, Any]) -> list[str]:
        return []

    def display_team_names(self, team_names: list[str]) -> list[tuple[str, str]]:
        return [(team_name, team_name) for team_name in team_names]

    def user_group(self, claims: dict[str, Any]) -> str:
        return DEFAULT_USER_GROUP


def compile_snippet(code: str, require_subclass: bool = True) -> ConfigSnippet | None:
    try:
        source = compile(code, "<snippet>", "exec")
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}") from e

    if not require_subclass:
        return None

    namespace: dict[str, Any] = {"ConfigSnippet": ConfigSnippet}
    try:
        exec(source, namespace)  # noqa: S102 - superuser-only, no sandboxing
    except Exception as e:
        raise ValidationError(f"Error executing snippet code: {e}") from e

    subclasses = [
        value
        for value in namespace.values()
        if inspect.isclass(value)
        and issubclass(value, ConfigSnippet)
        and value is not ConfigSnippet
    ]
    if not subclasses:
        raise ValidationError(
            "Snippet must define exactly one class that subclasses 'ConfigSnippet'."
        )
    if len(subclasses) > 1:
        raise ValidationError(
            "Snippet must define exactly one class that subclasses 'ConfigSnippet' "
            f"(found {len(subclasses)})."
        )
    cls = subclasses[0]

    _validate_signatures(cls)

    try:
        return cls()
    except Exception as e:
        raise ValidationError(f"Could not instantiate snippet class: {e}") from e


def _validate_signatures(cls: type) -> None:
    for method_name in CONFIG_METHODS:
        method = getattr(cls, method_name, None)
        if not callable(method):
            raise ValidationError(f"Snippet class must define a '{method_name}' method.")
        params = list(inspect.signature(method).parameters.values())
        if len(params) != METHOD_SELF_PLUS_ARG_COUNT or params[0].name != "self":
            raise ValidationError(
                f"'{method_name}' must be an instance method taking exactly one "
                f"argument, e.g. 'def {method_name}(self, claims)'."
            )


def get_config_snippet() -> ConfigSnippet:
    snippet = Snippet.objects.filter(type=SnippetType.CONFIG, active=True).first()
    if not snippet:
        return ConfigSnippet()
    return compile_snippet(snippet.code)
