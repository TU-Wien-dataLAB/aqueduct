from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from django import forms
from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import SafeString
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _

from gateway.config import get_mcp_config, model_names


def configured_mcp_servers() -> list[str]:
    try:
        return list(get_mcp_config())
    except RuntimeError:
        return []


@dataclass(frozen=True)
class Reach:
    configured: int
    usable: int

    @property
    def every(self) -> bool:
        return self.usable == self.configured

    @classmethod
    def of(cls, configured: Sequence[str], excluded: Collection[str]) -> "Reach":
        return cls(len(configured), len(set(configured) - set(excluded)))


CHECKLISTS: dict[str, tuple[str, Callable[[], Sequence[str]]]] = {
    "available_models": ("excluded_models", model_names),
    "available_mcp_servers": ("excluded_mcp_servers", configured_mcp_servers),
}
INHERIT_LABELS = {
    "merge_exclusion_lists": _("Also apply the organization's unavailable models"),
    "merge_mcp_server_exclusion_lists": _("Also apply the organization's unavailable MCP servers"),
}
INHERIT_HELP = {
    "merge_exclusion_lists": _("Clear this to use only the boxes above."),
    "merge_mcp_server_exclusion_lists": _("Clear this to use only the boxes above."),
}
ORDER = (
    "available_models",
    "merge_exclusion_lists",
    "available_mcp_servers",
    "merge_mcp_server_exclusion_lists",
)


SHOWN = forms.MultipleHiddenInput()


class Checklist(forms.CheckboxSelectMultiple):
    def render(self, name, value, attrs=None, renderer=None) -> SafeString:
        checklist = super().render(name, value, attrs, renderer)
        shown = [choice for choice, _ in self.choices]
        return format_html("{}{}", checklist, SHOWN.render(f"{name}-shown", shown))


class AvailabilityForm(forms.ModelForm):
    available_models = forms.MultipleChoiceField(
        required=False,
        widget=Checklist,
        label=_("Available models"),
        help_text=_("Clear a box to make the model unavailable."),
    )
    available_mcp_servers = forms.MultipleChoiceField(
        required=False,
        widget=Checklist,
        label=_("Available MCP servers"),
        help_text=_("Clear a box to make the MCP server unavailable."),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name in ORDER:
            if name in self.fields:
                self.fields[name] = self.fields.pop(name)
        for name, (stored, configured) in CHECKLISTS.items():
            if name not in self.fields:
                continue
            excluded: list[str] = getattr(self.instance, stored)
            known = configured()
            missing = [n for n in excluded if n not in known]
            self.fields[name].choices = [(n, n) for n in known] + [
                (n, gettext("%(name)s (not in the current config)") % {"name": n}) for n in missing
            ]
            self.initial[name] = [n for n in known if n not in excluded]

    def save(self, commit: bool = True) -> Any:
        for name, (stored, _configured) in CHECKLISTS.items():
            shown = SHOWN.value_from_datadict(
                self.data, self.files, f"{self.add_prefix(name)}-shown"
            )
            if name not in self.cleaned_data or not shown:
                continue
            checked = set(self.cleaned_data[name])
            offered = {value for value, _ in self.fields[name].choices}
            excluded: list[str] = getattr(self.instance, stored)
            kept = [n for n in excluded if n not in checked]
            cleared = [n for n in dict.fromkeys(shown) if n in offered and n not in checked]
            setattr(self.instance, stored, list(dict.fromkeys(kept + cleared)))
        return super().save(commit)


class AvailabilityAdmin:
    """Shows the checklists as text to staff who may view but not change the object."""

    @admin.display(description=_("Available models"))
    def available_models(self, obj: Any) -> str:
        return _available(obj, "available_models")

    @admin.display(description=_("Available MCP servers"))
    def available_mcp_servers(self, obj: Any) -> str:
        return _available(obj, "available_mcp_servers")


def _available(obj: Any, name: str) -> str:
    stored, configured = CHECKLISTS[name]
    excluded = getattr(obj, stored)
    return ", ".join(n for n in configured() if n not in excluded) or "—"
