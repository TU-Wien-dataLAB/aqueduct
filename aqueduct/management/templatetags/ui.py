"""Presentation helpers for the management templates: icons, copy buttons, confirm dialogs."""

from django import template
from django.forms import BoundField
from django.template.loader import get_template
from django.utils.html import format_html
from django.utils.safestring import SafeString, mark_safe
from django.utils.translation import gettext

register = template.Library()


@register.simple_tag
def icon(name: str, small: bool = False, extra_class: str = "") -> SafeString:
    """An icon from `management/icons/<name>.svg`, drawn in the text colour."""
    try:
        shapes = get_template(f"management/icons/{name}.svg").render()
    except template.TemplateDoesNotExist:
        raise template.TemplateSyntaxError(f"unknown icon {name!r}") from None
    classes = " ".join(filter(None, ("aq-ic", "sm" if small else "", extra_class)))
    return format_html(
        '<svg class="{}" width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">{}</svg>',
        classes,
        mark_safe(shapes.strip()),  # noqa: S308 - our own template files
    )


@register.simple_tag
def copy_button(value: str, label: str, size: str = "xs") -> SafeString:
    """An icon button that copies `value`; aqueduct.js does the copying."""
    return format_html(
        '<button type="button" class="btn btn-ghost btn-{} btn-square" data-copy="{}" '
        'aria-label="{}" title="{}">{}</button>',
        size,
        value,
        label,
        label,
        icon("copy", small=True),
    )


@register.simple_tag
def confirm(title: str, detail: str = "", action: str = "") -> SafeString:
    """Attributes that make a submit button or link ask in the confirm dialog first."""
    return format_html(
        'data-confirm="{}" data-confirm-detail="{}" data-confirm-action="{}"',
        title,
        detail,
        action or gettext("Delete"),
    )


@register.filter
def widget(field: BoundField) -> SafeString:
    """The field's widget, with daisyUI's error style when the field has errors."""
    classes = field.field.widget.attrs.get("class", "")
    if not (field.errors and classes):
        return field.as_widget()
    component = classes.split()[0]
    return field.as_widget(attrs={"class": f"{classes} {component}-error"})
