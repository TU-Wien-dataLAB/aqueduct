from collections.abc import Iterable
from datetime import UTC, date, datetime

from django import template
from django.forms import BoundField
from django.template.defaultfilters import date as format_date
from django.template.defaulttags import querystring
from django.urls import reverse
from django.utils import dateformat, timezone
from django.utils.html import conditional_escape, format_html, format_html_join
from django.utils.http import urlencode
from django.utils.safestring import SafeString, mark_safe
from django.utils.translation import gettext, ngettext_lazy

from management.formatting import MISSING, grouped
from management.icons import Icon

register = template.Library()


@register.simple_tag
def icon(name: Icon | str, small: bool = False, extra_class: str = "") -> SafeString:
    if isinstance(name, str):
        try:
            name = Icon[name.upper()]
        except KeyError:
            raise template.TemplateSyntaxError(
                f"unknown icon {name!r}; add it to management/icons.py"
            ) from None
    classes = " ".join(filter(None, ("aq-ic", "sm" if small else "", extra_class)))
    body = format_html_join(
        "",
        "<{} {}/>",
        ((tag, format_html_join(" ", '{}="{}"', attrs)) for tag, attrs in name.value),
    )
    return format_html(
        '<svg class="{}" width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">{}</svg>',
        classes,
        body,
    )


@register.simple_tag
def copy_button(
    value: str, label: str, title: str = "", size: str = "xs", extra_class: str = ""
) -> SafeString:
    classes = " ".join(filter(None, ("btn btn-ghost", f"btn-{size}", "btn-square", extra_class)))
    return format_html(
        '<button type="button" class="{}" data-copy="{}" aria-label="{}" title="{}">{}</button>',
        classes,
        value,
        label,
        title or label,
        icon(Icon.COPY, small=True),
    )


RESET_ON_FILTER = frozenset({"page", "key_q", "before", "after", "request"})


@register.simple_tag(takes_context=True)
def hidden_params(
    context: template.Context, *dropped: str, keep: Iterable[tuple[str, str]] | None = None
) -> SafeString:
    """Hidden inputs that carry a query to a GET form, minus `dropped` and paging.

    `keep` names the parameters to carry; without it, the page's own query is carried.
    """
    skip = _names(dropped) | RESET_ON_FILTER
    pairs = (
        keep
        if keep is not None
        else ((name, value) for name, values in context.request.GET.lists() for value in values)
    )
    return format_html_join(
        "",
        '<input type="hidden" name="{}" value="{}">',
        ((name, value) for name, value in pairs if name not in skip),
    )


@register.simple_tag(takes_context=True)
def query_with(context: template.Context, name: str, value: object, dropped: str = "") -> str:
    """`{% querystring %}` with a parameter name that comes from a variable."""
    return querystring(context, **{name: value}, **dict.fromkeys(_names((dropped,))))


@register.simple_tag
def url_with(viewname: str, *args: object, **query: object) -> str:
    return f"{reverse(viewname, args=args)}?{urlencode(query)}"


def _names(groups: Iterable[str]) -> set[str]:
    return {name for group in groups for name in group.split()}


@register.filter
def widget(field: BoundField) -> SafeString:
    classes = field.field.widget.attrs.get("class", "")
    if not (field.errors and classes):
        return field.as_widget()
    component = classes.split()[0]
    return field.as_widget(attrs={"class": f"{classes} {component}-error"})


register.filter("grouped", grouped)


@register.filter
def break_at_sign(value: object) -> SafeString:
    """An email address wraps before its @ and nowhere inside a word."""
    return mark_safe("<wbr>@".join(conditional_escape(part) for part in str(value).split("@")))  # noqa: S308


@register.filter
def dash(value: object) -> object:
    return MISSING if value is None or value == "" else value


WHEN_FORMATS = {"date": "j M Y", "datetime": "j M Y H:i", "recent": "j M H:i"}


@register.filter(expects_localtime=True)
def when(value: object, style: str = "datetime") -> SafeString | str:
    if not isinstance(value, date):
        return ""
    if style == "relative":
        if not isinstance(value, datetime):
            raise template.TemplateSyntaxError("when:'relative' needs a datetime, not a date")
        short = ago(value, timezone.now())
    elif style in WHEN_FORMATS:
        short = format_date(value, WHEN_FORMATS[style])
    else:
        raise template.TemplateSyntaxError(
            f"unknown when style {style!r}; use date, datetime, recent, or relative"
        )
    return _time(value, short)


AGO_UNITS = (
    (
        365 * 86400,
        ngettext_lazy("%(count)s\xa0year ago", "%(count)s\xa0years ago", "count"),
        ngettext_lazy("in %(count)s\xa0year", "in %(count)s\xa0years", "count"),
    ),
    (
        30 * 86400,
        ngettext_lazy("%(count)s\xa0month ago", "%(count)s\xa0months ago", "count"),
        ngettext_lazy("in %(count)s\xa0month", "in %(count)s\xa0months", "count"),
    ),
    (
        7 * 86400,
        ngettext_lazy("%(count)s\xa0week ago", "%(count)s\xa0weeks ago", "count"),
        ngettext_lazy("in %(count)s\xa0week", "in %(count)s\xa0weeks", "count"),
    ),
    (
        86400,
        ngettext_lazy("%(count)s\xa0day ago", "%(count)s\xa0days ago", "count"),
        ngettext_lazy("in %(count)s\xa0day", "in %(count)s\xa0days", "count"),
    ),
    (
        3600,
        ngettext_lazy("%(count)s\xa0hour ago", "%(count)s\xa0hours ago", "count"),
        ngettext_lazy("in %(count)s\xa0hour", "in %(count)s\xa0hours", "count"),
    ),
    (
        60,
        ngettext_lazy("%(count)s\xa0minute ago", "%(count)s\xa0minutes ago", "count"),
        ngettext_lazy("in %(count)s\xa0minute", "in %(count)s\xa0minutes", "count"),
    ),
)


def ago(value: datetime, now: datetime) -> str:
    seconds = (now - value).total_seconds()
    for index, (size, _past, _future) in enumerate(AGO_UNITS):
        if abs(seconds) >= size:
            unit, count = index, int(abs(seconds) / size + 0.5)
            if index and count >= round(AGO_UNITS[index - 1][0] / size):
                unit, count = index - 1, 1
            past, future = AGO_UNITS[unit][1:]
            return str((past if seconds > 0 else future) % {"count": count})
    return gettext("just now")


@register.simple_tag
def confirm(
    title: str, detail: str = "", action: str = "", dismiss: str = "", use: str = ""
) -> SafeString:
    attrs = format_html(
        'data-confirm="{}" data-confirm-detail="{}" data-confirm-action="{}"',
        title,
        detail,
        action or gettext("Delete"),
    )
    if dismiss:
        attrs = format_html('{} data-confirm-dismiss="{}"', attrs, dismiss)
    if use:
        attrs = format_html('{} data-confirm-use="{}"', attrs, use)
    return attrs




def _time(value: date, short: str) -> SafeString:
    if isinstance(value, datetime):
        utc = value.astimezone(UTC)
        machine, full = utc.isoformat(), f"{dateformat.format(utc, 'Y-m-d H:i:s')} UTC"
    else:
        machine = full = value.isoformat()
    return format_html('<time datetime="{}" title="{}">{}</time>', machine, full, short)
