"""What a key may use, what it has used, and when each count starts over.

The gateway counts a key's use in fixed UTC buckets: a calendar minute, hour and day
(`gateway.rate_limiting`). A bucket empties when its calendar unit ends, so "resets in" is
exact and a key that is full for the minute is free again at the next whole minute.

`panels` joins a key's resolved limits with those counters into a grid of `Cell`s, metric by
window. `configured` shows the limits alone, for a team, an organization or a person.
"""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

from django.conf import settings
from django.utils.functional import Promise
from django.utils.translation import gettext, gettext_lazy

from gateway.rate_limiting import WINDOWS as GATEWAY_WINDOWS
from gateway.rate_limiting import get_per_token_usage
from management.formatting import and_join, as_percent, compact, grouped
from management.models import LimitSet, Team, Token
from management.ownership import owner

NEARLY_FULL_PERCENT = 80
FULL_PERCENT = 100
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


class Tone(StrEnum):
    CALM = "calm"
    NEARLY_FULL = "nearly-full"
    FULL = "full"

    @property
    def meter(self) -> str:
        return {
            Tone.CALM: "progress-primary",
            Tone.NEARLY_FULL: "progress-warning",
            Tone.FULL: "progress-error",
        }[self]

    @property
    def figure(self) -> str:
        return {
            Tone.CALM: "",
            Tone.NEARLY_FULL: "st-warn font-semibold",
            Tone.FULL: "st-err font-semibold",
        }[self]


# Labels per gateway window: in use, in a compact label, and as a configured limit.
_LABELS = {
    "min": (gettext_lazy("This minute"), gettext_lazy("minute"), gettext_lazy("Per minute")),
    "hour": (gettext_lazy("This hour"), gettext_lazy("hour"), gettext_lazy("Per hour")),
    "day": (gettext_lazy("Today (UTC)"), gettext_lazy("today"), gettext_lazy("Per day")),
}


@dataclass(frozen=True)
class Window:
    """One of the gateway's windows, named as `gateway.rate_limiting.WINDOWS` names it."""

    name: str
    seconds: int

    @property
    def used_label(self) -> str:
        return str(_LABELS[self.name][0])

    @property
    def short_label(self) -> str:
        return str(_LABELS[self.name][1])

    @property
    def limit_label(self) -> str:
        return str(_LABELS[self.name][2])


WINDOWS = tuple(Window(name, seconds) for name, seconds in GATEWAY_WINDOWS)


class Metric(NamedTuple):
    counter: str  # the key in the gateway's buckets
    slot: int  # position in a window of `LimitSet.windows()`
    field: str  # the per-minute field of a limit model
    label: Promise
    requests: bool


METRICS = (
    Metric("req", 2, "requests_per_minute", gettext_lazy("Requests"), True),
    Metric("in", 3, "input_tokens_per_minute", gettext_lazy("Input tokens"), False),
    Metric("out", 4, "output_tokens_per_minute", gettext_lazy("Output tokens"), False),
)


def _count(value: float, requests: bool) -> str:
    return grouped(round(value)) if requests else compact(round(value))


def seconds_left(window_seconds: int, now: datetime) -> int:
    """Whole seconds until the current bucket ends; buckets align with the UTC epoch."""
    return math.ceil(window_seconds - now.timestamp() % window_seconds)


def _resets_in(seconds: int) -> str:
    if seconds < SECONDS_PER_MINUTE:
        return gettext("resets in %(seconds)s s") % {"seconds": seconds}
    minutes = math.ceil(seconds / SECONDS_PER_MINUTE)
    if seconds <= SECONDS_PER_HOUR:
        return gettext("resets in %(minutes)s min") % {"minutes": minutes}
    hours, minutes = divmod(minutes, SECONDS_PER_MINUTE)
    if minutes == 0:
        return gettext("resets in %(hours)s h") % {"hours": hours}
    return gettext("resets in %(hours)s h %(minutes)s min") % {"hours": hours, "minutes": minutes}


@dataclass(frozen=True)
class Cell:
    metric: str
    counter: str
    requests: bool
    window: Window
    used: float
    limit: int | None
    seconds_left: int
    resets_at: datetime

    @property
    def percent(self) -> int | None:
        if self.limit is None:
            return None
        if self.limit == 0 or self.used >= self.limit:
            return FULL_PERCENT
        return int(self.used * 100 // self.limit)

    @property
    def percent_text(self) -> str:
        return "" if self.percent is None else as_percent(self.percent)

    @property
    def tone(self) -> Tone:
        percent = self.percent
        if percent is None or percent < NEARLY_FULL_PERCENT:
            return Tone.CALM
        return Tone.FULL if percent >= FULL_PERCENT else Tone.NEARLY_FULL

    @property
    def used_text(self) -> str:
        return _count(self.used, self.requests)

    @property
    def limit_text(self) -> str:
        return gettext("No limit") if self.limit is None else _count(self.limit, self.requests)

    @property
    def exact(self) -> str:
        """The unabbreviated figures, for a tooltip."""
        used = grouped(round(self.used))
        if self.limit is None:
            return used
        return gettext("%(used)s of %(limit)s") % {"used": used, "limit": grouped(self.limit)}

    @property
    def resets_text(self) -> str:
        return _resets_in(self.seconds_left)

    @property
    def label(self) -> str:
        return gettext("%(metric)s, %(window)s") % {
            "metric": self.metric,
            "window": self.window.short_label,
        }


@dataclass(frozen=True)
class Row:
    label: str
    cells: tuple[Cell, ...]

    @property
    def limited(self) -> bool:
        return any(cell.limit is not None for cell in self.cells)


@dataclass(frozen=True)
class LimitsPanel:
    """A key's grid: one row per metric, one cell per window; and its concurrency limit.

    `concurrent` is None when the gateway does not enforce a concurrency limit. `source` says who
    set the limits.
    """

    rows: tuple[Row, ...]
    concurrent: int | None
    source: str

    @property
    def columns(self) -> tuple[Cell, ...]:
        """One cell per window, for its heading: every cell of a window resets together."""
        return self.rows[0].cells

    @property
    def limited(self) -> bool:
        """Whether any request or token limit applies; the concurrency limit is said apart."""
        return any(row.limited for row in self.rows)

    @property
    def closest(self) -> Cell | None:
        """The cell with a limit that is nearest to it; the first of equals, finest window first."""
        limited = [cell for row in self.rows for cell in row.cells if cell.limit is not None]
        return max(limited, key=lambda cell: cell.percent or 0, default=None)


def _source(token: Token) -> str:
    """Whose limits the grid shows.

    The gateway takes each per-minute limit from the first of the key, its team or person, and its
    organization that sets it, so the grid can draw on more than one of them.
    """
    found = owner(token)
    assert found is not None
    levels = (
        (token, gettext("this key")),
        (found, found.name if isinstance(found, Team) else found.user.email),
        (found.org, found.org.name),
    )
    names: list[str] = []
    for metric in METRICS:
        name = next((n for level, n in levels if getattr(level, metric.field) is not None), None)
        if name and name not in names:
            names.append(name)
    if names == [levels[0][1]]:
        return gettext("Set for this key.")
    return gettext("Set for %(names)s, counted per key.") % {"names": and_join(names)}


def effective_limits(token: Token) -> LimitSet:
    """`token.get_limits()` from the relations the page already loaded.

    `get_limits()` fetches the key again; on a list that is one query per row.
    """
    found = owner(token)
    assert found is not None
    return LimitSet.from_objects(token, [found, found.org])


def _concurrent(limits: LimitSet) -> int | None:
    if not settings.AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED:
        return None
    return limits.concurrent_requests_limit


def _cell(metric: Metric, window: Window, used: float, limit: int | None, now: datetime) -> Cell:
    ends = (now.timestamp() // window.seconds + 1) * window.seconds
    return Cell(
        str(metric.label),
        metric.counter,
        metric.requests,
        window,
        used,
        limit,
        seconds_left(window.seconds, now),
        datetime.fromtimestamp(ends, UTC),
    )


def _panel(token: Token, usage: Mapping[str, Mapping[str, float]], now: datetime) -> LimitsPanel:
    limits = effective_limits(token)
    rows = tuple(
        Row(
            str(metric.label),
            tuple(
                _cell(metric, window, usage[window.name][metric.counter], limit[metric.slot], now)
                for window, limit in zip(WINDOWS, limits.windows(), strict=True)
            ),
        )
        for metric in METRICS
    )
    return LimitsPanel(rows, _concurrent(limits), _source(token))


def panels(tokens: Iterable[Token], now: datetime) -> dict[int, LimitsPanel]:
    """The panel of each key, from one read of the gateway's counters.

    The caller passes only keys the viewer may open: a panel shows what a key has used.
    """
    found = list(tokens)
    if not found:
        return {}
    usage = get_per_token_usage([token.id for token in found])
    return {token.id: _panel(token, usage[token.id], now) for token in found}


@dataclass(frozen=True)
class ConfiguredRow:
    label: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class ConfiguredLimits:
    """Limits as set, with no use: a row per window, a column per metric."""

    headings: tuple[str, ...]
    rows: tuple[ConfiguredRow, ...]
    concurrent: int | None
    limited: bool


def configured(limits: LimitSet) -> ConfiguredLimits:
    rows = tuple(
        ConfiguredRow(
            str(window.limit_label),
            tuple(
                gettext("No limit")
                if per_window[metric.slot] is None
                else _count(per_window[metric.slot], metric.requests)
                for metric in METRICS
            ),
        )
        for window, per_window in zip(WINDOWS, limits.windows(), strict=True)
    )
    return ConfiguredLimits(
        tuple(str(metric.label) for metric in METRICS),
        rows,
        _concurrent(limits),
        limits.has_any_rate_limit(),
    )


MULTIPLIER_FIELDS = {"hour": "hourly_limit_multiplier", "day": "daily_limit_multiplier"}


def _whole(value: object) -> int | None:
    try:
        number = int(str(value))
    except ValueError:
        return None
    return number if number >= 0 else None


@dataclass(frozen=True)
class MultiplierPreview:
    """What a limits form's multiplier comes to with the per-minute numbers as typed.

    The browser recomputes `template` as the person types; `sentence` is the same line for the
    values the page was rendered with, so the page reads correctly without a script.
    """

    multiplier: str
    default: int
    sentence: str
    template: str
    none: str
    labels: str


def multiplier_previews(values: Mapping[str, object]) -> dict[str, MultiplierPreview]:
    """The previews of the hourly and daily multiplier fields, from a form's field values."""
    minutes = {metric.field: _whole(values[metric.field]) for metric in METRICS}
    defaults = {
        "hour": settings.AQUEDUCT_HOURLY_LIMIT_MULTIPLIER,
        "day": settings.AQUEDUCT_DAILY_LIMIT_MULTIPLIER,
    }
    labels = json.dumps({metric.field: str(metric.label) for metric in METRICS})
    previews = {}
    for window in WINDOWS[1:]:
        field = MULTIPLIER_FIELDS[window.name]
        multiplier = _whole(values[field]) or defaults[window.name]
        text = gettext("%(window)s: %(figures)s (%(multiplier)s × the per-minute limits).")  # noqa: RUF001
        none = gettext("%(window)s: no per-minute limit is set, so there is no limit.") % {
            "window": window.limit_label
        }
        figures = ", ".join(
            f"{metric.label} {grouped(limit * multiplier)}"
            for metric in METRICS
            if (limit := minutes[metric.field]) is not None
        )
        shown = {"window": window.limit_label, "multiplier": grouped(multiplier)}
        previews[field] = MultiplierPreview(
            multiplier=field,
            default=defaults[window.name],
            sentence=text % {**shown, "figures": figures} if figures else none,
            template=text % {**shown, "figures": "{figures}", "multiplier": "{multiplier}"},
            none=none,
            labels=labels,
        )
    return previews
