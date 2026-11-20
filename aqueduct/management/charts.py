import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypeVar

from django.utils import timezone
from django.utils.functional import Promise
from django.utils.translation import gettext, ngettext
from django.utils.translation import gettext_lazy as _

from management.activity import Activity, summed
from management.formatting import (
    as_input,
    as_output,
    as_percent,
    as_requests,
    as_tokens,
    compact,
    grouped,
)
from management.periods import Step

SPARK_BARS = 12
AXIS_LABELS = 8
TOKEN_SERIES = {"c": _("From cache"), "n": _("New input"), "o": _("Output")}
REQUEST_SERIES = {"i": _("OK"), "r": _("Rate-limited"), "f": _("Failed")}
K = TypeVar("K")


def scaled(values: Iterable[float], top: float | None = None) -> tuple[int, ...]:
    values = list(values)
    top = max(values, default=0) if top is None else top
    return tuple(round(100 * v / top) if top else 0 for v in values)


@dataclass(frozen=True)
class Tip:
    heading: str
    value: str
    parts: tuple[tuple[str, str | Promise, str], ...] = ()

    def __str__(self) -> str:
        detail = ", ".join(f"{name} {value}" for _, name, value in self.parts)
        return f"{self.heading}: {self.value}" + (f" ({detail})" if detail else "")


@dataclass(frozen=True)
class SparkBar:
    height: int
    tip: Tip


def spark(
    points: Sequence[tuple[str, float]], show: Callable[[float], str]
) -> tuple[SparkBar, ...]:
    return tuple(
        SparkBar(height, Tip(label, show(value)))
        for (label, value), height in zip(points, scaled(v for _, v in points), strict=True)
    )


@dataclass(frozen=True)
class Stat:
    label: str
    value: str
    spark: tuple[SparkBar, ...]
    note: str = ""


@dataclass(frozen=True)
class Column:
    label: str
    tip: Tip
    segments: tuple[tuple[str, int], ...]
    short: str = ""
    span: int = 1
    """How many columns the label spans: one label in `every` sits over all of them."""

    @property
    def height(self) -> int:
        return sum(height for _, height in self.segments)


@dataclass(frozen=True)
class Chart:
    columns: tuple[Column, ...]
    # The tallest column, drawn as a gridline so the bars carry a scale.
    top: str


@dataclass(frozen=True)
class Share:
    name: str
    value: str
    width: int


def chart(points: Sequence[tuple[str, Activity]], metric: str, every: int = 1) -> Chart:
    values: list[tuple[tuple[str, int], ...]]
    if metric == "requests":
        names, show = REQUEST_SERIES, as_requests
        values = [
            (("i", a.requests - a.failed - a.rate_limited), ("r", a.rate_limited), ("f", a.failed))
            for _, a in points
        ]
    else:
        names, show = TOKEN_SERIES, as_tokens
        values = [a.segments for _, a in points]
    top = max((sum(n for _, n in segs) for segs in values), default=0)

    def tip(label: str, segs: tuple[tuple[str, int], ...]) -> Tip:
        total = sum(n for _, n in segs)
        parts = tuple((css, names[css], grouped(n)) for css, n in segs) if total else ()
        return Tip(label, show(total), parts)

    return Chart(
        tuple(
            Column(
                label if index % every == 0 else "",
                tip(label, segs),
                tuple((css, round(100 * n / top)) for css, n in segs if n),
                span=min(every, len(points) - index),
            )
            for index, ((label, _), segs) in enumerate(zip(points, values, strict=True))
        ),
        show(top),
    )


def columns(points: Sequence[tuple[datetime, Activity]], step: Step, metric: str) -> Chart:
    local = [timezone.localtime(moment) for moment, _ in points]
    labelled = [(step.format(moment), a) for moment, (_, a) in zip(local, points, strict=True)]
    found = chart(labelled, metric, every=math.ceil(len(points) / AXIS_LABELS) or 1)
    return Chart(
        tuple(
            replace(column, short=step.format(moment, short=True) if column.label else "")
            for column, moment in zip(found.columns, local, strict=True)
        ),
        found.top,
    )


def slots(
    points: Sequence[tuple[datetime, Activity]], step: Step
) -> tuple[tuple[str, Activity], ...]:
    size = math.ceil(len(points) / SPARK_BARS) or 1

    def slot(group: Sequence[tuple[datetime, Activity]]) -> tuple[str, Activity]:
        first, last = (step.format(timezone.localtime(m)) for m in (group[0][0], group[-1][0]))
        between = gettext("%(first)s to %(last)s") % {"first": first, "last": last}
        label = first if first == last else between
        return label, summed([a for _, a in group])

    return tuple(slot(points[i : i + size]) for i in range(0, len(points), size))


@dataclass(frozen=True)
class Metric:
    label: Promise
    total: Callable[[Activity], str]
    each: Callable[[Activity], float]
    show: Callable[[float], str]
    note: Callable[[Activity], str] | None = None


def stats(points: Sequence[tuple[str, Activity]], metrics: Sequence[Metric]) -> tuple[Stat, ...]:
    total = summed([a for _, a in points])
    return tuple(
        Stat(
            str(metric.label),
            metric.total(total),
            spark([(label, metric.each(a)) for label, a in points], metric.show),
            "" if metric.note is None else metric.note(total),
        )
        for metric in metrics
    )


def _failed_percent(a: Activity) -> float:
    return 100 * a.failed / a.requests if a.requests else 0.0


REQUESTS = Metric(_("Requests"), lambda a: grouped(a.requests), lambda a: a.requests, as_requests)


def _from_cache(a: Activity) -> str:
    return gettext("%(share)s from cache") % {"share": a.cache_hits} if a.input else ""


def _rate_limited(a: Activity) -> str:
    return ngettext("%(count)s rate-limited", "%(count)s rate-limited", a.rate_limited) % {
        "count": grouped(a.rate_limited)
    }


OVERVIEW_STATS = (
    REQUESTS,
    Metric(_("Input tokens"), lambda a: compact(a.input), lambda a: a.input, as_input, _from_cache),
    Metric(_("Output tokens"), lambda a: compact(a.output), lambda a: a.output, as_output),
    Metric(_("Failed"), lambda a: a.failed_share, _failed_percent, as_percent, _rate_limited),
)


def token_widths(activity: Mapping[K, Activity]) -> dict[K, int]:
    return dict(zip(activity, scaled(a.tokens for a in activity.values()), strict=True))
