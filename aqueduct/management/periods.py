from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from django.db.models import QuerySet
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.functional import Promise
from django.utils.translation import gettext_lazy as _

from management.models import Request


@dataclass(frozen=True)
class Step:
    kind: str
    size: timedelta
    label_format: str
    zeroed: tuple[str, ...]
    short_format: str
    tokens_title: str | Promise
    requests_title: str | Promise

    def floor(self, moment: datetime) -> datetime:
        local = timezone.localtime(moment)
        zeroed: dict[str, Any] = dict.fromkeys(self.zeroed, 0)
        return local.replace(microsecond=0, **zeroed)

    def format(self, moment: datetime, short: bool = False) -> str:
        return date_format(moment, self.short_format if short else self.label_format)


MINUTE = Step(
    "minute",
    timedelta(minutes=1),
    "H:i",
    ("second",),
    "H:i",
    _("Tokens per minute"),
    _("Requests per minute"),
)
HOUR = Step(
    "hour",
    timedelta(hours=1),
    "H:00",
    ("second", "minute"),
    "H",
    _("Tokens per hour"),
    _("Requests per hour"),
)
DAY = Step(
    "day",
    timedelta(days=1),
    "j M",
    ("second", "minute", "hour"),
    "j",
    _("Tokens per day"),
    _("Requests per day"),
)


class Span(StrEnum):
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"

    @property
    def label(self) -> str:
        return str(SPANS[self].label)

    @property
    def no_requests(self) -> str:
        return str(SPANS[self].no_requests)

    @classmethod
    def parse(cls, raw: str) -> "Span | None":
        try:
            return cls(raw)
        except ValueError:
            return None


@dataclass(frozen=True)
class SpanSpec:
    label: Promise
    no_requests: Promise
    step: Step
    count: int


DEFAULT_SPAN = Span.DAY
SPANS: Mapping[Span, SpanSpec] = {
    Span.HOUR: SpanSpec(_("1 hour"), _("No requests in the last hour."), MINUTE, 60),
    Span.DAY: SpanSpec(_("24 hours"), _("No requests in the last 24 hours."), HOUR, 24),
    Span.WEEK: SpanSpec(_("7 days"), _("No requests in the last 7 days."), DAY, 7),
    Span.MONTH: SpanSpec(_("30 days"), _("No requests in the last 30 days."), DAY, 30),
}


@dataclass(frozen=True)
class Period:
    start: datetime
    end: datetime
    step: Step

    @classmethod
    def last(cls, span: Span, now: datetime) -> "Period":
        spec = SPANS[span]
        end = spec.step.floor(now) + spec.step.size
        return cls(end - spec.count * spec.step.size, end, spec.step)

    @property
    def buckets(self) -> list[datetime]:
        moment, found = self.step.floor(self.start), []
        while moment < self.end:
            found.append(moment)
            moment += self.step.size
        return found

    def of(self, requests: QuerySet[Request]) -> QuerySet[Request]:
        return requests.filter(timestamp__gte=self.start, timestamp__lt=self.end)


def week(now: datetime) -> Period:
    return Period.last(Span.WEEK, now)
