from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from django.db.models import Aggregate, BigIntegerField, Count, F, IntegerField, QuerySet, Sum
from django.db.models.functions import Cast

from gateway.config import model_names
from management.formatting import compact, percent
from management.models import Request
from management.outcomes import FAILED, RATE_LIMITED, Outcome, above_alert
from management.periods import Period

# Postgres adds two integer columns as 32-bit, so one huge row would overflow the sum.
TOKENS = Cast("input_tokens", BigIntegerField()) + F("output_tokens")
TOTALS = {
    "requests": Count("id"),
    "failed": Count("id", filter=FAILED),
    "rate_limited": Count("id", filter=RATE_LIMITED),
    "input": Sum("input_tokens", default=0),
    "cached": Sum("cached_input_tokens", default=0),
    "output": Sum("output_tokens", default=0),
}


@dataclass(frozen=True)
class Activity:
    requests: int = 0
    failed: int = 0
    rate_limited: int = 0
    input: int = 0
    cached: int = 0
    output: int = 0
    last_used: datetime | None = None

    @property
    def tokens(self) -> int:
        return self.input + self.output

    @property
    def tokens_short(self) -> str:
        return compact(self.tokens)

    @property
    def cache_hits(self) -> str:
        return percent(self.cached, self.input)

    @property
    def failed_share(self) -> str:
        return percent(self.failed, self.requests)

    @property
    def failing(self) -> bool:
        return above_alert(self.failed, self.requests)

    @property
    def segments(self) -> tuple[tuple[str, int], ...]:
        return (("c", self.cached), ("n", self.input - self.cached), ("o", self.output))


def summed(activities: Sequence[Activity]) -> Activity:
    return Activity(**{field: sum(getattr(a, field) for a in activities) for field in TOTALS})


def activity_by(field: str, ids: Iterable[int], period: Period) -> dict[int, Activity]:
    rows = (
        period.of(Request.objects.filter(**{f"{field}__in": list(ids)}))
        .values(field)
        .annotate(**TOTALS)
        .order_by()
    )
    return {row.pop(field): Activity(**row) for row in rows}


class Median(Aggregate):
    function = "PERCENTILE_DISC"
    template = "%(function)s(0.5) WITHIN GROUP (ORDER BY %(expressions)s)"
    output_field = IntegerField()

    def as_sqlite(self, compiler: Any, connection: Any, **extra_context: Any) -> tuple[str, Any]:
        return super().as_sql(
            compiler,
            connection,
            function="GROUP_CONCAT",
            template="%(function)s(%(expressions)s)",
            **extra_context,
        )

    def get_db_converters(self, connection: Any) -> list[Callable[..., Any]]:
        if connection.vendor == "sqlite":
            return [_lower_median]
        return super().get_db_converters(connection)


def _lower_median(value: str | None, *_: object) -> int | None:
    if not value:
        return None
    ordered = sorted(int(v) for v in value.split(","))
    return ordered[(len(ordered) - 1) // 2]


MEDIAN = Median("response_time_ms")


@dataclass(frozen=True)
class Series:
    points: list[tuple[datetime, Activity]]
    outcomes: dict[Outcome, int]

    @property
    def total(self) -> Activity:
        return summed([a for _, a in self.points])


@dataclass(frozen=True)
class ModelUsage:
    activity: Activity
    median_ms: int | None


@dataclass(frozen=True)
class ModelBreakdown:
    served: dict[str, ModelUsage]
    # Names the router config does not list, typos among them, and requests that name no model.
    other: ModelUsage | None


def model_usage(requests: QuerySet[Request]) -> ModelBreakdown:
    names = model_names()
    served = {}
    for row in (
        requests.filter(model__in=names)
        .values("model")
        .annotate(**TOTALS, median=MEDIAN)
        .order_by()
    ):
        model, median = row.pop("model"), row.pop("median")
        served[model] = ModelUsage(Activity(**row), median)
    rest = requests.exclude(model__in=names).aggregate(**TOTALS, median=MEDIAN)
    median = rest.pop("median")
    return ModelBreakdown(
        served, ModelUsage(Activity(**rest), median) if rest["requests"] else None
    )
