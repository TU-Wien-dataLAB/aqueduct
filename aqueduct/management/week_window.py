from dataclasses import dataclass
from datetime import date, datetime

from django.db.models import QuerySet
from django.db.models.functions import TruncDate
from django.utils import timezone

from management.activity import TOTALS, Activity, summed
from management.charts import OVERVIEW_STATS, Stat, stats
from management.formatting import day_label
from management.models import Request
from management.periods import week


@dataclass(frozen=True)
class Day:
    day: date
    activity: Activity


@dataclass(frozen=True)
class Window:
    totals: Activity
    days: tuple[Day, ...]

    @property
    def points(self) -> list[tuple[str, Activity]]:
        return [(day_label(d.day), d.activity) for d in self.days]

    @property
    def stats(self) -> tuple[Stat, ...]:
        return stats(self.points, OVERVIEW_STATS)


def window(requests: QuerySet[Request], now: datetime) -> Window:
    period = week(now)
    per_day = {
        row.pop("day"): Activity(**row)
        for row in period.of(requests)
        .annotate(day=TruncDate("timestamp"))
        .values("day")
        .annotate(**TOTALS)
        .order_by()
    }
    days = tuple(
        Day(day, per_day.get(day, Activity()))
        for day in (timezone.localdate(bucket) for bucket in period.buckets)
    )
    return Window(summed(list(per_day.values())), days)
