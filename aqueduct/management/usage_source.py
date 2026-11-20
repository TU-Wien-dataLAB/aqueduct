from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from django.db.models import Q, QuerySet, Sum
from django.db.models.functions import Trunc

from management.activity import TOKENS, TOTALS, Activity, ModelBreakdown, Series, model_usage
from management.models import Request
from management.outcomes import OUTCOME_COUNTS, Outcome
from management.ownership import key_org
from management.periods import Period
from management.request_filters import Filters


@dataclass(frozen=True)
class UsageSource:
    requests: QuerySet[Request]
    period: Period

    @classmethod
    def of(cls, filters: Filters, narrower: Q | None = None) -> "UsageSource":
        requests = filters.requests if narrower is None else filters.requests.filter(narrower)
        return cls(requests, filters.period)

    def series(self) -> Series:
        rows = (
            self.period.of(self.requests)
            .order_by()
            .values(bucket=Trunc("timestamp", self.period.step.kind))
            .annotate(**TOTALS, **OUTCOME_COUNTS)
        )
        found: defaultdict[Any, Counter[str]] = defaultdict(Counter)
        for row in rows:
            found[row.pop("bucket")].update(row)
        return Series(
            [(bucket, _activity(found.get(bucket, Counter()))) for bucket in self.period.buckets],
            {o: sum(counts[o.name] for counts in found.values()) for o in Outcome},
        )

    def models(self) -> ModelBreakdown:
        return model_usage(self.period.of(self.requests))

    def org_tokens(self) -> dict[int | None, int]:
        rows = (
            self.period.of(self.requests)
            .order_by()
            .values(org_id=key_org("token__"))
            .annotate(tokens=Sum(TOKENS))
        )
        return {row["org_id"]: row["tokens"] for row in rows}

    def key_totals(self) -> dict[int, Activity]:
        rows = self.period.of(self.requests).values("token_id").annotate(**TOTALS).order_by()
        return {row.pop("token_id"): Activity(**row) for row in rows}


def _activity(counts: Counter[str]) -> Activity:
    return Activity(**{field: counts[field] for field in TOTALS})
