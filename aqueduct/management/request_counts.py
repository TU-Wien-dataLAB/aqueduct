import hashlib
from dataclasses import dataclass

from django.core.cache import cache
from django.db.models import Q

from management.outcomes import FAILED, Outcome
from management.request_filters import Filters
from management.usage_source import UsageSource

FAILED_TAB = "failed"

STATUS_FILTERS = {FAILED_TAB: FAILED} | {o.codes: o.matches for o in Outcome}


@dataclass(frozen=True)
class Tab:
    value: str | None
    label: str
    count: int
    codes: str | None = None


COUNTS_SECONDS = 60


def tab_counts(filters: Filters, search: Q | None) -> dict[str | None, int]:

    def count() -> dict[str | None, int]:
        found = UsageSource.of(filters, narrower=search).series()
        return {None: found.total.requests, FAILED_TAB: found.total.failed} | {
            o.codes: n for o, n in found.outcomes.items()
        }

    period = filters.period
    scope = (filters.visibility, period.start.isoformat(), period.end.isoformat(), search)
    key = "request-log-tab-counts:" + hashlib.sha256(repr(scope).encode()).hexdigest()
    found: dict[str | None, int] | None = cache.get(key)
    if found is None:
        found = count()
        cache.set(key, found, COUNTS_SECONDS)
    return found
