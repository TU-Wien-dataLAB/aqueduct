from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from django.db.models import QuerySet, Sum
from django.db.models.functions import Lower
from django.utils.translation import gettext

from management.activity import TOKENS, TOTALS, Activity, ModelBreakdown, ModelUsage, Series
from management.charts import Share, scaled
from management.formatting import MISSING, NO_ORG, compact, duration, percent
from management.limits import LimitsPanel, panels
from management.models import Org, Request, Token
from management.outcomes import Outcome
from management.ownership import RELATED

if TYPE_CHECKING:
    from management.request_filters import Key

TOP_KEYS = 10
TOP_USER_IDS = 25


@dataclass(frozen=True)
class StatusRow:
    outcome: Outcome
    count: int
    share: str
    href: str


def statuses(timeline: Series, link: Callable[..., str]) -> tuple[StatusRow, ...]:
    total = timeline.total.requests
    return tuple(
        StatusRow(outcome, count, percent(count, total), link(status=outcome.codes))
        for outcome, count in timeline.outcomes.items()
    )


class Figures:
    activity: Activity

    def _metered(self, value: str) -> str:
        return value if self.activity.tokens else MISSING

    @property
    def input(self) -> str:
        return self._metered(compact(self.activity.input))

    @property
    def cached(self) -> str:
        return self._metered(self.activity.cache_hits)

    @property
    def output(self) -> str:
        return self._metered(compact(self.activity.output))

    @property
    def failed(self) -> str:
        return self.activity.failed_share


@dataclass(frozen=True)
class ModelRow(Figures):
    name: str
    kind: str
    activity: Activity
    median: str
    share: str
    width: int
    log_url: str | None


def model_rows(
    usage: ModelBreakdown, kinds: Mapping[str, str], link: Callable[..., str]
) -> tuple[ModelRow, ...]:
    found = {name: u.activity for name, u in usage.served.items()}
    ranked = sorted(found.items(), key=lambda item: (-item[1].tokens, -item[1].requests, item[0]))
    others = usage.other
    every = [*found.values(), *([others.activity] if others else [])]
    total = sum(a.tokens for a in every)
    busiest = max((a.tokens for a in every), default=0)

    def row(name: str, kind: str, shown: ModelUsage, log_url: str | None) -> ModelRow:
        activity = shown.activity
        return ModelRow(
            name=name,
            kind=kind,
            activity=activity,
            median=MISSING if shown.median_ms is None else duration(shown.median_ms),
            share=percent(activity.tokens, total) if activity.tokens else MISSING,
            width=round(100 * activity.tokens / busiest) if busiest else 0,
            log_url=log_url,
        )

    rows = [
        row(name, kinds.get(name, ""), usage.served[name], link(model=name)) for name, _ in ranked
    ]
    if others:
        rows.append(
            row(gettext("Unknown model names"), gettext("Not on this gateway"), others, None)
        )
    return tuple(rows)


@dataclass(frozen=True)
class KeyShare(Figures):
    name: str
    detail: str
    href: str | None
    activity: Activity
    share: str
    width: int
    limits: LimitsPanel | None = None


def _figures(
    activities: Sequence[Activity], total: int, busiest: int | None = None
) -> list[tuple[Activity, str, int]]:
    return [
        (activity, percent(activity.tokens, total) if activity.tokens else MISSING, width)
        for activity, width in zip(
            activities, scaled((a.tokens for a in activities), busiest), strict=True
        )
    ]


def _activity(row: Mapping[str, Any]) -> Activity:
    return Activity(**{name: row[name] for name in TOTALS})


class KeyOrder(StrEnum):
    TOKENS = "tokens"
    REQUESTS = "requests"
    NAME = "name"

    @property
    def ascending(self) -> bool:
        return self is KeyOrder.NAME

    @classmethod
    def parse(cls, raw: str) -> "KeyOrder | None":
        try:
            return cls(raw)
        except ValueError:
            return None


@dataclass(frozen=True)
class KeyTotals:
    keys: int
    busiest: int
    tokens: int

    @classmethod
    def of(cls, totals: Mapping[int, Activity]) -> "KeyTotals":
        tokens = [a.tokens for a in totals.values()]
        return cls(len(tokens), max(tokens, default=0), sum(tokens))


def ranked(
    totals: Mapping[int, Activity], order: KeyOrder, tokens: QuerySet[Token]
) -> list[tuple[int, Activity]]:
    if order is KeyOrder.NAME:
        named = [
            token_id
            for token_id in tokens.order_by(Lower("name"), "id").values_list("id", flat=True)
            if token_id in totals
        ]
        gone = sorted(set(totals) - set(named))
        return [(token_id, totals[token_id]) for token_id in [*named, *gone]]
    if order is KeyOrder.REQUESTS:
        return sorted(totals.items(), key=lambda r: (-r[1].requests, -r[1].tokens, r[0]))
    return sorted(totals.items(), key=lambda r: (-r[1].tokens, -r[1].requests, r[0]))


def key_shares(
    rows: Sequence[tuple[int, Activity]],
    total: int,
    describe: Callable[[Token], "Key"],
    busiest: int | None = None,
    limits_at: datetime | None = None,
) -> tuple[KeyShare, ...]:
    """The rows' keys with their shares.

    With `limits_at`, the limits panel of each openable key is read from the gateway's counters.
    """
    tokens = Token.objects.select_related(*RELATED).in_bulk([token_id for token_id, _ in rows])
    rows = [row for row in rows if row[0] in tokens]
    keys = [describe(tokens[token_id]) for token_id, _ in rows]
    figures = _figures([activity for _, activity in rows], total, busiest)
    found = (
        panels((key.token for key in keys if key.href is not None), limits_at) if limits_at else {}
    )
    return tuple(
        KeyShare(key.name, key.note, key.href, *figure, limits=found.get(key.token.id))
        for key, figure in zip(keys, figures, strict=True)
    )


def user_id_shares(requests: QuerySet[Request], total: int) -> tuple[KeyShare, ...]:
    rows = list(
        requests.values("user_id")
        .annotate(**TOTALS, tokens=Sum(TOKENS, default=0))
        .order_by("-tokens", "-requests", "user_id")[:TOP_USER_IDS]
    )
    figures = _figures([_activity(row) for row in rows], total)
    return tuple(
        KeyShare(row["user_id"] or gettext("Not sent"), "", None, *figure)
        for row, figure in zip(rows, figures, strict=True)
    )


def org_shares(tokens: Mapping[int | None, int]) -> tuple[Share, ...]:
    orgs = Org.objects.in_bulk([org_id for org_id in tokens if org_id is not None])
    no_org = str(NO_ORG)
    rows = sorted(
        (
            (orgs[org_id].name if org_id in orgs else no_org, count)
            for org_id, count in tokens.items()
        ),
        key=lambda row: (-row[1], row[0] == no_org, row[0]),
    )
    return tuple(
        Share(name, compact(count), width)
        for (name, count), width in zip(rows, scaled(count for _, count in rows), strict=True)
    )
