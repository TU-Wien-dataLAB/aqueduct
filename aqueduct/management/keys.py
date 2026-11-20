import shlex
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from django.db.models import Count, F, OuterRef, Subquery, Sum
from django.forms import Form
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.translation import gettext, ngettext

from management.access import KeyAct, TeamAct, Viewer
from management.activity import Activity, activity_by
from management.charts import Chart, Share, chart, scaled
from management.formatting import compact, percent
from management.limits import LimitsPanel, panels
from management.model_catalog import Category, ModelCard, Scope, load_catalog, scopes_for
from management.models import Request, ServiceAccount, Token
from management.ownership import RELATED
from management.periods import Period, Span, week
from management.week_window import Window, window

EXPIRY_WARNING = timedelta(days=7)
CHAT_ENDPOINT = "gateway:chat_completions"
FIRST_KEY_NAME = "my-first-key"


class KeyKind(StrEnum):
    PERSONAL = "personal"
    SERVICE = "service"

    @property
    def label(self) -> str:
        return gettext("Personal") if self is KeyKind.PERSONAL else gettext("Service accounts")

    @classmethod
    def parse(cls, raw: str) -> "KeyKind | None":
        try:
            return cls(raw)
        except ValueError:
            return None


@dataclass(frozen=True)
class Expiry:
    label: str
    sentence: str
    tone: str = "muted"
    at: datetime | None = None

    @classmethod
    def of(cls, expires_at: datetime | None, now: datetime) -> "Expiry":
        if expires_at is None:
            return cls(gettext("No expiry"), gettext("Never expires"))
        if expires_at <= now:
            return cls(gettext("Expired"), gettext("Expired"), "st-err font-semibold", expires_at)
        if expires_at - now <= EXPIRY_WARNING:
            days = (timezone.localdate(expires_at) - timezone.localdate(now)).days
            if days == 0:
                label, sentence = gettext("Today"), gettext("Expires today")
            else:
                label = ngettext("In %(days)s day", "In %(days)s days", days) % {"days": days}
                sentence = ngettext("Expires in %(days)s day", "Expires in %(days)s days", days) % {
                    "days": days
                }
            return cls(label, sentence, "st-warn font-semibold", expires_at)
        day = date_format(timezone.localtime(expires_at), "j M Y")
        return cls(day, gettext("Expires %(date)s") % {"date": day}, at=expires_at)


@dataclass(frozen=True)
class NewKey:
    """What a key's page shows once, right after the key was created or rotated."""

    token: Token
    secret: str
    snippet: str

    @property
    def for_service_account(self) -> bool:
        return self.token.service_account_id is not None

    @classmethod
    def of(cls, token: Token, secret: str, gateway_url: str) -> "NewKey":
        if token.service_account_id is not None:
            snippet = f"OPENAI_BASE_URL={gateway_url}\nOPENAI_API_KEY={secret}"
            return cls(token, secret, snippet)
        exports = (
            f"export OPENAI_BASE_URL={shlex.quote(gateway_url)}\n"
            f"export OPENAI_API_KEY={shlex.quote(secret)}\n"
        )
        return cls(token, secret, exports + _first_call(token, gateway_url))


def _first_call(token: Token, gateway_url: str) -> str:
    catalog = load_catalog((Scope(None, token.model_exclusion_list()),))
    chat = next((card for card in catalog.cards if card.mode.endpoint == CHAT_ENDPOINT), None)
    if chat is not None:
        return chat.curl(gateway_url, chat.aliases[0] if chat.aliases else chat.name) or ""
    return 'curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"'


def key_activity(token_ids: Iterable[int], now: datetime) -> dict[int, Activity]:
    ids = list(token_ids)
    activity = activity_by("token_id", ids, week(now))
    newest = Request.objects.filter(token=OuterRef("pk")).order_by("-timestamp")
    used = Token.objects.filter(id__in=ids).annotate(used=Subquery(newest.values("timestamp")[:1]))
    return {
        token_id: replace(activity.get(token_id, Activity()), last_used=at)
        for token_id, at in used.values_list("id", "used")
    }


@dataclass(frozen=True)
class KeyRow:
    token: Token
    kind: KeyKind
    team: str | None
    href: str
    activity: Activity
    expiry: Expiry
    manageable: bool
    limits: LimitsPanel | None


@dataclass(frozen=True)
class KeyList:
    rows: tuple[KeyRow, ...]
    personal_count: int
    has_service_accounts: bool


def key_list(viewer: Viewer, kind: KeyKind | None, query: str, now: datetime) -> KeyList:
    personal = list(viewer.keys(KeyAct.OWN).select_related(*RELATED).order_by("-created_at"))
    service = list(
        ServiceAccount.objects.filter(team__in=viewer.teams(TeamAct.MINE))
        .select_related("team__org", "token")
        .order_by("team__name", "name")
    )
    entries: list[tuple[Token, KeyKind, str | None, str]] = [
        (token, KeyKind.PERSONAL, None, reverse("token_detail", kwargs={"id": token.id}))
        for token in personal
    ] + [
        (
            sa.token,
            KeyKind.SERVICE,
            sa.team.name,
            reverse("token_detail", kwargs={"id": sa.token.id}),
        )
        for sa in service
        if hasattr(sa, "token")
    ]
    needle = query.casefold()
    entries = [
        (token, entry_kind, team, href)
        for token, entry_kind, team, href in entries
        if (kind is None or entry_kind is kind)
        and (needle in token.name.casefold() or needle in (team or "").casefold())
    ]
    activity = key_activity((token.id for token, *_ in entries), now)
    found = panels((token for token, *_ in entries if viewer.may_key(KeyAct.OPEN, token)), now)
    rows = tuple(
        KeyRow(
            token=token,
            kind=entry_kind,
            team=team,
            href=href,
            activity=activity.get(token.id, Activity()),
            expiry=Expiry.of(token.expires_at, now),
            manageable=viewer.may_key(KeyAct.MANAGE, token),
            limits=found.get(token.id),
        )
        for token, entry_kind, team, href in entries
    )
    return KeyList(rows, len(personal), bool(service))


@dataclass(frozen=True)
class FirstKey:
    form: Form
    example: str
    model: ModelCard | None
    alias: str


def _first_chat(cards: tuple[ModelCard, ...]) -> ModelCard | None:
    """The first chat model a personal key may call."""
    return next(
        (
            card
            for card in cards
            if card.mode.category is Category.CHAT
            and card.mode.endpoint
            and (not card.off_for or any(scope.team is None for scope in card.on_for))
        ),
        None,
    )


def first_key(viewer: Viewer, form: Form, base_url: str) -> FirstKey:
    card = _first_chat(load_catalog(scopes_for(viewer.profile, viewer.teams(TeamAct.MINE))).cards)
    alias = (card.aliases[0] if card.aliases else card.name) if card else ""
    call = replace(card, name=alias).curl("$OPENAI_BASE_URL") if card else None
    lines = [f"export OPENAI_BASE_URL={base_url}", "export OPENAI_API_KEY=sk-..."]
    example = "\n".join(lines) + (f"\n\n{call}" if call else "")
    return FirstKey(form, example, card, alias)


@dataclass(frozen=True)
class KeyReport(Window):
    models: tuple[Share, ...]
    clients: str
    latest: tuple[Request, ...]

    def chart(self, metric: str) -> Chart:
        return chart(self.points, metric)


def key_report(token: Token, now: datetime, latest: int = 10, top: int = 5) -> KeyReport:
    requests = Request.objects.filter(token=token)
    recent = week(now).of(requests)
    base = window(requests, now)

    by_model = list(
        recent.exclude(model="")
        .values("model")
        .annotate(tokens=Sum(F("input_tokens") + F("output_tokens")))
        .order_by("-tokens", "model")[:top]
    )
    models = tuple(
        Share(row["model"], compact(row["tokens"]), width)
        for row, width in zip(by_model, scaled(row["tokens"] for row in by_model), strict=True)
    )

    clients = [
        (row["user_agent"].split(" ")[0] or "Unknown", row["n"])
        for row in recent.values("user_agent").annotate(n=Count("id")).order_by("-n")[:3]
    ]
    client_text = ", ".join(f"{name} {percent(n, base.totals.requests)}" for name, n in clients)

    newest = requests.order_by("-timestamp").values_list("timestamp", flat=True).first()
    return KeyReport(
        totals=replace(base.totals, last_used=newest),
        days=base.days,
        models=models,
        clients=client_text,
        latest=tuple(Period.last(Span.MONTH, now).of(requests).order_by("-timestamp")[:latest]),
    )
