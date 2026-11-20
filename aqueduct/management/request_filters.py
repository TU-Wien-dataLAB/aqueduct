from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from django.db.models import Q, QuerySet
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.utils.http import urlencode
from django.utils.translation import gettext

from gateway.config import model_names
from management.access import EveryOrg, KeyAct, OrgWide, TeamAct, Viewer, requests_of
from management.formatting import NO_ORG
from management.ids import parse_id
from management.model_catalog import scopes_for, usable_models
from management.models import Request, Token
from management.ownership import RELATED, owner
from management.periods import Period, Span


@dataclass(frozen=True)
class Key:
    token: Token
    href: str | None
    note: str
    own: bool = False

    @property
    def name(self) -> str:
        return self.token.name


def describe_key(token: Token, viewer: Viewer, *, with_org: bool = False) -> Key:
    href = reverse("token_detail", args=[token.id]) if viewer.may_key(KeyAct.OPEN, token) else None
    account, own = token.service_account, False
    if account is not None:
        note = gettext("Service account · %(team)s") % {"team": account.team.name}
    elif token.user_id == viewer.user_id:
        note, own = gettext("Personal"), True
    else:
        note = token.user.email
    if with_org:
        found = owner(token)
        note += f" · {found.org.name if found is not None else NO_ORG}"
    return Key(token, href, note, own)


@dataclass(frozen=True)
class Filters:
    viewer: Viewer
    mine: bool
    keys: tuple[Key, ...]
    only_key: Key | None
    key: Key | None
    key_q: str
    matches: tuple[Key, ...]
    page_params: tuple[tuple[str, str], ...]
    models: tuple[str, ...]
    model: str | None
    span: Span
    period: Period
    requests: QuerySet[Request]

    @property
    def can_pick_org(self) -> bool:
        return self.viewer.can_switch_org

    @property
    def can_pick_scope(self) -> bool:
        return self.viewer.sees_others

    @property
    def org_wide(self) -> OrgWide | None:
        return None if self.mine or self.key is not None else self.viewer.org_wide

    @property
    def visibility(self) -> tuple[object, ...]:
        return (
            self.viewer.shown.cache_key,
            ("mine", self.viewer.user_id, *(k.token.id for k in self.keys)) if self.mine else (),
            self.key.token.id if self.key else None,
            self.model,
        )

    @property
    def spans(self) -> tuple[Span, ...]:
        return tuple(Span)

    @property
    def query(self) -> dict[str, str]:
        found = {}
        if self.can_pick_org:
            found["org"] = self.viewer.chosen.param
        if self.mine:
            found["scope"] = "mine"
        if self.key:
            found["key"] = str(self.key.token.id)
        if self.model:
            found["model"] = self.model
        found["span"] = self.span.value
        return found

    @property
    def search_params(self) -> tuple[tuple[str, str], ...]:
        return tuple((dict(self.page_params) | self.query).items())

    def log_url(self, **narrower: str) -> str:
        return f"{reverse('requests')}?{urlencode(self.query | narrower)}"


KEY_MATCHES = 20
MODEL_MAX = 255


_NOT_PAGE_PARAMS = frozenset(
    {"org", "scope", "key", "model", "span", "key_q", "before", "after", "request", "page"}
)


def key_search(text: str, *, by_owner: bool) -> Q:
    found = Q(name__icontains=text)
    return found | Q(user__email__icontains=text) if by_owner else found


def read_filters(
    viewer: Viewer, params: Mapping[str, str], *, default_span: Span, now: datetime
) -> Filters:
    sees_others = viewer.sees_others
    requests = requests_of(viewer.shown)
    if sees_others:
        names = set(model_names())
    else:
        used = requests.filter(model__in=model_names()).order_by().values_list("model", flat=True)
        names = {
            *usable_models(scopes_for(viewer.profile, viewer.teams(TeamAct.MINE))),
            *used.distinct(),
        }
    models = tuple(sorted(names, key=lambda name: (name.casefold(), name)))
    tokens = viewer.keys(KeyAct.SEE).select_related(*RELATED).order_by("name", "id")

    def describe(token: Token) -> Key:
        return describe_key(token, viewer, with_org=isinstance(viewer.chosen, EveryOrg))

    keys = tuple(describe(token) for token in tokens & viewer.keys(KeyAct.MINE))
    first_two = list(tokens[:2])
    only_key = describe(first_two[0]) if len(first_two) == 1 else None
    mine = sees_others and params.get("scope") == "mine"
    if mine:
        requests = requests.filter(token_id__in=[key.token.id for key in keys])

    key = None
    if raw_key := params.get("key", ""):
        if (key_id := parse_id(raw_key)) is None:
            raise Http404
        key = describe(get_object_or_404(tokens, pk=key_id))
        requests = requests.filter(token=key.token)

    key_q = params.get("key_q", "").replace("\x00", "").strip()
    matches: tuple[Key, ...] = ()
    if key_q:
        found = tokens.filter(key_search(key_q, by_owner=sees_others))
        matches = tuple(describe(token) for token in found[:KEY_MATCHES])

    # Any name may be asked for: the log links the names clients sent, which may not be served.
    model = params.get("model", "").replace("\x00", "")[:MODEL_MAX] or None
    if model is not None:
        requests = requests.filter(model=model)
        if model not in models:
            models = (*models, model)

    span = Span.parse(params.get("span", "")) or default_span
    period = Period.last(span, now)

    return Filters(
        viewer=viewer,
        mine=mine,
        keys=keys,
        only_key=only_key,
        key=key,
        key_q=key_q,
        matches=matches,
        page_params=tuple(
            (name, value) for name, value in params.items() if name not in _NOT_PAGE_PARAMS
        ),
        models=models,
        model=model,
        span=span,
        period=period,
        requests=requests,
    )
