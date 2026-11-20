"""The request boundary for access: a platform admin's org choice and the request's Viewer.

The Viewer is memoized on the request, so the view and the shell share one resolution.
"""

from functools import partial
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from django.contrib.sessions.backends.base import SessionBase
from django.http import HttpRequest
from django.shortcuts import get_object_or_404

from management.access import EveryOrg, OrgWide, Viewer, WholeOrg
from management.ids import parse_id
from management.models import Org, UserProfile

SESSION_KEY: Final = "org"
EVERY_ORG: Final = EveryOrg.param
ORG_BOUND_PARAMS: Final = ("org", "key", "key_q", "before", "after", "request", "page")
_MEMO: Final = "_aqueduct_viewer"


def _remember(session: SessionBase, choice: OrgWide) -> None:
    value = EVERY_ORG if isinstance(choice, EveryOrg) else str(choice.org.pk)
    if session.get(SESSION_KEY) != value:
        session[SESSION_KEY] = value


def chosen(request: HttpRequest, own: Org) -> OrgWide:
    raw = request.GET.get("org", request.session.get(SESSION_KEY))
    choice: OrgWide = EveryOrg()
    if raw != EVERY_ORG:
        pk = parse_id(raw)
        org = Org.objects.filter(pk=pk).first() if pk is not None else None
        if org is None:
            return WholeOrg(own)
        choice = WholeOrg(org)
    _remember(request.session, choice)
    return choice


def switch(request: HttpRequest, raw: str) -> OrgWide:
    choice = EveryOrg() if raw == EVERY_ORG else WholeOrg(get_object_or_404(Org, pk=parse_id(raw)))
    _remember(request.session, choice)
    return choice


def without_org_params(url: str) -> str:
    parts = urlsplit(url)
    query = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name not in ORG_BOUND_PARAMS
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


def viewer_of(request: HttpRequest) -> Viewer | None:
    if hasattr(request, _MEMO):
        memo: Viewer | None = getattr(request, _MEMO)
        return memo
    viewer = None
    if request.user.is_authenticated:
        profile = (
            UserProfile.objects.select_related("org", "user")
            .filter(user_id=request.user.pk)
            .first()
        )
        if profile is not None:
            viewer = Viewer.of(profile, partial(chosen, request))
    setattr(request, _MEMO, viewer)
    return viewer
