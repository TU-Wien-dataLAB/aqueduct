from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any, Final

from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import Q
from django.db.models.functions import Coalesce

if TYPE_CHECKING:
    from management.models import Team, Token, UserProfile

TEAM: Final = "service_account__team"
OWNERS: Final = (TEAM, "user__profile")
RELATED: Final = tuple(f"{path}__org" for path in OWNERS)


def key_org(prefix: str = "") -> Coalesce:
    return Coalesce(*(f"{prefix}{path}" for path in RELATED))


def of_orgs(org_ids: Collection[int]) -> Q:
    found = Q()
    absent: dict[str, bool] = {}
    for path in OWNERS:
        found |= Q(**absent, **{f"{path}__org__in": org_ids})
        absent[f"{path.split('__')[0]}__isnull"] = True
    return found


def _hop(obj: models.Model, name: str) -> Any:
    try:
        return getattr(obj, name)
    except ObjectDoesNotExist:
        return None


def _owners(token: "Token") -> "Iterator[Team | UserProfile]":
    for path in OWNERS:
        found: Any = token
        for name in path.split("__"):
            found = _hop(found, name)
            if found is None:
                break
        if found is not None:
            yield found


def owner(token: "Token") -> "Team | UserProfile | None":
    return next(_owners(token), None)


def org_id(token: "Token") -> int | None:
    return next((found.org_id for found in _owners(token) if found.org_id is not None), None)


def personal(user_id: int | None, prefix: str = "") -> Q:
    return Q(**{f"{prefix}user_id": user_id, f"{prefix}service_account__isnull": True})


def is_personal_of(token: "Token", user_id: int | None) -> bool:
    return token.service_account_id is None and token.user_id == user_id


def of_teams(team_ids: "Collection[int] | models.QuerySet[Team]", prefix: str = "") -> Q:
    return Q(**{f"{prefix}{TEAM}__in": team_ids})


def team_of(token: "Token") -> "Team | None":
    return token.service_account.team if token.service_account is not None else None
