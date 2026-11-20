from dataclasses import dataclass
from functools import reduce
from operator import or_

from django.conf import settings
from django.db.models import Q

from management.limits import ConfiguredLimits, configured
from management.models import LimitSet, Org, Team
from management.models.exclusions import Exclusion, excluded
from management.models.mixins import LIMIT_FIELDS


@dataclass(frozen=True)
class UnavailableRow:
    team: Team | None
    models: tuple[str, ...]
    mcp_servers: tuple[str, ...]


@dataclass(frozen=True)
class OrgPage:
    teams: int
    limits: ConfiguredLimits
    teams_with_limits: int
    people_with_limits: int
    max_personal_keys: int
    unavailable: tuple[UnavailableRow, ...]


SETS_LIMITS = reduce(or_, (Q(**{f"{field}__isnull": False}) for field in LIMIT_FIELDS))


def org_page(org: Org) -> OrgPage:
    teams = list(org.teams.order_by("name"))
    everyone = UnavailableRow(
        team=None,
        models=tuple(sorted(excluded(org, Exclusion.MODELS))),
        mcp_servers=tuple(sorted(excluded(org, Exclusion.MCP_SERVERS))),
    )
    team_rows = tuple(
        UnavailableRow(
            team=team,
            models=tuple(sorted(team.excluded_models)),
            mcp_servers=tuple(sorted(team.excluded_mcp_servers)),
        )
        for team in teams
        if team.excluded_models or team.excluded_mcp_servers
    )
    return OrgPage(
        teams=len(teams),
        limits=configured(LimitSet.from_objects(org)),
        teams_with_limits=org.teams.filter(SETS_LIMITS).count(),
        people_with_limits=org.user_profiles.filter(SETS_LIMITS).count(),
        max_personal_keys=settings.MAX_USER_TOKENS,
        unavailable=(everyone, *team_rows)
        if everyone.models or everyone.mcp_servers
        else team_rows,
    )
