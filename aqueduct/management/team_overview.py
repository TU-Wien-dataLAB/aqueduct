from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from django.contrib.auth import get_user_model
from django.core.paginator import Page, Paginator
from django.db.models import Count, OuterRef, QuerySet, Subquery
from django.db.models.functions import Coalesce
from django.utils.translation import gettext, gettext_noop

from management.activity import Activity, activity_by
from management.charts import token_widths
from management.keys import Expiry, key_activity
from management.limits import ConfiguredLimits, LimitsPanel, configured, panels
from management.models import (
    LimitSet,
    ServiceAccount,
    Team,
    TeamMembership,
    Token,
    UserGroup,
    UserProfile,
)
from management.models.exclusions import Exclusion, excluded
from management.periods import week

PAGE_SIZE = 50


class Role(StrEnum):
    ADMIN = gettext_noop("Platform admin")
    ORG_ADMIN = gettext_noop("Org admin")
    TEAM_ADMIN = gettext_noop("Team admin")
    MEMBER = gettext_noop("Member")

    @property
    def label(self) -> str:
        return gettext(self.value)


def role_of(groups: set[str], membership: TeamMembership) -> Role:
    match UserGroup.of(groups):
        case UserGroup.ADMIN:
            return Role.ADMIN
        case UserGroup.ORG_ADMIN if membership.user_profile.org_id == membership.team.org_id:
            return Role.ORG_ADMIN
        case _:
            return Role.TEAM_ADMIN if membership.is_admin else Role.MEMBER


@dataclass(frozen=True)
class MemberRow:
    profile: UserProfile
    role: Role
    is_viewer: bool
    owned: tuple[ServiceAccount, ...]
    is_team_admin: bool
    can_change_admin: bool


@dataclass(frozen=True)
class TeamRow:
    team: Team
    is_team_admin: bool
    members: int
    service_accounts: int
    limit: int | None
    activity: Activity
    bar_width: int


def _count_per_team(rows: QuerySet) -> Coalesce:
    return Coalesce(
        Subquery(
            rows.filter(team=OuterRef("pk")).values("team").annotate(n=Count("pk")).values("n")
        ),
        0,
    )


@dataclass(frozen=True)
class TeamList:
    rows: tuple[TeamRow, ...]
    pages: Page


def team_rows(
    teams: QuerySet[Team], admin_of: frozenset[int], query: str, page: str | int, now: datetime
) -> TeamList:
    pages = Paginator(
        Team.objects.filter(id__in=teams.values("id"), name__icontains=query)
        .select_related("org")
        .annotate(
            member_count=_count_per_team(TeamMembership.objects.all()),
            service_account_count=_count_per_team(ServiceAccount.objects.all()),
        )
        .order_by("name", "id"),
        PAGE_SIZE,
    ).get_page(page)
    found = list(pages)
    activity = activity_by("token__service_account__team_id", (t.id for t in found), week(now))
    widths = token_widths(activity)
    rows = tuple(
        TeamRow(
            team=team,
            is_team_admin=team.id in admin_of,
            members=team.member_count,
            service_accounts=team.service_account_count,
            limit=LimitSet.from_objects(team, [team.org]).requests_per_minute,
            activity=activity.get(team.id, Activity()),
            bar_width=widths.get(team.id, 0),
        )
        for team in found
    )
    return TeamList(rows=rows, pages=pages)


@dataclass(frozen=True)
class ServiceAccountRow:
    service_account: ServiceAccount
    token: Token | None
    activity: Activity
    expiry: Expiry | None
    limits: LimitsPanel | None


@dataclass(frozen=True)
class TeamPage:
    service_accounts: tuple[ServiceAccountRow, ...]
    members: tuple[MemberRow, ...]
    member_pages: Page
    member_count: int
    limits: ConfiguredLimits
    excluded_models: tuple[str, ...]
    excluded_mcp_servers: tuple[str, ...]


def team_page(
    team: Team,
    viewer: UserProfile,
    can_administer: bool,
    member_query: str,
    member_page: str | int,
    now: datetime,
    may_open: Callable[[Token], bool],
) -> TeamPage:
    accounts = list(team.service_accounts.select_related("token__user").order_by("name"))
    tokens = {sa.id: getattr(sa, "token", None) for sa in accounts}
    activity = key_activity((t.id for t in tokens.values() if t is not None), now)
    found = panels((t for t in tokens.values() if t is not None and may_open(t)), now)
    service_accounts = tuple(
        ServiceAccountRow(
            service_account=sa,
            token=token,
            activity=activity.get(token.id, Activity()) if token else Activity(),
            expiry=Expiry.of(token.expires_at, now) if token else None,
            limits=found.get(token.id) if token else None,
        )
        for sa, token in ((sa, tokens[sa.id]) for sa in accounts)
    )
    owned = defaultdict(list)
    for sa in accounts:
        if tokens[sa.id] is not None:
            owned[tokens[sa.id].user_id].append(sa)

    all_memberships = TeamMembership.objects.filter(team=team)
    member_pages = Paginator(
        all_memberships.filter(user_profile__user__email__icontains=member_query)
        .select_related("team", "user_profile__user")
        .order_by("-is_admin", "user_profile__user__email", "id"),
        PAGE_SIZE,
    ).get_page(member_page)
    groups = defaultdict(set)
    for user_id, name in (
        get_user_model()
        .groups.through.objects.filter(user_id__in=[m.user_profile.user_id for m in member_pages])
        .values_list("user_id", "group__name")
    ):
        groups[user_id].add(name)
    members = []
    for membership in member_pages:
        profile = membership.user_profile
        role = role_of(groups[profile.user_id], membership)
        is_viewer = profile.id == viewer.id
        members.append(
            MemberRow(
                profile=profile,
                role=role,
                is_viewer=is_viewer,
                owned=tuple(owned[profile.user_id]),
                is_team_admin=membership.is_admin,
                can_change_admin=can_administer
                and role in (Role.TEAM_ADMIN, Role.MEMBER)
                and not is_viewer,
            )
        )

    return TeamPage(
        service_accounts=service_accounts,
        members=tuple(members),
        member_pages=member_pages,
        member_count=all_memberships.count() if member_query else member_pages.paginator.count,
        limits=configured(LimitSet.from_objects(team, [team.org])),
        excluded_models=tuple(sorted(excluded(team, Exclusion.MODELS))),
        excluded_mcp_servers=tuple(sorted(excluded(team, Exclusion.MCP_SERVERS))),
    )
