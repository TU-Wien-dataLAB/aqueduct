"""Who may see, open, use and manage which keys, teams and orgs.

A viewer sees their personal keys, the service-account keys of the teams of their own org
they belong to, and every key of their org if they are an org admin. A platform admin opens
every key and lists the org they chose, or every org. A membership in a team of another org
is an invalid state an org move leaves behind, and grants nothing.

Every question has a SQL form and a Python form next to each other; `tests/test_access.py`
proves they agree.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from django.db.models import Q, QuerySet
from django.utils.translation import gettext

from management.models import Org, Request, Team, TeamMembership, Token, UserGroup, UserProfile
from management.ownership import is_personal_of, of_orgs, of_teams, org_id, personal, team_of


class KeyAct(StrEnum):
    SEE = "see"
    """Lists: request log, Usage, All keys, files, batches, vector stores."""
    OPEN = "open"
    """The key page by URL, whatever the org switch shows."""
    MINE = "mine"
    """Own personal keys and the member teams' keys: the API keys page and "My keys"."""
    OWN = "own"
    """Own personal keys: edit, delete."""
    MANAGE = "manage"
    """Own personal keys and the keys of managed teams: regenerate."""
    SYNC = "sync"
    """Refresh a vector store: a listed key that is the viewer's, or a member or managed
    team's."""


class TeamAct(StrEnum):
    SEE = "see"
    """The team list. Follows the org switch."""
    OPEN = "open"
    """The team page."""
    MINE = "mine"
    """Teams of the own org the viewer belongs to: quotas, model availability."""
    MANAGE = "manage"
    """Edit the team, its members and its service accounts."""
    ADMINISTER = "administer"
    """Delete the team, set its availability, appoint team admins."""


@dataclass(frozen=True)
class EveryOrg:
    param: ClassVar[str] = "all"
    cache_key: ClassVar[tuple[object, ...]] = ("all orgs",)

    @property
    def name(self) -> str:
        return gettext("All organizations")

    def keys(self) -> Q:
        return Q()

    def has_key(self, token: Token) -> bool:
        return True

    def teams(self) -> Q:
        return Q()

    def has_team(self, team: Team) -> bool:
        return True

    def orgs(self) -> Q:
        return Q()

    def has_org(self, org: Org) -> bool:
        return True


@dataclass(frozen=True)
class WholeOrg:
    org: Org

    @property
    def param(self) -> str:
        return str(self.org.pk)

    @property
    def name(self) -> str:
        return self.org.name

    @property
    def cache_key(self) -> tuple[object, ...]:
        return ("org", self.org.pk)

    def keys(self) -> Q:
        return of_orgs({self.org.pk})

    def has_key(self, token: Token) -> bool:
        return org_id(token) == self.org.pk

    def teams(self) -> Q:
        return Q(org_id=self.org.pk)

    def has_team(self, team: Team) -> bool:
        return team.org_id == self.org.pk

    def orgs(self) -> Q:
        return Q(pk=self.org.pk)

    def has_org(self, org: Org) -> bool:
        return org.pk == self.org.pk


@dataclass(frozen=True)
class Memberships:
    user_id: int
    team_ids: frozenset[int]

    @property
    def cache_key(self) -> tuple[object, ...]:
        return ("keys of", self.user_id, *sorted(self.team_ids))

    def keys(self) -> Q:
        return personal(self.user_id) | of_teams(self.team_ids)

    def has_key(self, token: Token) -> bool:
        team = team_of(token)
        return is_personal_of(token, self.user_id) or (
            team is not None and team.pk in self.team_ids
        )

    def teams(self) -> Q:
        return Q(pk__in=self.team_ids)

    def has_team(self, team: Team) -> bool:
        return team.pk in self.team_ids

    def orgs(self) -> Q:
        return Q(pk__in=())

    def has_org(self, org: Org) -> bool:
        return False


OrgWide = EveryOrg | WholeOrg
Extent = OrgWide | Memberships


@dataclass(frozen=True)
class Viewer:
    """A signed-in user's access, resolved once per request by `org_choice.viewer_of`.

    `member_of` holds only teams of the own org, `admin_of` is a subset of it, and `chosen`
    is the own org unless the viewer is a platform admin.
    """

    profile: UserProfile
    rank: UserGroup
    member_of: frozenset[int]
    admin_of: frozenset[int]
    chosen: OrgWide

    @classmethod
    def of(cls, profile: UserProfile, choose: Callable[[Org], OrgWide] | None = None) -> "Viewer":
        """`choose` gives a platform admin's org choice from their own org; it is asked only
        for a platform admin, since it may read and write the session."""
        # An org move deletes the old org's memberships, but a member added in a transaction
        # that races the move still lands in the old org.
        rows = TeamMembership.objects.filter(
            user_profile=profile, team__org_id=profile.org_id
        ).values_list("team_id", "is_admin")
        memberships = dict(rows)
        rank = profile.rank
        return cls(
            profile=profile,
            rank=rank,
            member_of=frozenset(memberships),
            admin_of=frozenset(team for team, is_admin in memberships.items() if is_admin),
            chosen=choose(profile.org)
            if choose is not None and rank is UserGroup.ADMIN
            else WholeOrg(profile.org),
        )

    @property
    def user_id(self) -> int:
        return self.profile.user_id

    @property
    def org(self) -> Org:
        return self.profile.org

    @property
    def can_switch_org(self) -> bool:
        return self.rank is UserGroup.ADMIN

    @property
    def _mine(self) -> Memberships:
        return Memberships(self.user_id, self.member_of)

    @property
    def _reach(self) -> Extent:
        match self.rank:
            case UserGroup.ADMIN:
                return EveryOrg()
            case UserGroup.ORG_ADMIN:
                return WholeOrg(self.org)
            case UserGroup.USER:
                return self._mine

    @property
    def shown(self) -> Extent:
        return self.chosen if self.rank is UserGroup.ADMIN else self._reach

    @property
    def org_wide(self) -> OrgWide | None:
        shown = self.shown
        return None if isinstance(shown, Memberships) else shown

    @property
    def sees_others(self) -> bool:
        return self.org_wide is not None

    @property
    def _managing(self) -> Extent:
        match self.rank:
            case UserGroup.ADMIN | UserGroup.ORG_ADMIN:
                return self._reach
            case UserGroup.USER:
                return Memberships(self.user_id, self.admin_of)

    @property
    def _administering(self) -> Extent:
        match self.rank:
            case UserGroup.ADMIN | UserGroup.ORG_ADMIN:
                return self._reach
            case UserGroup.USER:
                return Memberships(self.user_id, frozenset())

    def keys(self, act: KeyAct) -> QuerySet[Token]:
        return Token.objects.filter(self._key_filter(act))

    def _key_filter(self, act: KeyAct) -> Q:
        match act:
            case KeyAct.SEE:
                return self.shown.keys()
            case KeyAct.OPEN:
                return self._reach.keys()
            case KeyAct.MINE:
                return self._mine.keys()
            case KeyAct.OWN:
                return personal(self.user_id)
            case KeyAct.MANAGE:
                return personal(self.user_id) | of_teams(self.teams(TeamAct.MANAGE))
            case KeyAct.SYNC:
                return self.shown.keys() & (
                    self._mine.keys() | of_teams(self.teams(TeamAct.MANAGE))
                )

    def may_key(self, act: KeyAct, token: Token) -> bool:
        match act:
            case KeyAct.SEE:
                return self.shown.has_key(token)
            case KeyAct.OPEN:
                return self._reach.has_key(token)
            case KeyAct.MINE:
                return self._mine.has_key(token)
            case KeyAct.OWN:
                return is_personal_of(token, self.user_id)
            case KeyAct.MANAGE:
                team = team_of(token)
                return is_personal_of(token, self.user_id) or (
                    team is not None and self.may_team(TeamAct.MANAGE, team)
                )
            case KeyAct.SYNC:
                team = team_of(token)
                return self.shown.has_key(token) and (
                    self._mine.has_key(token)
                    or (team is not None and self.may_team(TeamAct.MANAGE, team))
                )

    def _team_extent(self, act: TeamAct) -> Extent:
        match act:
            case TeamAct.SEE:
                return self.shown
            case TeamAct.OPEN:
                return self._reach
            case TeamAct.MINE:
                return self._mine
            case TeamAct.MANAGE:
                return self._managing
            case TeamAct.ADMINISTER:
                return self._administering

    def teams(self, act: TeamAct) -> QuerySet[Team]:
        return Team.objects.filter(self._team_extent(act).teams())

    def may_team(self, act: TeamAct, team: Team) -> bool:
        return self._team_extent(act).has_team(team)

    def orgs(self) -> QuerySet[Org]:
        return Org.objects.filter(self._administering.orgs())

    def administers(self, org: Org) -> bool:
        return self._administering.has_org(org)

    def leaving(self, team: Team) -> "Viewer":
        return replace(
            self, member_of=self.member_of - {team.pk}, admin_of=self.admin_of - {team.pk}
        )


_LISTED_KEYS = 500


def _listed(tokens: QuerySet[Token]) -> QuerySet[Request]:
    """Postgres estimates rows for a literal id list but not for a subquery, so up to
    `_LISTED_KEYS` ids keep the plan on the (token, timestamp) index."""
    ids = list(tokens.order_by().values_list("id", flat=True)[: _LISTED_KEYS + 1])
    if len(ids) > _LISTED_KEYS:
        return Request.objects.filter(token__in=tokens)
    return Request.objects.filter(token_id__in=ids)


def requests_of(extent: Extent) -> QuerySet[Request]:
    if isinstance(extent, EveryOrg):
        return Request.objects.all()
    return _listed(Token.objects.filter(extent.keys()))
