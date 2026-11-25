import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models, transaction
from django.db.models import Q, QuerySet

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin
from management.models.org import Org
from management.models.service_account import ServiceAccount
from management.models.team import Team

if TYPE_CHECKING:
    from management.models import Request, Token

log = logging.getLogger("aqueduct")


class UserGroup(models.TextChoices):
    ADMIN = "admin", "Admin"
    ORG_ADMIN = "org-admin", "Org Admin"
    USER = "user", "User"


class UserProfile(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """
    Holds additional information related to the built-in Django User model.
    Each Django User should have one corresponding UserProfile.
    """

    # Link to the standard Django User model
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,  # Use settings.AUTH_USER_MODEL
        on_delete=models.CASCADE,
        related_name="profile",  # Access profile from user: user.profile
    )

    org = models.ForeignKey(
        Org,
        on_delete=models.PROTECT,  # Keep PROTECT if you don't want to delete Org if profiles exist
        related_name="user_profiles",
    )

    teams: "models.ManyToManyField[Team, UserProfile]" = models.ManyToManyField(
        Team, through="TeamMembership", related_name="member_profiles", blank=True
    )

    def __str__(self) -> str:
        return f"{self.user.email} (Profile - {self.org.name})"

    def save(self, *args: Any, **kwargs: Any) -> None:
        """An org move ends the memberships in teams of the old org."""
        update_fields = kwargs.get("update_fields")
        writes_org = update_fields is None or not {"org", "org_id"}.isdisjoint(update_fields)
        if self._state.adding or not writes_org:
            super().save(*args, **kwargs)
            return
        with transaction.atomic():
            # Compare with the stored row, not the loaded one: a deferred, stale or retried
            # instance would otherwise report a move that never happened, or miss one.
            stored = UserProfile.objects.filter(pk=self.pk).values_list("org_id", flat=True).first()
            super().save(*args, **kwargs)
            if stored is not None and stored != self.org_id:
                self._leave_other_orgs()

    def _leave_other_orgs(self) -> None:
        dropped = self.teammembership_set.exclude(team__org_id=self.org_id)
        email = self.user.email
        for team_id, team, is_admin in dropped.values_list("team_id", "team__name", "is_admin"):
            log.info(
                "Org move of %s ends their membership in team %s (%s), admin: %s.",
                email,
                team_id,
                team,
                is_admin,
            )
        owned = ServiceAccount.objects.filter(
            team__in=dropped.values("team_id"), token__user_id=self.user_id
        )
        for account in owned:
            log.warning(
                "%s left team %s but owns its service account %s (%s); transfer it.",
                email,
                account.team_id,
                account.pk,
                account.name,
            )
        dropped.delete()

    @cached_property
    def _group(self) -> UserGroup | None:
        """Read once per instance: the access checks below ask for it many times."""
        names = set(self.user.groups.values_list("name", flat=True))
        ranked = (UserGroup.ADMIN, UserGroup.ORG_ADMIN, UserGroup.USER)
        return next((group for group in ranked if group in names), None)

    @property
    def group(self) -> UserGroup:
        if self._group is None:
            raise ValidationError("User has no group")
        return self._group

    @group.setter
    def group(self, group: str) -> None:
        if group not in UserGroup.values:
            raise ValueError(f"Group {group} does not exist!")

        self.__dict__.pop("_group", None)
        # Clear existing groups first
        self.user.groups.clear()
        # Add the new group
        try:
            group_obj = Group.objects.get(name=group)
            self.user.groups.add(group_obj)
        except Group.DoesNotExist:
            # Handle case where the group doesn't exist in DB (shouldn't happen with check above)
            raise ObjectDoesNotExist(
                f"The group '{group}' does not exist in the database."
            ) from None

    def clean(self) -> None:
        """
        Validation moved from the old User model.
        Ensures assigned teams belong to the user's profile organization.
        """
        super().clean()
        # Assumes self.org is always set due to non-nullable ForeignKey.
        # Check teams only if the profile instance exists in the DB (has a PK).
        # This prevents issues when accessing M2M relations before the instance is saved.
        if self.pk:
            for team in self.teams.all():  # Query M2M relationship
                if team.org != self.org:
                    raise ValidationError(
                        f"Team '{team.name}' (Org: {team.org.name}) does not belong to "
                        f"the profile's organization '{self.org.name}'."
                    )
            # No need for the try/except ValueError, as self.pk ensures the instance is saved.

    def is_admin(self) -> bool:
        """A platform admin: the global `admin` group."""
        return self._group is UserGroup.ADMIN

    def is_org_admin(self, org_to_check: Org) -> bool:
        """Platform admins administer every org, org admins only their own."""
        return self.is_admin() or (
            self._group is UserGroup.ORG_ADMIN and self.org_id == org_to_check.pk
        )

    def is_team_admin(self, team_to_check: Team) -> bool:
        """An admin of the team's org, or a team admin by membership in the own org."""
        return (
            self.is_org_admin(team_to_check.org)
            or self.member_teams(admin=True).filter(pk=team_to_check.pk).exists()
        )

    def administered_orgs(self) -> QuerySet[Org]:
        if self.is_admin():
            return Org.objects.all()
        if self._group is UserGroup.ORG_ADMIN:
            return Org.objects.filter(pk=self.org_id)
        return Org.objects.none()

    def member_teams(self, *, admin: bool = False) -> QuerySet[Team]:
        """Teams of the own org the user is in, or is a team admin of with `admin`.

        A membership in a team of another org, which an org move can leave behind, grants
        nothing.
        """
        # One filter() call, so both conditions hold for the same membership row.
        mine = {"teammembership__is_admin": True} if admin else {}
        return Team.objects.filter(org_id=self.org_id, teammembership__user_profile=self, **mine)

    def visible_teams(self, org: Org | None = None) -> QuerySet[Team]:
        """Org admins see their org's teams, everyone else the teams they are in. A platform
        admin sees `org`'s teams, or every team without one."""
        if self.is_admin():
            return Team.objects.all() if org is None else Team.objects.filter(org=org)
        if self._group is UserGroup.ORG_ADMIN:
            return Team.objects.filter(org_id=self.org_id)
        return self.member_teams()

    def managed_teams(self) -> QuerySet[Team]:
        """Teams whose settings, members and service accounts the user may change."""
        if self._group in (UserGroup.ADMIN, UserGroup.ORG_ADMIN):
            return self.visible_teams()
        return self.member_teams(admin=True)

    def own_tokens(self) -> "QuerySet[Token]":
        """Personal keys: the user's keys that belong to no service account."""
        from management.models.token import Token

        return Token.objects.filter(user_id=self.user_id, service_account__isnull=True)

    def my_tokens(self) -> "QuerySet[Token]":
        """Personal keys and the keys of the teams the user is in."""
        return self.own_tokens() | self._team_tokens(self.member_teams())

    def visible_tokens(self, org: Org | None = None) -> "QuerySet[Token]":
        """Org admins see their org's keys, everyone else `my_tokens()`. A platform admin sees
        `org`'s keys, or every key without one."""
        from management.models.token import Token

        if self.is_admin():
            return Token.objects.all() if org is None else Token.objects.filter(_in_org(org))
        if self._group is UserGroup.ORG_ADMIN:
            return Token.objects.filter(_in_org(self.org))
        return self.my_tokens()

    def manageable_tokens(self) -> "QuerySet[Token]":
        """Personal keys and the keys of managed teams."""
        return self.own_tokens() | self._team_tokens(self.managed_teams())

    def visible_requests(self, org: Org | None = None) -> "QuerySet[Request]":
        from management.models.request import Request

        if self.is_admin() and org is None:
            return Request.objects.all()
        return Request.objects.filter(token__in=self.visible_tokens(org))

    @staticmethod
    def _team_tokens(teams: QuerySet[Team]) -> "QuerySet[Token]":
        from management.models.token import Token

        return Token.objects.filter(service_account__team__in=teams)


def _in_org(org: Org) -> Q:
    """Keys of `org`: its teams' service-account keys and its people's personal keys."""
    return Q(service_account__team__org=org) | Q(
        service_account__isnull=True, user__profile__org=org
    )
