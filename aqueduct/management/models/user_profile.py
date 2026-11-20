import logging
from collections.abc import Collection
from typing import Any

from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models, transaction

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin
from management.models.org import Org
from management.models.service_account import ServiceAccount
from management.models.team import Team

log = logging.getLogger("aqueduct")


class UserGroup(models.TextChoices):
    ADMIN = "admin", "Admin"
    ORG_ADMIN = "org-admin", "Org Admin"
    USER = "user", "User"

    @classmethod
    def of(cls, names: Collection[str]) -> "UserGroup | None":
        return next(
            (group for group in (cls.ADMIN, cls.ORG_ADMIN, cls.USER) if group in names), None
        )


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

    @property
    def group(self) -> UserGroup:
        groups = self.user.groups
        if groups.filter(name=UserGroup.ADMIN).exists():
            return UserGroup.ADMIN
        if groups.filter(name=UserGroup.ORG_ADMIN).exists():
            return UserGroup.ORG_ADMIN
        if groups.filter(name=UserGroup.USER).exists():
            return UserGroup.USER
        raise ValidationError("User has no group")

    @group.setter
    def group(self, group: str) -> None:
        if group not in UserGroup.values:
            raise ValueError(f"Group {group} does not exist!")

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

    @property
    def rank(self) -> UserGroup:
        return UserGroup.of(set(self.user.groups.values_list("name", flat=True))) or UserGroup.USER
