from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin
from management.models.org import Org
from management.models.team import Team


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
        """Checks if the user has the global 'admin' group."""
        try:
            return self.group == UserGroup.ADMIN
        except ValidationError:  # Raised if user has no valid group assigned
            return False

    def is_org_admin(self, org_to_check: Org) -> bool:
        """
        Checks if the user is an administrator for the given organization.
        Rules:
        1. Superusers are admins of everything.
        2. Users with the 'org-admin' group are admins ONLY of their own organization.
        """
        try:
            user_group = self.group
            if user_group == UserGroup.ADMIN:
                return True
            if user_group == UserGroup.ORG_ADMIN:
                return self.org == org_to_check
        except ValidationError:
            return False
        return False

    def is_team_admin(self, team_to_check: Team) -> bool:
        """
        Checks if the user is an administrator for the given team.
        Rules:
        1. If the user is an admin of the team's organization, they are admin of the team.
        2. Otherwise, check the specific TeamMembership for the 'is_admin' flag.
        """
        # First, check if they are an admin of the team's parent organization
        if self.is_org_admin(team_to_check.org):
            return True

        # If not an org admin, check the specific membership for this team
        try:
            # Assumes TeamMembership model exists and is related via user_profile.
            # Django ensures 'teammembership_set' exists if TeamMembership has a ForeignKey
            # to UserProfile.
            membership = self.teammembership_set.get(team=team_to_check)
        except ObjectDoesNotExist:
            # No specific membership record found for this user and team
            return False
        else:
            return membership.is_admin
