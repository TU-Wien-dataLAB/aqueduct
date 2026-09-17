from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin


class Org(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """Represents an Organization."""

    name = models.CharField(verbose_name="Org name", max_length=255, unique=True)

    def __str__(self) -> str:
        return self.name


class Team(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """Represents a Team within an Organization."""

    name = models.CharField(verbose_name="Team name", max_length=255)
    description = models.TextField(blank=True)

    org = models.ForeignKey(Org, on_delete=models.CASCADE, related_name="teams")

    oauth_group_name = models.CharField(
        verbose_name="OAuth group name",
        max_length=255,
        blank=True,
        default="",
        db_index=True,
        help_text="The OAuth group that created this team (auto-managed, not user-editable)",
    )

    class Meta:
        unique_together = ("name", "org")

    def __str__(self) -> str:
        return f"{self.name} ({self.org.name})"

    @property
    def managed_by_oauth(self) -> bool:
        """Returns True if this team is managed by OAuth group synchronization."""
        return bool(self.oauth_group_name)


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


class TeamMembership(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)

    # Your extra data about the relationship
    is_admin = models.BooleanField(default=False)
    date_added = models.DateField(auto_now_add=True)

    class Meta:
        # Ensure a user can only be in a team once
        unique_together = ("user_profile", "team")

    def __str__(self) -> str:
        return f"{self.user_profile} in {self.team}{' (Admin)' if self.is_admin else ''}"


class ServiceAccount(models.Model):
    """Represents a Service Account, typically associated with a Team."""

    name = models.CharField(verbose_name="Service Account name", max_length=255)
    description = models.TextField(blank=True)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="service_accounts")

    class Meta:
        unique_together = ("name", "team")

    def __str__(self) -> str:
        # Handle case where team might not be set yet
        return f"{self.name} (Team: {self.team.name if self.team_id else 'N/A'})"

    def clean(self) -> None:
        """
        Validates that the associated team does not exceed the maximum
        number of service accounts allowed.
        """
        super().clean()

        # This check only makes sense if the team field is actually set.
        # During object creation via a form that *doesn't* include the 'team' field,
        # self.team might be None or raise an exception when accessed before being saved,
        # depending on how the instance is constructed prior to clean().
        # It relies on the team being assigned *before* clean() is called.
        if self.team_id:
            try:
                # Get the actual team object - needed if not already loaded
                # This might cause an extra query if team wasn't select_related
                team_instance = self.team
                limit = getattr(settings, "MAX_SERVICE_ACCOUNTS_PER_TEAM", 10)

                # Query existing accounts for this team
                query = ServiceAccount.objects.filter(team=team_instance)

                # If updating an existing instance, exclude self from the count
                if self.pk:
                    query = query.exclude(pk=self.pk)

                current_count = query.count()

                # Check if adding this one would exceed the limit
                # This check is primarily for *new* instances (self.pk is None)
                if current_count >= limit:
                    # Raising ValidationError here will attach the error to the form
                    # if called via form.is_valid()
                    raise ValidationError(
                        {
                            # You can attach the error to a specific field or make it non-field
                            # None: f"Team '{team_instance.name}' has reached the maximum limit of
                            # {limit} service accounts."
                            "team": f"Team '{team_instance.name}' has reached the maximum limit of "
                            f"{limit} service accounts."
                            # Attach to team conceptually
                        }
                    )

            except Team.DoesNotExist:
                # This case shouldn't happen if ForeignKey validation runs,
                # but good to handle defensively.
                raise ValidationError("Associated team does not exist.") from None
