import hashlib
import secrets
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models
from django.utils import timezone

from gateway.config import resolve_model_alias
from management.models.mixins import (
    LimitMixin,
    LimitSet,
    MCPServerExclusionMixin,
    ModelExclusionMixin,
)

_T = TypeVar("_T")


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


class Token(models.Model):
    """
    Represents an authentication token (e.g., API Key), associated with a User
    and optionally a Service Account.
    """

    name = models.CharField(verbose_name="Token name", max_length=255, null=False)
    # Link to the standard Django User model
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # Use settings.AUTH_USER_MODEL
        on_delete=models.CASCADE,
        related_name="custom_auth_tokens",  # Changed related_name to avoid potential clashes
    )
    # This structure implies the Token is *created by* a User, potentially *for* a Service Account.
    service_account = models.OneToOneField(
        ServiceAccount, on_delete=models.CASCADE, related_name="token", null=True, blank=True
    )
    # Store hash and preview, not the original key
    key_hash = models.CharField(
        max_length=64,  # SHA-256 hash length
        unique=True,
        editable=False,
        help_text="SHA-256 hash of the token key.",
    )
    key_preview = models.CharField(
        max_length=12,  # e.g., "T0K3..."
        editable=False,
        help_text="First few characters of the original token key for display.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)  # Optional expiry

    def __str__(self) -> str:
        if self.service_account:
            return f"'{self.name}' ({self.service_account.name})"
        return f"'{self.name}'"

    def save(self, *args: Any, **kwargs: Any) -> None:
        """
        Ensures key_hash and key_preview are set before the first save.
        """
        if not self.pk and (not self.key_hash or not self.key_preview):
            # This check ensures _set_new_key() was called before the first save.
            raise ValueError(
                "Token cannot be saved without key_hash and key_preview. "
                "Call _set_new_key() before saving."
            )
        super().save(*args, **kwargs)

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= timezone.now()

    # The clean method checking for self.user is implicitly handled by the ForeignKey
    # unless null=True is added to the user field, which doesn't seem intended here.

    @staticmethod
    def _generate_secret_key(prefix: str = "sk-") -> str:
        """Generates a unique secret token key."""
        return prefix + secrets.token_urlsafe(nbytes=32)

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hashes the key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_preview(key: str, start: int = 3, end: int = 4) -> str:
        """Generates a preview string for the token key."""
        if not key:
            return ""
        return f"{key[:start]}...{key[-end:]}"

    def _set_new_key(self) -> str:
        """
        Generates a new secret key, sets the instance's hash and preview fields.
        Returns the generated secret key. Does NOT save the instance.
        """
        secret_key = self._generate_secret_key()
        self.key_hash = self._hash_key(secret_key)
        self.key_preview = self._generate_preview(secret_key)
        return secret_key

    def regenerate_key(self) -> str:
        """
        Generates a new secret key, updates the hash and preview, saves the instance,
        and returns the *new secret key*.
        """
        new_secret_key = self._set_new_key()  # Use the helper method
        self.save(update_fields=["key_hash", "key_preview"])  # Save the changes
        return new_secret_key  # Return the original new key

    def clean(self) -> None:
        """
        Model-level validation. The user-specific token limit is checked in the form.
        """
        super().clean()

    def _get_from_hierarchy(self, retrieval_function: Callable[..., _T]) -> _T:
        token_instance = Token.objects.select_related(
            "user__profile__org", "service_account__team__org"
        ).get(pk=self.pk)

        if token_instance.service_account:
            team = token_instance.service_account.team
            org = team.org
            return retrieval_function(team, org)
        profile = token_instance.user.profile
        org = profile.org
        return retrieval_function(profile, org)

    def get_limit(self) -> "LimitSet":
        """
        Determines the effective rate limits for this token, returning a LimitSet dataclass.
        Assumes database integrity for related objects.

        Hierarchy Rules:
        - Service Account Token: Uses Team limits, falls back to Org limits.
        - User Token: Uses UserProfile limits, falls back to Org limits.
        """
        return self._get_from_hierarchy(LimitSet.from_objects)

    @classmethod
    def _exclusion_list_from_objects(
        cls,
        specific_exclusion: Optional["ModelExclusionMixin"],
        org_exclusion: Optional["ModelExclusionMixin"],
    ) -> list[str]:
        exclusion_list: list[str] = specific_exclusion.excluded_models if specific_exclusion else []
        if specific_exclusion and specific_exclusion.merge_exclusion_lists:
            org_exclusion_list: list[str] = org_exclusion.excluded_models if org_exclusion else []
            exclusion_list = exclusion_list + org_exclusion_list
            if org_exclusion and org_exclusion.merge_exclusion_lists:
                settings_exclusion_list: list[str] = getattr(
                    settings, "AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST", []
                )
                exclusion_list = exclusion_list + settings_exclusion_list

        return list(set(exclusion_list))

    def model_exclusion_list(self) -> list[str]:
        """
        Determines if a model is excluded for this token, returning either True or False.
        Assumes database integrity for related objects.

        Hierarchy Rules:
        - Service Account Token: Uses the Team exclusion list, falls back to the Org exclusion list.
        - User Token: Uses the UserProfile exclusion list, falls back to the Org exclusion list.
        """
        return self._get_from_hierarchy(Token._exclusion_list_from_objects)

    def model_excluded(self, model: str) -> bool:
        # Resolve alias to actual model name before checking exclusion
        resolved_model = resolve_model_alias(model)
        return resolved_model in self.model_exclusion_list()

    @classmethod
    def _mcp_server_exclusion_list_from_objects(
        cls,
        specific_exclusion: Optional["MCPServerExclusionMixin"],
        org_exclusion: Optional["MCPServerExclusionMixin"],
    ) -> list[str]:
        exclusion_list: list[str] = (
            specific_exclusion.excluded_mcp_servers if specific_exclusion else []
        )
        if specific_exclusion and specific_exclusion.merge_mcp_server_exclusion_lists:
            org_exclusion_list: list[str] = (
                org_exclusion.excluded_mcp_servers if org_exclusion else []
            )
            exclusion_list = exclusion_list + org_exclusion_list
            if org_exclusion and org_exclusion.merge_mcp_server_exclusion_lists:
                settings_exclusion_list: list[str] = getattr(
                    settings, "AQUEDUCT_DEFAULT_MCP_SERVER_EXCLUSION_LIST", []
                )
                exclusion_list = exclusion_list + settings_exclusion_list

        return list(set(exclusion_list))

    def mcp_server_exclusion_list(self) -> list[str]:
        """
        Determines if an MCP server is excluded for this token,
        returning a list of excluded servers.
        Assumes database integrity for related objects.

        Hierarchy Rules:
        - Service Account Token: Uses the Team exclusion list, falls back to the Org exclusion list.
        - User Token: Uses the UserProfile exclusion list, falls back to the Org exclusion list.
        """
        return self._get_from_hierarchy(Token._mcp_server_exclusion_list_from_objects)

    def mcp_server_excluded(self, server_name: str) -> bool:
        return server_name in self.mcp_server_exclusion_list()

    @classmethod
    def find_by_key(cls, key_value: str) -> Optional["Token"]:
        """
        Finds a token by its original (unhashed) key value.
        Returns the Token instance or None if not found.
        """
        if not key_value:
            return None
        try:
            hashed_key = cls._hash_key(key_value)
            # Use select_related for efficiency if you often need related objects after lookup
            return cls.objects.select_related(
                "user__profile__org", "service_account__team__org"
            ).get(key_hash=hashed_key)
        except cls.DoesNotExist:
            return None
