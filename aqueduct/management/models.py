# models.py
import dataclasses
import hashlib
import logging
import secrets
from collections.abc import Callable
from typing import ClassVar, Literal, Optional

import openai.types
from django.conf import settings
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models
from django.db.models import BooleanField, JSONField
from django.utils import timezone

from gateway.config import resolve_model_alias

log = logging.getLogger("aqueduct")


# ── Legacy ID generators ──────────────────────────────────────────────
# These are referenced by historical migrations (0004, 0005) and must remain
# importable from management.models.  They are no longer used at runtime;
# migration 0008_file_proxy removes the defaults that depend on them.
def generate_file_id() -> str:
    return f"file-{secrets.token_hex(12)}"


def generate_batch_id() -> str:
    return f"batch-{secrets.token_hex(12)}"


@dataclasses.dataclass(frozen=True)  # frozen=True makes instances immutable
class LimitSet:
    """Represents a resolved set of rate limits."""

    requests_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None

    # Add future limit fields here with default None

    @classmethod
    def from_objects(cls, specific_limiter: Optional["LimitMixin"], org_limiter: Optional["LimitMixin"]) -> "LimitSet":
        """
        Creates a LimitSet by resolving limits from a specific limiter
        (like Team or UserProfile) and a fallback Org limiter object.

        Args:
            specific_limiter: The object with the primary limits (e.g., Team, UserProfile).
            org_limiter: The object with the fallback limits (Org).

        Returns:
            A LimitSet instance with the effectively resolved limits.
        """

        # Helper to resolve a single limit field value using the hierarchy
        def _resolve(field_name: str) -> int | None:
            # Get value from the specific level first
            # Use getattr for safe access, defaulting to None if field absent
            specific_value = getattr(specific_limiter, field_name, None) if specific_limiter else None
            if specific_value is not None:
                # Return immediately if a specific limit is set
                return specific_value

            return getattr(org_limiter, field_name, None) if org_limiter else None

        return cls(
            requests_per_minute=_resolve("requests_per_minute"),
            input_tokens_per_minute=_resolve("input_tokens_per_minute"),
            output_tokens_per_minute=_resolve("output_tokens_per_minute"),
        )


class LimitMixin(models.Model):
    """
    An abstract base model providing common rate limit fields.
    Set fields to None to indicate no specific limit at this level (use fallback).
    """

    requests_per_minute = models.PositiveIntegerField(
        null=True, blank=True, help_text="Maximum requests allowed per minute. Null means use fallback or no limit."
    )
    input_tokens_per_minute = models.PositiveIntegerField(
        null=True, blank=True, help_text="Maximum input tokens allowed per minute. Null means use fallback or no limit."
    )
    output_tokens_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum output tokens allowed per minute. Null means use fallback or no limit.",
    )

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created


class ModelExclusionMixin(models.Model):
    """
    An abstract base model providing a model exclusion list.
    Add Model names to the list to indicate which models should be excluded for the specific object.
    """

    excluded_models = JSONField(default=list, help_text="Models to exclude from the config.")

    merge_exclusion_lists = BooleanField(
        default=True,
        null=False,
        help_text="When enabled, this object's exclusion list will be combined with the exclusion "
        "list from its parent in the hierarchy (such as combining a User's and an Org's lists). "
        "Disable to use only this object's exclusions.",
    )

    def add_excluded_model(self, model_name: str):
        if model_name not in self.excluded_models:
            self.excluded_models.append(model_name)
            self.save(update_fields=["excluded_models"])

    def remove_excluded_model(self, model_name: str):
        if model_name in self.excluded_models:
            self.excluded_models.remove(model_name)
            self.save(update_fields=["excluded_models"])

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created


class MCPServerExclusionMixin(models.Model):
    """
    An abstract base model providing an MCP server exclusion list.
    Add MCP server names to the list to indicate which servers should be excluded for the specific object.
    """

    excluded_mcp_servers = JSONField(default=list, help_text="MCP servers to exclude from the config.")

    merge_mcp_server_exclusion_lists = BooleanField(
        default=True,
        null=False,
        help_text="When enabled, this object's MCP server exclusion list will be combined with the exclusion "
        "list from its parent in the hierarchy (such as combining a User's and an Org's lists). "
        "Disable to use only this object's exclusions.",
    )

    def add_excluded_mcp_server(self, server_name: str):
        if server_name not in self.excluded_mcp_servers:
            self.excluded_mcp_servers.append(server_name)
            self.save(update_fields=["excluded_mcp_servers"])

    def remove_excluded_mcp_server(self, server_name: str):
        if server_name in self.excluded_mcp_servers:
            self.excluded_mcp_servers.remove(server_name)
            self.save(update_fields=["excluded_mcp_servers"])

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created


class Org(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """Represents an Organization."""

    name = models.CharField(verbose_name="Org name", max_length=255, unique=True)

    def __str__(self):
        return self.name


class Team(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """Represents a Team within an Organization."""

    name = models.CharField(verbose_name="Team name", max_length=255)
    description = models.TextField(blank=True)

    org = models.ForeignKey(Org, on_delete=models.CASCADE, related_name="teams")

    class Meta:
        unique_together = ("name", "org")

    def __str__(self):
        return f"{self.name} ({self.org.name})"


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

    teams = models.ManyToManyField(Team, through="TeamMembership", related_name="member_profiles", blank=True)

    @property
    def group(self) -> Literal["admin", "org-admin", "user"]:
        g = self.user.groups
        if g.filter(name="admin").exists():
            return "admin"
        if g.filter(name="org-admin").exists():
            return "org-admin"
        if g.filter(name="user").exists():
            return "user"
        raise ValidationError("User has no group")

    @group.setter
    def group(self, group: Literal["admin", "org-admin", "user"]):
        if group not in ["admin", "org-admin", "user"]:
            raise ValueError(f"Group {group} does not exist!")

        # Clear existing groups first
        self.user.groups.clear()
        # Add the new group
        try:
            group_obj = Group.objects.get(name=group)
            self.user.groups.add(group_obj)
        except Group.DoesNotExist:
            # Handle case where the group doesn't exist in DB (shouldn't happen with check above)
            raise ObjectDoesNotExist(f"The group '{group}' does not exist in the database.") from None

    def clean(self):
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
            return self.group == "admin"
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
            user_group = self.group  # Use the property to get the group name
            if user_group == "admin":  # Use admin group independent of super-user status to quickly debug in Admin UI.
                return True
            if user_group == "org-admin":
                # Check if the profile's org matches the org being checked
                return self.org == org_to_check
        except ValidationError:  # Raised if user has no valid group
            return False  # Or handle as an error depending on requirements
        return False  # Default case (e.g., 'user' group)

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
        # Removed overly defensive AttributeError check. If 'teammembership_set' is missing,
        # it indicates a fundamental model setup error that should not be caught here.

    def __str__(self):
        return f"{self.user.email} (Profile - {self.org.name})"


class TeamMembership(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)

    # Your extra data about the relationship
    is_admin = models.BooleanField(default=False)
    date_added = models.DateField(auto_now_add=True)

    class Meta:
        # Ensure a user can only be in a team once
        unique_together = ("user_profile", "team")

    def __str__(self):
        return f"{self.user_profile} in {self.team}{' (Admin)' if self.is_admin else ''}"


class ServiceAccount(models.Model):
    """Represents a Service Account, typically associated with a Team."""

    name = models.CharField(verbose_name="Service Account name", max_length=255)
    description = models.TextField(blank=True)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="service_accounts")

    class Meta:
        unique_together = ("name", "team")

    def __str__(self):
        # Handle case where team might not be set yet
        return f"{self.name} (Team: {self.team.name if self.team_id else 'N/A'})"

    def clean(self):
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
        ServiceAccount,  # Removed quotes as ServiceAccount is defined above
        on_delete=models.CASCADE,
        related_name="token",
        null=True,
        blank=True,
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

    @property
    def is_expired(self):
        return self.expires_at is not None and self.expires_at <= timezone.now()

    # The clean method checking for self.user is implicitly handled by the ForeignKey
    # unless null=True is added to the user field, which doesn't seem intended here.

    def __str__(self):
        if self.service_account:
            return f"'{self.name}' ({self.service_account.name})"
        return f"'{self.name}'"

    @staticmethod
    def _generate_secret_key(prefix="sk-") -> str:
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

    # Removed static generate_key - logic moved to _generate_secret_key

    def _set_new_key(self) -> str:
        """
        Generates a new secret key, sets the instance's hash and preview fields.
        Returns the generated secret key. Does NOT save the instance.
        """
        secret_key = self._generate_secret_key()
        self.key_hash = self._hash_key(secret_key)
        self.key_preview = self._generate_preview(secret_key)
        return secret_key

    def save(self, *args, **kwargs):
        """
        Ensures key_hash and key_preview are set before the first save.
        """
        if not self.pk and (not self.key_hash or not self.key_preview):
            # This check ensures _set_new_key() was called before the first save.
            raise ValueError(
                "Token cannot be saved without key_hash and key_preview. Call _set_new_key() before saving."
            )
        super().save(*args, **kwargs)

    # Removed the key_preview @property as it's now a direct field

    def regenerate_key(self) -> str:
        """
        Generates a new secret key, updates the hash and preview, saves the instance,
        and returns the *new secret key*.
        """
        new_secret_key = self._set_new_key()  # Use the helper method
        self.save(update_fields=["key_hash", "key_preview"])  # Save the changes
        return new_secret_key  # Return the original new key

    def clean(self):
        """
        Model-level validation. The user-specific token limit is checked in the form.
        """
        # Removed the user token limit check here, as it's handled in TokenCreateForm.clean()
        # Ensure hash and preview are present before final validation/save if needed
        # (Handled by save() method)
        super().clean()

    def _get_from_hierarchy(self, retrieval_function: Callable):
        token_instance = Token.objects.select_related(
            "user__profile__org",  # Needed for User Tokens
            "service_account__team__org",  # Needed for Service Account Tokens
        ).get(pk=self.pk)  # Assumes the token instance exists

        # Determine the primary (specific) and fallback (org) limit sources
        if token_instance.service_account:
            # Path for Service Account Tokens
            team = token_instance.service_account.team  # Team holds specific SA limits
            org = team.org  # Team's Org holds fallback limits
            return retrieval_function(team, org)
        # Path for standard User Tokens
        profile = token_instance.user.profile  # UserProfile holds specific user limits
        org = profile.org  # Profile's Org holds fallback limits
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
        cls, specific_exclusion: Optional["ModelExclusionMixin"], org_exclusion: Optional["ModelExclusionMixin"]
    ) -> list[str]:
        exclusion_list: list[str] = specific_exclusion.excluded_models if specific_exclusion else []
        if specific_exclusion.merge_exclusion_lists:
            org_exclusion_list: list[str] = org_exclusion.excluded_models if org_exclusion else []
            exclusion_list = exclusion_list + org_exclusion_list
            if org_exclusion.merge_exclusion_lists:
                settings_exclusion_list: list[str] = getattr(settings, "AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST", [])
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
        cls, specific_exclusion: Optional["MCPServerExclusionMixin"], org_exclusion: Optional["MCPServerExclusionMixin"]
    ) -> list[str]:
        exclusion_list: list[str] = specific_exclusion.excluded_mcp_servers if specific_exclusion else []
        if specific_exclusion and specific_exclusion.merge_mcp_server_exclusion_lists:
            org_exclusion_list: list[str] = org_exclusion.excluded_mcp_servers if org_exclusion else []
            exclusion_list = exclusion_list + org_exclusion_list
            if org_exclusion and org_exclusion.merge_mcp_server_exclusion_lists:
                settings_exclusion_list: list[str] = getattr(settings, "AQUEDUCT_DEFAULT_MCP_SERVER_EXCLUSION_LIST", [])
                exclusion_list = exclusion_list + settings_exclusion_list

        return list(set(exclusion_list))

    def mcp_server_exclusion_list(self) -> list[str]:
        """
        Determines if an MCP server is excluded for this token, returning a list of excluded servers.
        Assumes database integrity for related objects.

        Hierarchy Rules:
        - Service Account Token: Uses the Team exclusion list, falls back to the Org exclusion list.
        - User Token: Uses the UserProfile exclusion list, falls back to the Org exclusion list.
        """
        return self._get_from_hierarchy(Token._mcp_server_exclusion_list_from_objects)

    def mcp_server_excluded(self, server_name: str):
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
            return cls.objects.select_related("user__profile__org", "service_account__team__org").get(
                key_hash=hashed_key
            )
        except cls.DoesNotExist:
            return None


@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other):
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens
        )

    def __sub__(self, other):
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens - other.input_tokens, output_tokens=self.output_tokens - other.output_tokens
        )


class Request(models.Model):
    """Represents a request made using a custom Token."""

    input_tokens = models.PositiveIntegerField(default=0, help_text="Tokens consumed by the input for this request")
    output_tokens = models.PositiveIntegerField(default=0, help_text="Tokens generated by the output for this request")

    @property
    def token_usage(self) -> Usage:
        """Get a TokenUsage dataclass instance representing this request's token usage."""
        return Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)

    @token_usage.setter
    def token_usage(self, usage: Usage):
        """Set input_tokens and output_tokens from a Usage dataclass instance."""
        if not isinstance(usage, Usage):
            raise TypeError("token_usage must be a Usage dataclass instance")
        self.input_tokens = usage.input_tokens
        self.output_tokens = usage.output_tokens

    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,  # If Token is deleted, delete its associated Requests
        related_name="requests",
    )
    # django: DJ001 - null=True on CharField may cause SQL NULL instead of empty string
    model = models.CharField(max_length=255, null=True, blank=True, help_text="Model used in request")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    # Additional fields (endpoint_url removed)
    method = models.CharField(max_length=16, blank=True, help_text="HTTP method used (e.g., GET, POST, etc.)")
    status_code = models.PositiveIntegerField(
        null=True, blank=True, help_text="HTTP status code returned by the endpoint"
    )
    response_time_ms = models.PositiveIntegerField(
        null=True, blank=True, help_text="Time taken to respond, in milliseconds"
    )
    user_agent = models.CharField(
        max_length=256, blank=True, help_text="User agent string of the client making the request"
    )
    ip_address = models.GenericIPAddressField(
        null=True, blank=True, help_text="IP address from which the request originated"
    )
    path = models.CharField(
        max_length=512,
        blank=True,
        help_text="The specific API path requested after the base endpoint URL (e.g., '/chat/completions')",
    )
    user_id = models.CharField(
        max_length=256,
        blank=True,
        db_index=True,
        help_text="A string (user ID, email address, username, etc.) identifying "
        "the particular user who made the request, "
        "and who may be a different user than the token owner.",
    )

    class Meta:
        # Crucial for the rate limit query!
        indexes: ClassVar[list] = [
            models.Index(fields=["token", "timestamp"]),
            models.Index(fields=["model", "timestamp"]),
        ]
        ordering: ClassVar[list] = ["-timestamp"]  # Optional: default ordering

    def __str__(self):
        return f"{self.id}"


class FileObject(models.Model):
    """
    Mirrors the structure of OpenAI's FileObject type, excluding deprecated fields.
    """

    id = models.CharField(
        max_length=100,
        primary_key=True,
        editable=False,
        help_text="The file identifier, which can be referenced in the API endpoints.",
    )
    bytes = models.BigIntegerField(help_text="The size of the file, in bytes.")
    created_at = models.PositiveIntegerField(help_text="The Unix timestamp (in seconds) for when the file was created.")
    filename = models.CharField(max_length=255, help_text="The name of the file.")
    PURPOSE_CHOICES: ClassVar[list] = [
        ("assistants", "assistants"),
        ("assistants_output", "assistants_output"),
        ("batch", "batch"),
        ("batch_output", "batch_output"),
        ("fine-tune", "fine-tune"),
        ("vision", "vision"),
        ("user_data", "user_data"),
        ("evals", "evals"),
    ]
    purpose = models.CharField(max_length=20, choices=PURPOSE_CHOICES, help_text="The intended purpose of the file.")
    expires_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the file will expire."
    )

    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,  # If Token is deleted, delete its associated Files
        related_name="files",
    )

    preview = models.TextField(
        blank=True, default="", help_text="Preview of file content (first 10 lines for JSONL files)"
    )

    upstream_url = models.URLField(blank=True, default="", help_text="The upstream API URL this file was uploaded to")

    class Meta:
        verbose_name = "File Object"
        verbose_name_plural = "File Objects"

    @property
    def model(self) -> openai.types.FileObject:
        return openai.types.FileObject(
            id=self.id,
            bytes=self.bytes,
            created_at=self.created_at,
            filename=self.filename,
            purpose=self.purpose,
            expires_at=self.expires_at,
            object="file",
            status="processed",
        )

    async def adelete_upstream(self, client=None, raise_on_error=True) -> bool:
        """
        Delete the file from the upstream API.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns False.

        Returns:
            True on success.
            False on failure (only if raise_on_error=False).
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            await client.files.delete(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to delete file {self.id} from upstream: {e}")
            return False
        else:
            return True

    async def areload_from_upstream(self, client=None, raise_on_error=True) -> openai.types.FileObject | None:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.files.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to reload file {self.id} from upstream: {e}")
            return None
        else:
            self.purpose = remote.purpose
            self.expires_at = remote.expires_at
            await self.asave()
            return remote

    def delete(self, using=None, keep_parents=False, delete_upstream=False):
        """
        Override ORM delete - files are stored upstream, so we only delete the local DB record.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            import asyncio

            asyncio.run(self.adelete_upstream())
        super().delete(using=using, keep_parents=keep_parents)

    def __str__(self):
        return self.id


def default_request_counts() -> dict[str, int]:
    return {"total": 0, "completed": 0, "failed": 0}


class BatchStatus(models.TextChoices):
    VALIDATING = "validating", "validating"
    FAILED = "failed", "failed"
    IN_PROGRESS = "in_progress", "in_progress"
    FINALIZING = "finalizing", "finalizing"
    COMPLETED = "completed", "completed"
    EXPIRED = "expired", "expired"
    CANCELLING = "cancelling", "cancelling"
    CANCELLED = "cancelled", "cancelled"


class Batch(models.Model):
    """
    Mirrors the structure of OpenAI's Batch type.
    """

    id = models.CharField(max_length=100, primary_key=True, editable=False, help_text="The batch identifier.")
    completion_window = models.CharField(
        max_length=100, help_text="The time frame within which the batch should be processed."
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the batch was created."
    )
    endpoint = models.CharField(max_length=255, help_text="The OpenAI API endpoint used by the batch.")
    input_file = models.ForeignKey(
        FileObject, on_delete=models.CASCADE, related_name="batches", help_text="The input file for the batch."
    )
    output_file = models.ForeignKey(
        FileObject,
        on_delete=models.SET_NULL,
        related_name="output_batches",
        null=True,
        blank=True,
        help_text="The output file for the batch (set when batch completes).",
    )
    error_file = models.ForeignKey(
        FileObject,
        on_delete=models.SET_NULL,
        related_name="error_batches",
        null=True,
        blank=True,
        help_text="The error file for the batch (set when batch completes with errors).",
    )
    status = models.CharField(max_length=20, choices=BatchStatus.choices, help_text="The current status of the batch.")
    cancelled_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch was cancelled."
    )
    cancelling_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch started cancelling."
    )
    completed_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch was completed."
    )
    expired_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch expired."
    )
    expires_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch will expire."
    )
    failed_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch failed."
    )
    finalizing_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch started finalizing."
    )
    in_progress_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the batch started processing."
    )
    metadata = JSONField(null=True, blank=True, help_text="Metadata attached to the batch.")
    # {"input": 0, "total": 0, "completed": 0, "failed": 0 }
    request_counts = JSONField(
        default=default_request_counts,
        null=True,
        blank=True,
        help_text="The request counts for different statuses within the batch.",
    )

    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,
        related_name="batches",
        null=True,
        blank=True,
        help_text="The token (API key) that created this batch.",
    )

    class Meta:
        verbose_name = "Batch"
        verbose_name_plural = "Batches"

    def __str__(self) -> str:
        return self.id

    @property
    def model(self) -> openai.types.batch.Batch:
        return openai.types.batch.Batch(
            id=self.id,
            completion_window=self.completion_window,
            created_at=self.created_at,
            endpoint=self.endpoint,
            input_file_id=self.input_file_id,
            object="batch",
            status=self.status,
            cancelled_at=self.cancelled_at,
            cancelling_at=self.cancelling_at,
            completed_at=self.completed_at,
            error_file_id=self.error_file_id if self.error_file else None,
            errors=None,
            expired_at=self.expired_at,
            expires_at=self.expires_at,
            failed_at=self.failed_at,
            finalizing_at=self.finalizing_at,
            in_progress_at=self.in_progress_at,
            metadata=self.metadata,
            output_file_id=self.output_file_id if self.output_file else None,
            request_counts=openai.types.BatchRequestCounts(
                total=self.request_counts.get("total", 0),
                completed=self.request_counts.get("completed", 0),
                failed=self.request_counts.get("failed", 0),
            ),
        )

    async def areload_from_upstream(self, client=None, raise_on_error=True) -> openai.types.Batch | None:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.batches.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to reload batch {self.id} from upstream: {e}")
            return None
        else:
            self.status = remote.status
            if remote.request_counts:
                self.request_counts = remote.request_counts.model_dump()
            self.completed_at = remote.completed_at
            self.failed_at = remote.failed_at
            self.cancelled_at = remote.cancelled_at
            self.cancelling_at = remote.cancelling_at
            self.expired_at = remote.expired_at
            self.in_progress_at = remote.in_progress_at
            self.finalizing_at = remote.finalizing_at
            await self.asave()
            return remote

    def delete(self, using=None, keep_parents=False):
        """
        Override delete - files are stored upstream, so we only delete the local DB record.
        """
        super().delete(using=using, keep_parents=keep_parents)


class VectorStoreStatus(models.TextChoices):
    """Vector store status choices matching OpenAI's API."""

    EXPIRED = "expired", "expired"
    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"


class VectorStore(models.Model):
    """
    Mirrors the structure of OpenAI's VectorStore type.
    Stores metadata for vector stores with upstream relay.
    """

    id = models.CharField(max_length=100, primary_key=True, editable=False, help_text="The vector store identifier.")
    token = models.ForeignKey(Token, on_delete=models.CASCADE, related_name="vector_stores")
    name = models.CharField(max_length=255, help_text="The name of the vector store.")
    expires_after = models.JSONField(null=True, blank=True, help_text="Expiration policy for the vector store.")
    chunking_strategy = models.JSONField(
        null=True, blank=True, help_text="Chunking configuration for the vector store."
    )
    metadata = models.JSONField(null=True, blank=True, help_text="Custom metadata for the vector store.")
    status = models.CharField(
        max_length=20,
        choices=VectorStoreStatus.choices,
        default=VectorStoreStatus.IN_PROGRESS,
        help_text="The current status of the vector store.",
    )
    usage_bytes = models.BigIntegerField(default=0, help_text="The total number of bytes used by the vector store.")
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the vector store was created."
    )
    last_active_at = models.PositiveIntegerField(
        null=True, blank=True, help_text="The Unix timestamp (in seconds) for when the vector store was last active."
    )
    upstream_url = models.URLField(
        blank=True, default="", help_text="The upstream API URL this vector store was created on"
    )

    class Meta:
        verbose_name = "Vector Store"
        verbose_name_plural = "Vector Stores"

    async def adelete_upstream(self, client=None, raise_on_error=True) -> bool:
        """
        Delete the vector store from the upstream API.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns False.

        Returns:
            True on success.
            False on failure (only if raise_on_error=False).
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            await client.vector_stores.delete(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to delete vector store {self.id} from upstream: {e}")
            return False
        else:
            return True

    async def areload_from_upstream(self, client=None, raise_on_error=True) -> openai.types.VectorStore | None:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to reload vector store {self.id} from upstream: {e}")
            return None
        else:
            self.status = remote.status or self.status
            self.usage_bytes = getattr(remote, "usage_bytes", self.usage_bytes)
            self.last_active_at = int(timezone.now().timestamp())
            await self.asave()
            await self.async_file_statuses(client)
            return remote

    async def async_file_statuses(self, client=None) -> tuple[int, int]:
        """
        Sync all VectorStoreFile statuses from upstream by listing files.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote_files_response = await client.vector_stores.files.list(vector_store_id=self.id)
            remote_files = remote_files_response.data if hasattr(remote_files_response, "data") else []
        except Exception as e:
            log.warning(f"Failed to list files for vector store {self.id} from upstream: {e}")
            return 0, 0

        success = 0
        failed = 0

        for remote_file in remote_files:
            try:
                local_file = await VectorStoreFile.objects.aget(id=remote_file.id, vector_store=self)
                local_file.status = remote_file.status or local_file.status
                local_file.usage_bytes = remote_file.usage_bytes
                if hasattr(remote_file, "last_error") and remote_file.last_error:
                    local_file.last_error = remote_file.last_error
                await local_file.asave()
                success += 1
            except VectorStoreFile.DoesNotExist:
                log.debug(f"VectorStoreFile {remote_file.id} not found locally, skipping sync")
            except Exception as e:
                log.warning(f"Failed to sync VectorStoreFile {remote_file.id}: {e}")
                failed += 1

        return success, failed

    def __str__(self) -> str:
        return self.id

    def delete(self, using=None, keep_parents=False, delete_upstream=False):
        """
        Override ORM delete - vector stores are stored upstream, so we only delete the local DB record.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            import asyncio

            asyncio.run(self.adelete_upstream())
        super().delete(using=using, keep_parents=keep_parents)


class VectorStoreFileStatus(models.TextChoices):
    """Vector store file processing status choices."""

    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"
    FAILED = "failed", "failed"
    CANCELLED = "cancelled", "cancelled"


class VectorStoreFile(models.Model):
    """
    Represents a file within a vector store.
    This is NOT the same as FileObject - it's a join table that tracks
    a FileObject's association with a specific VectorStore.
    """

    id = models.CharField(
        max_length=100, primary_key=True, editable=False, help_text="The vector store file identifier."
    )
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="files")
    file_obj = models.ForeignKey(
        FileObject,
        on_delete=models.CASCADE,
        related_name="vector_store_files",
        help_text="Reference to the actual file object.",
    )
    batch = models.ForeignKey(
        "VectorStoreFileBatch",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="files",
        help_text="The batch that created this file, if any.",
    )
    status = models.CharField(
        max_length=20,
        choices=VectorStoreFileStatus.choices,
        default=VectorStoreFileStatus.IN_PROGRESS,
        help_text="Processing status of the file in the vector store.",
    )
    last_error = models.JSONField(null=True, blank=True, help_text="Error details if processing failed.")
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the file was added to the vector store."
    )
    usage_bytes = models.BigIntegerField(
        default=0, help_text="Bytes used in vector store (may differ from original file after chunking)."
    )

    async def adelete_upstream(self, client=None, raise_on_error=True) -> bool:
        """
        Delete the vector store file from the upstream API.

        Note: Callers should use select_related("vector_store") when fetching
        VectorStoreFile instances to avoid extra queries.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns False.

        Returns:
            True on success.
            False on failure (only if raise_on_error=False).
        """
        if not self.vector_store_id:
            log.warning(f"Cannot delete vector store file {self.id}: vector_store not set")
            if raise_on_error:
                raise ValueError("Vector store not loaded")
            return False

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            await client.vector_stores.files.delete(vector_store_id=self.vector_store_id, file_id=self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to delete vector store file {self.id} from upstream: {e}")
            return False
        else:
            return True

    async def areload_from_upstream(
        self, client=None, raise_on_error=True
    ) -> Optional["openai.types.vector_stores.VectorStoreFile"]:
        """
        Fetch current state from upstream and update local DB fields.

        Note: Callers should use select_related("vector_store") when fetching
        VectorStoreFile instances to avoid extra queries.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        if not self.vector_store_id:
            if raise_on_error:
                raise ValueError("Vector store not set")
            log.warning(f"Cannot reload vector store file {self.id}: vector_store not set")
            return None

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.files.retrieve(vector_store_id=self.vector_store_id, file_id=self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to reload vector store file {self.id} from upstream: {e}")
            return None
        else:
            self.status = remote.status or self.status
            self.usage_bytes = remote.usage_bytes
            if hasattr(remote, "last_error") and remote.last_error:
                self.last_error = remote.last_error
            await self.asave()
            return remote

    def delete(self, using=None, keep_parents=False, delete_upstream=False):
        """
        Override ORM delete.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            import asyncio

            asyncio.run(self.adelete_upstream())
        super().delete(using=using, keep_parents=keep_parents)

    class Meta:
        verbose_name = "Vector Store File"
        verbose_name_plural = "Vector Store Files"

    def __str__(self) -> str:
        return self.id


class VectorStoreFileBatchStatus(models.TextChoices):
    """Vector store file batch status choices."""

    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"
    FAILED = "failed", "failed"
    CANCELLED = "cancelled", "cancelled"


class VectorStoreFileBatch(models.Model):
    """
    Represents a batch operation for adding files to a vector store.
    """

    id = models.CharField(
        max_length=100, primary_key=True, editable=False, help_text="The vector store file batch identifier."
    )
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="file_batches")
    file_counts = models.JSONField(
        default=dict, help_text="Processing counts (completed/failed/total/cancelled/in_progress)."
    )
    status = models.CharField(
        max_length=20,
        choices=VectorStoreFileBatchStatus.choices,
        default=VectorStoreFileBatchStatus.IN_PROGRESS,
        help_text="The current status of the batch.",
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the batch was created."
    )

    async def areload_from_upstream(
        self, client=None, raise_on_error=True
    ) -> Optional["openai.types.vector_stores.VectorStoreFileBatch"]:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        # Ensure vector_store is loaded
        if not self.vector_store_id:
            if raise_on_error:
                raise ValueError("Vector store not set")
            log.warning(f"Cannot reload vector store file batch {self.id}: vector_store not set")
            return None

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.file_batches.retrieve(
                vector_store_id=self.vector_store_id, batch_id=self.id
            )
        except Exception as e:
            if raise_on_error:
                raise
            log.warning(f"Failed to reload vector store file batch {self.id} from upstream: {e}")
            return None
        else:
            self.status = remote.status or self.status
            if hasattr(remote, "file_counts") and remote.file_counts:
                self.file_counts = remote.file_counts.model_dump()
            await self.asave()
            return remote

    def __str__(self) -> str:
        return self.id

    class Meta:
        verbose_name = "Vector Store File Batch"
        verbose_name_plural = "Vector Store File Batches"
