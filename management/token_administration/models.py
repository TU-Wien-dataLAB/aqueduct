# models.py
import dataclasses
import os
import secrets
import hashlib
from typing import Literal, Optional

from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.contrib.auth.models import Group


@dataclasses.dataclass(frozen=True)  # frozen=True makes instances immutable
class LimitSet:
    """Represents a resolved set of rate limits."""
    requests_per_minute: Optional[int] = None
    input_tokens_per_minute: Optional[int] = None
    output_tokens_per_minute: Optional[int] = None

    # Add future limit fields here with default None

    @classmethod
    def from_objects(cls, specific_limiter: Optional['LimitMixin'], org_limiter: Optional['LimitMixin']) -> 'LimitSet':
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
        def _resolve(field_name: str) -> Optional[int]:
            # Get value from the specific level first
            # Use getattr for safe access, defaulting to None if field absent
            specific_value = getattr(specific_limiter, field_name, None) if specific_limiter else None
            if specific_value is not None:
                # Return immediately if a specific limit is set
                return specific_value

            return getattr(org_limiter, field_name, None) if org_limiter else None

        return cls(
            requests_per_minute=_resolve('requests_per_minute'),
            input_tokens_per_minute=_resolve('input_tokens_per_minute'),
            output_tokens_per_minute=_resolve('output_tokens_per_minute'),
        )


class LimitMixin(models.Model):
    """
    An abstract base model providing common rate limit fields.
    Set fields to None to indicate no specific limit at this level (use fallback).
    """
    requests_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum requests allowed per minute. Null means use fallback or no limit."
    )
    input_tokens_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum input tokens allowed per minute. Null means use fallback or no limit."
    )
    output_tokens_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum output tokens allowed per minute. Null means use fallback or no limit."
    )

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created


class Org(LimitMixin, models.Model):
    """Represents an Organization."""
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Team(LimitMixin, models.Model):
    """Represents a Team within an Organization."""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    org = models.ForeignKey(
        Org,
        on_delete=models.CASCADE,
        related_name='teams'
    )

    class Meta:
        unique_together = ('name', 'org')

    def __str__(self):
        return f"{self.name} ({self.org.name})"


class UserProfile(LimitMixin, models.Model):
    """
    Holds additional information related to the built-in Django User model.
    Each Django User should have one corresponding UserProfile.
    """
    # Link to the standard Django User model
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,  # Use settings.AUTH_USER_MODEL
        on_delete=models.CASCADE,
        related_name='profile'  # Access profile from user: user.profile
    )

    org = models.ForeignKey(
        Org,
        on_delete=models.PROTECT,  # Keep PROTECT if you don't want to delete Org if profiles exist
        related_name='user_profiles'  # Changed related_name
    )

    teams = models.ManyToManyField(
        Team,
        through='TeamMembership',
        related_name='member_profiles',  # Changed related_name for clarity
        blank=True
    )

    @property
    def group(self) -> Literal['admin', 'org-admin', 'user']:
        g = self.user.groups
        if g.filter(name='admin').exists():
            return 'admin'
        elif g.filter(name='org-admin').exists():
            return 'org-admin'
        elif g.filter(name='user').exists():
            return 'user'
        else:
            raise ValidationError('User has no group')

    @group.setter
    def group(self, group: Literal['admin', 'org-admin', 'user']):
        if group not in ['admin', 'org-admin', 'user']:
            raise ValueError(f'Group {group} does not exist!')

        # Clear existing groups first
        self.user.groups.clear()
        # Add the new group
        try:
            group_obj = Group.objects.get(name=group)
            self.user.groups.add(group_obj)
        except Group.DoesNotExist:
            # Handle case where the group doesn't exist in DB (shouldn't happen with check above)
            raise ObjectDoesNotExist(f"The group '{group}' does not exist in the database.")

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
                        f"Team '{team.name}' (Org: {team.org.name}) does not belong to the profile's organization '{self.org.name}'."
                    )
            # No need for the try/except ValueError, as self.pk ensures the instance is saved.

    def is_admin(self) -> bool:
        """Checks if the user has the global 'admin' group."""
        try:
            return self.group == 'admin'
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
            if user_group == 'admin':  # Use admin group independent of super-user status to quickly debug in Admin UI.
                return True
            if user_group == 'org-admin':
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
            # Assumes TeamMembership model exists and is related via user_profile
            # Django ensures 'teammembership_set' exists if TeamMembership has a ForeignKey to UserProfile.
            membership = self.teammembership_set.get(team=team_to_check)
            return membership.is_admin
        except ObjectDoesNotExist:
            # No specific membership record found for this user and team
            return False
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
        unique_together = ('user_profile', 'team')

    def __str__(self):
        return f"{self.user_profile} in {self.team}{' (Admin)' if self.is_admin else ''}"


class ServiceAccount(models.Model):
    """Represents a Service Account, typically associated with a Team."""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='service_accounts'
    )

    class Meta:
        unique_together = ('name', 'team')

    def __str__(self):
        return f"{self.name} (Team: {self.team.name if self.team_id else 'N/A'})"  # Handle case where team might not be set yet

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
                limit = getattr(settings, 'MAX_SERVICE_ACCOUNTS_PER_TEAM', 10)

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
                    raise ValidationError({
                        # You can attach the error to a specific field or make it non-field
                        # None: f"Team '{team_instance.name}' has reached the maximum limit of {limit} service accounts."
                        'team': f"Team '{team_instance.name}' has reached the maximum limit of {limit} service accounts."
                        # Attach to team conceptually
                    })

            except Team.DoesNotExist:
                # This case shouldn't happen if ForeignKey validation runs,
                # but good to handle defensively.
                raise ValidationError("Associated team does not exist.")


class Token(models.Model):
    """
    Represents an authentication token (e.g., API Key), associated with a User
    and optionally a Service Account.
    """
    name = models.CharField(max_length=255, null=False)
    # Link to the standard Django User model
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # Use settings.AUTH_USER_MODEL
        on_delete=models.CASCADE,
        related_name='custom_auth_tokens',  # Changed related_name to avoid potential clashes
    )
    # This structure implies the Token is *created by* a User, potentially *for* a Service Account.
    service_account = models.OneToOneField(
        ServiceAccount,  # Removed quotes as ServiceAccount is defined above
        on_delete=models.CASCADE,
        related_name='token',
        null=True,
        blank=True
    )
    # Store hash and preview, not the original key
    key_hash = models.CharField(
        max_length=64,  # SHA-256 hash length
        unique=True,
        editable=False,
        help_text="SHA-256 hash of the token key."
    )
    key_preview = models.CharField(
        max_length=12,  # e.g., "T0K3..."
        editable=False,
        help_text="First few characters of the original token key for display."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)  # Optional expiry

    # The clean method checking for self.user is implicitly handled by the ForeignKey
    # unless null=True is added to the user field, which doesn't seem intended here.

    def __str__(self):
        if self.service_account:
            return f"'{self.name}' ({self.service_account.name})"
        else:
            return f"'{self.name}'"

    @staticmethod
    def _generate_secret_key(prefix="sk-") -> str:  # Renamed for clarity
        """Generates a unique secret token key."""
        # Consider prefixing keys, e.g., "aqt_" for Aqueduct Token
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
                "Token cannot be saved without key_hash and key_preview. Call _set_new_key() before saving.")
        super().save(*args, **kwargs)

    # Removed the key_preview @property as it's now a direct field

    def regenerate_key(self) -> str:
        """
        Generates a new secret key, updates the hash and preview, saves the instance,
        and returns the *new secret key*.
        """
        new_secret_key = self._set_new_key()  # Use the helper method
        self.save(update_fields=['key_hash', 'key_preview'])  # Save the changes
        return new_secret_key  # Return the original new key

    def clean(self):
        """
        Model-level validation. The user-specific token limit is checked in the form.
        """
        # Removed the user token limit check here, as it's handled in TokenCreateForm.clean()
        # Ensure hash and preview are present before final validation/save if needed
        # (Handled by save() method)
        super().clean()

    def get_limit(self) -> 'LimitSet':
        """
        Determines the effective rate limits for this token, returning a LimitSet dataclass.
        Assumes database integrity for related objects.

        Hierarchy Rules:
        - Service Account Token: Uses Team limits, falls back to Org limits.
        - User Token: Uses UserProfile limits, falls back to Org limits.
        """
        # Fetch the current token instance with potentially needed related objects
        # Using select_related is efficient as it avoids subsequent DB queries
        token_instance = Token.objects.select_related(
            'user__profile__org',  # Needed for User Tokens
            'service_account__team__org'  # Needed for Service Account Tokens
        ).get(pk=self.pk)  # Assumes the token instance exists

        # Determine the primary (specific) and fallback (org) limit sources
        if token_instance.service_account:
            # Path for Service Account Tokens
            team = token_instance.service_account.team  # Team holds specific SA limits
            org = team.org  # Team's Org holds fallback limits
            return LimitSet.from_objects(team, org)
        else:
            # Path for standard User Tokens
            profile = token_instance.user.profile  # UserProfile holds specific user limits
            org = profile.org  # Profile's Org holds fallback limits
            return LimitSet.from_objects(profile, org)

    @classmethod
    def find_by_key(cls, key_value: str) -> Optional['Token']:
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
                'user__profile__org',
                'service_account__team__org'
            ).get(key_hash=hashed_key)
        except cls.DoesNotExist:
            return None


class Request(models.Model):
    """Represents a request made using a custom Token."""
    input_tokens = models.PositiveIntegerField(default=0, help_text="Tokens consumed by the input for this request")
    output_tokens = models.PositiveIntegerField(default=0, help_text="Tokens generated by the output for this request")
    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,  # If Token is deleted, delete its associated Requests
        related_name='requests'
    )
    model = models.ForeignKey(
        'Model',
        on_delete=models.CASCADE,
        related_name='requests'
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    # Additional fields (endpoint_url removed)
    method = models.CharField(
        max_length=16,
        blank=True,
        help_text="HTTP method used (e.g., GET, POST, etc.)"
    )
    status_code = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="HTTP status code returned by the endpoint"
    )
    response_time_ms = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Time taken to respond, in milliseconds"
    )
    user_agent = models.CharField(
        max_length=256,
        blank=True,
        help_text="User agent string of the client making the request"
    )
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="IP address from which the request originated"
    )

    def __str__(self):
        return f"{self.id}"


class Endpoint(models.Model):
    """Represents an API endpoint, likely serving multiple Models."""
    name = models.CharField(max_length=255, unique=True)
    url = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    access_token = models.CharField(
        max_length=128,
        unique=False,
        help_text="API access token required to access this endpoint. Can be the literal token, prefixed with 'os.environ/' to read from environment variables, or prefixed with 'settings/' to read from Django settings (e.g., 'settings/MY_ENDPOINT_TOKEN')."
    )

    def __str__(self):
        model_count = self.models.count()
        return f"{self.name} ({model_count} model{'s' if model_count != 1 else ''})"

    def get_access_token(self) -> Optional[str]:
        if self.access_token.startswith("os.environ/"):
            return os.environ.get(self.access_token.lstrip("os.environ/"))
        elif self.access_token.startswith("settings/"):
            setting_key = self.access_token.lstrip("settings/")
            return getattr(settings, setting_key, None)  # Return None if setting not found
        else:
            return self.access_token


class Model(models.Model):
    """Represents a LLM model."""
    name = models.CharField(max_length=255, unique=True)
    display_name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    endpoint = models.ForeignKey(
        Endpoint,
        on_delete=models.CASCADE,  # If endpoint is deleted, delete associated models
        related_name='models'
    )

    class Meta:
        unique_together = ('name', 'endpoint')

    def __str__(self):
        return f"{self.name} ({self.endpoint.name})"
