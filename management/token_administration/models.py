# models.py
import secrets
from typing import Literal

from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
from django.contrib.auth.models import Group


class Org(models.Model):
    """Represents an Organization."""
    name = models.CharField(max_length=255, unique=True)
    orig_name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Team(models.Model):
    """Represents a Team within an Organization."""
    name = models.CharField(max_length=255)
    org = models.ForeignKey(
        Org,
        on_delete=models.CASCADE,
        related_name='teams'
    )

    class Meta:
        unique_together = ('name', 'org')

    def __str__(self):
        return f"{self.name} ({self.org.name})"


# Removed the custom User model

class UserProfile(models.Model):
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
    # Org relationship moved here
    org = models.ForeignKey(
        Org,
        on_delete=models.PROTECT,  # Keep PROTECT if you don't want to delete Org if profiles exist
        related_name='user_profiles'  # Changed related_name
    )
    # Teams relationship moved here
    teams = models.ManyToManyField(
        Team,
        related_name='member_profiles',  # Changed related_name for clarity
        blank=True
    )

    @property
    def group(self) -> Literal['admin', 'org-admin', 'team-admin', 'user']:
        g = self.user.groups
        if g.filter(name='admin').exists():
            return 'admin'
        elif g.filter(name='org-admin').exists():
            return 'org-admin'
        elif g.filter(name='team-admin').exists():
            return 'team-admin'
        elif g.filter(name='user').exists():
            return 'user'
        else:
            raise ValidationError('User has no group')

    @group.setter
    def group(self, group: Literal['admin', 'org-admin', 'team-admin', 'user']):
        print("setting group")
        if group not in ['admin', 'org-admin', 'team-admin', 'user']:
            raise ValueError(f'Group {group} does not exist!')

        try:
            g = self.group
            self.user.group.remove(g)
        except ValidationError:
            pass
        self.user.groups.add(Group.objects.get(name=group))

    def clean(self):
        """
        Validation moved from the old User model.
        Ensures assigned teams belong to the user's profile organization.
        Note: Runs on save() via forms/admin, but not on direct M2M .add()/.set() unless explicitly called.
        """
        super().clean()
        # Check only if org is set (it should be, unless profile is being created incomplete)
        if hasattr(self, 'org') and self.org is not None:
            # Check teams only if the M2M relationship has been saved (i.e., self.pk exists)
            # or if being added through a form where teams might be present before saving the profile itself.
            # Using try-except is safer if teams manager isn't available yet.
            try:
                teams_to_check = self.teams.all()
            except ValueError:  # Happens if UserProfile instance isn't saved yet
                teams_to_check = []

            for team in teams_to_check:
                if team.org != self.org:
                    raise ValidationError(
                        f"Team '{team.name}' (Org: {team.org.name}) does not belong to the profile's organization '{self.org.name}'."
                    )

    def __str__(self):
        return f"{self.user.email} ({self.org.name})"


class ServiceAccount(models.Model):
    """Represents a Service Account, typically associated with a Team."""
    name = models.CharField(max_length=255)
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='service_accounts'
    )

    class Meta:
        unique_together = ('name', 'team')

    def __str__(self):
        return f"{self.name} (Team: {self.team.name})"


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
        on_delete=models.SET_NULL,  # Allows Token to exist if SA is deleted, links SA creation to token? Review this.
        related_name='token',
        null=True,
        blank=True
    )
    # You likely need a field for the actual token key/secret
    key = models.CharField(max_length=128, unique=True, editable=False)  # Example field
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
    def generate_key() -> str:
        """Generates a unique token key."""
        return secrets.token_urlsafe(nbytes=32)

    def save(self, *args, **kwargs):
        """Generates the 'key' on save if it doesn't exist."""
        if not self.key:
            self.key = self.generate_key()
        super().save(*args, **kwargs)

    def clean(self):
        """
        Validates that a user can only generate a configurable number of tokens
        per user (tokens not associated with a service account).
        """
        if not self.service_account:
            max_tokens = getattr(settings, 'MAX_USER_TOKENS', 3)
            token_count = Token.objects.filter(user=self.user, service_account__isnull=True).count()
            if self.pk is None:  # Check if the token is being created
                if token_count >= max_tokens:
                    raise ValidationError(
                        f"Users can only have a maximum of {max_tokens} tokens not associated with a service account."
                    )
            else:  # If the token is being updated, exclude the current token from the count
                existing_tokens = Token.objects.filter(user=self.user, service_account__isnull=True).exclude(pk=self.pk)
                if existing_tokens.count() >= max_tokens:
                    raise ValidationError(
                        f"Users can only have a maximum of {max_tokens} tokens not associated with a service account."
                    )
        super().clean()


class Request(models.Model):
    """Represents a request made using a custom Token."""
    # This model seems tied to your custom Token model.
    token_usage = models.IntegerField(default=0)  # What does this track? Count per request or total on token?
    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,  # If Token is deleted, delete its associated Requests
        related_name='requests'
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    # TODO: Add other relevant fields: endpoint_url, method, status_code, response_time_ms, request_data, response_data etc.

    def __str__(self):
        return f"{self.id}"


class Endpoint(models.Model):
    """Represents an API endpoint, likely serving multiple Models."""
    name = models.CharField(max_length=255, unique=True)  # Added unique=True

    # TODO: Add more fields like path, description, etc.

    def __str__(self):
        model_count = self.models.count()
        return f"{self.name} ({model_count} model{'s' if model_count != 1 else ''})"


class Model(models.Model):
    """Represents a computational model (e.g., ML model)."""
    name = models.CharField(max_length=255, unique=True)
    endpoint = models.ForeignKey(
        Endpoint,
        on_delete=models.CASCADE,  # If endpoint is deleted, delete associated models
        related_name='models'
    )

    # TODO: Add more fields like version, description, file_path, etc.

    def __str__(self):
        return f"{self.name} ({self.endpoint.name})"
