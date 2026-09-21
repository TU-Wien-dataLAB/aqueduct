import hashlib
import secrets
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from django.conf import settings
from django.db import models
from django.utils import timezone

from gateway.config import resolve_model_alias
from management.models.mixins import LimitSet, MCPServerExclusionMixin, ModelExclusionMixin
from management.models.service_account import ServiceAccount

_T = TypeVar("_T")


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
