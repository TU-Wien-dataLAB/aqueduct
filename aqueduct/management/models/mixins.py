import dataclasses
from typing import Optional

from django.db import models
from django.db.models import BooleanField, JSONField


@dataclasses.dataclass(frozen=True)  # frozen=True makes instances immutable
class LimitSet:
    """Represents a resolved set of rate limits."""

    requests_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None

    # Add future limit fields here with default None

    @classmethod
    def from_objects(
        cls, specific_limiter: Optional["LimitMixin"], org_limiter: Optional["LimitMixin"]
    ) -> "LimitSet":
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
            specific_value: int | None = (
                getattr(specific_limiter, field_name, None) if specific_limiter else None
            )
            if specific_value is not None:
                return specific_value

            org_value: int | None = getattr(org_limiter, field_name, None) if org_limiter else None
            return org_value

        return cls(
            requests_per_minute=_resolve("requests_per_minute"),
            input_tokens_per_minute=_resolve("input_tokens_per_minute"),
            output_tokens_per_minute=_resolve("output_tokens_per_minute"),
        )


class LimitMixin(models.Model):
    """
    An abstract base model providing common rate limit fields.
    Set fields to `None` to indicate no specific limit at this level (use fallback).
    """

    requests_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum requests allowed per minute. Null means use fallback or no limit.",
    )
    input_tokens_per_minute = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum input tokens allowed per minute. Null means use fallback or no limit.",
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

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created

    def add_excluded_model(self, model_name: str) -> None:
        if model_name not in self.excluded_models:
            self.excluded_models.append(model_name)
            self.save(update_fields=["excluded_models"])

    def remove_excluded_model(self, model_name: str) -> None:
        if model_name in self.excluded_models:
            self.excluded_models.remove(model_name)
            self.save(update_fields=["excluded_models"])


class MCPServerExclusionMixin(models.Model):
    """
    An abstract base model providing an MCP server exclusion list.
    Add MCP server names to the list to indicate which servers should be excluded.
    """

    excluded_mcp_servers = JSONField(
        default=list, help_text="MCP servers to exclude from the config."
    )

    merge_mcp_server_exclusion_lists = BooleanField(
        default=True,
        null=False,
        help_text="When enabled, this object's MCP server exclusion list will be combined "
        "with the exclusion list from its parent in the hierarchy "
        "(such as combining a User's and an Org's lists). "
        "Disable to use only this object's exclusions.",
    )

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created

    def add_excluded_mcp_server(self, server_name: str) -> None:
        if server_name not in self.excluded_mcp_servers:
            self.excluded_mcp_servers.append(server_name)
            self.save(update_fields=["excluded_mcp_servers"])

    def remove_excluded_mcp_server(self, server_name: str) -> None:
        if server_name in self.excluded_mcp_servers:
            self.excluded_mcp_servers.remove(server_name)
            self.save(update_fields=["excluded_mcp_servers"])
