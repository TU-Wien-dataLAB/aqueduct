import dataclasses
from typing import Optional

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import BooleanField, JSONField

# Physical maxima used to validate the hourly/daily limit multipliers: an hour has
# 60 minutes and a day has 1440 minutes (24 hours). A larger-window limit above
# these is unreachable because the per-minute cap already bounds usage.
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR


@dataclasses.dataclass(frozen=True)  # frozen=True makes instances immutable
class LimitSet:
    """Represents a resolved set of rate limits.

    The per-minute limits are the primary caps. The hourly/daily limits are
    derived from the per-minute limits via ``hourly_limit_multiplier`` /
    ``daily_limit_multiplier`` ("how many minutes-worth of the per-minute rate
    the larger window allows"). A ``None`` per-minute limit implies a ``None``
    derived limit for that metric in every larger window.
    """

    requests_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None
    hourly_limit_multiplier: int | None = None
    daily_limit_multiplier: int | None = None

    # Add future limit fields here with default None

    @classmethod
    def from_objects(
        cls, specific_limiter: Optional["LimitMixin"], org_limiter: Optional["LimitMixin"]
    ) -> "LimitSet":
        """
        Creates a LimitSet by resolving limits from a specific limiter
        (like Team or UserProfile) and a fallback Org limiter object.

        Each field is resolved independently through the hierarchy
        (specific -> org -> None), exactly like the per-minute fields.

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
            hourly_limit_multiplier=_resolve("hourly_limit_multiplier"),
            daily_limit_multiplier=_resolve("daily_limit_multiplier"),
        )

    def _hourly_mult(self) -> int:
        return (
            settings.AQUEDUCT_HOURLY_LIMIT_MULTIPLIER
            if self.hourly_limit_multiplier is None
            else self.hourly_limit_multiplier
        )

    def _daily_mult(self) -> int:
        return (
            settings.AQUEDUCT_DAILY_LIMIT_MULTIPLIER
            if self.daily_limit_multiplier is None
            else self.daily_limit_multiplier
        )

    @staticmethod
    def _scaled(base: int | None, mult: int) -> int | None:
        return None if base is None else base * mult

    def windows(self) -> list[tuple[str, int, int | None, int | None, int | None]]:
        """Effective per-window limits.

        Returns a list of ``(name, window_seconds, rpm, input_tokens, output_tokens)``,
        finest window first (``min``, ``hour``, ``day``). Hour/day limits are derived
        from the per-minute limits via the resolved multipliers; a ``None`` per-minute
        limit yields a ``None`` derived limit for that metric.
        """
        h, d = self._hourly_mult(), self._daily_mult()
        return [
            (
                "min",
                60,
                self.requests_per_minute,
                self.input_tokens_per_minute,
                self.output_tokens_per_minute,
            ),
            (
                "hour",
                3600,
                self._scaled(self.requests_per_minute, h),
                self._scaled(self.input_tokens_per_minute, h),
                self._scaled(self.output_tokens_per_minute, h),
            ),
            (
                "day",
                86400,
                self._scaled(self.requests_per_minute, d),
                self._scaled(self.input_tokens_per_minute, d),
                self._scaled(self.output_tokens_per_minute, d),
            ),
        ]


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
    hourly_limit_multiplier = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Hourly limit = per-minute limit x this. Null = fallback (default 60). "
        "Valid range 1-60 (an hour has 60 minutes). Lower = stricter sustained cap.",
    )
    daily_limit_multiplier = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Daily limit = per-minute limit x this. Null = fallback (default 1440). "
        "Valid range 1-1440 (a day has 1440 minutes). Lower = stricter sustained cap.",
    )

    class Meta:
        abstract = True  # Important: Makes this a mixin, no DB table created

    def clean(self) -> None:
        """Validate the hourly/daily limit multipliers.

        Because the per-minute window is the tightest, the most a token can do in
        one hour is ``per_minute x 60`` and in one day ``per_minute x 1440``. A
        larger-window limit above that is unreachable and therefore meaningless, so
        the multipliers are bounded by the number of minutes in the window. The
        lower bound ``1`` rejects ``0`` (which would make the derived limit ``0`` and
        block every request).
        """
        super().clean()
        errors: dict[str, str] = {}
        if self.hourly_limit_multiplier is not None and not (
            1 <= self.hourly_limit_multiplier <= MINUTES_PER_HOUR
        ):
            errors["hourly_limit_multiplier"] = "Must be between 1 and 60 (an hour has 60 minutes)."
        if self.daily_limit_multiplier is not None and not (
            1 <= self.daily_limit_multiplier <= MINUTES_PER_DAY
        ):
            errors["daily_limit_multiplier"] = (
                "Must be between 1 and 1440 (a day has 1440 minutes)."
            )
        if (
            self.hourly_limit_multiplier is not None
            and self.daily_limit_multiplier is not None
            and self.daily_limit_multiplier > HOURS_PER_DAY * self.hourly_limit_multiplier
        ):
            errors["daily_limit_multiplier"] = (
                "Cannot exceed 24 x hourly_limit_multiplier (a day has 24 hours)."
            )
        if errors:
            raise ValidationError(errors)


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
