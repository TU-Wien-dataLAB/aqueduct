from django.db import models

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin
from management.models.org import Org


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
