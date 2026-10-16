from django.db import models

from management.models.mixins import LimitMixin, MCPServerExclusionMixin, ModelExclusionMixin


class Org(LimitMixin, ModelExclusionMixin, MCPServerExclusionMixin, models.Model):
    """Represents an Organization."""

    name = models.CharField(verbose_name="Org name", max_length=255, unique=True)

    def __str__(self) -> str:
        return self.name
