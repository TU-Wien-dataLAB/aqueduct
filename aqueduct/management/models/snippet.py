from typing import Any

from django.db import models


class SnippetType(models.TextChoices):
    CONFIG = "config", "config"
    PLUGIN = "plugin", "plugin"


class Snippet(models.Model):
    name = models.CharField(
        max_length=127, help_text="Human-readable name identifying this snippet."
    )
    type = models.CharField(max_length=20, choices=SnippetType.choices, default=SnippetType.CONFIG)
    active = models.BooleanField(
        default=False,
        help_text="Whether this snippet is enabled. At most one 'config' snippet may "
        "be active at a time.",
    )
    code = models.TextField(help_text="Python source of the snippet class.")
    updated_at = models.DateTimeField(auto_now=True, help_text="Last modification time.")

    class Meta:
        ordering = ("name", "type")
        verbose_name = "Snippet"
        verbose_name_plural = "Snippets"

    def __str__(self) -> str:
        return f"{self.name} ({self.get_type_display()})"

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self.active and self.type == SnippetType.CONFIG:
            Snippet.objects.filter(type=SnippetType.CONFIG, active=True).exclude(pk=self.pk).update(
                active=False
            )
        super().save(*args, **kwargs)
