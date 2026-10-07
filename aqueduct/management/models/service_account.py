from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from management.models.team import Team


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
