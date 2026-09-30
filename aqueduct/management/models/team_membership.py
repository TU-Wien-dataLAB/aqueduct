from django.db import models

from management.models.team import Team
from management.models.user_profile import UserProfile


class TeamMembership(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)

    # Your extra data about the relationship
    is_admin = models.BooleanField(default=False)
    date_added = models.DateField(auto_now_add=True)

    class Meta:
        # Ensure a user can only be in a team once
        unique_together = ("user_profile", "team")

    def __str__(self) -> str:
        return f"{self.user_profile} in {self.team}{' (Admin)' if self.is_admin else ''}"
