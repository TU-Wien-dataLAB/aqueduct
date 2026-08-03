from datetime import UTC, datetime

from django.db.models import Q
from django.utils import timezone
from django.views.generic import TemplateView

from management.models import FileObject
from management.views.base import BaseAqueductView


class UserFilesView(BaseAqueductView, TemplateView):
    """
    Displays a list of FileObject instances accessible by the current user.
    """

    template_name = "management/files.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        files = FileObject.objects.filter(
            Q(token__user=user) | Q(token__service_account__team__in=teams)
        ).order_by("-created_at")

        now_ts = int(timezone.now().timestamp())

        for f in files:
            f.created_dt = datetime.fromtimestamp(f.created_at, tz=UTC)
            # Convert expires_at for template display
            if f.expires_at is not None:
                f.expires_dt = datetime.fromtimestamp(f.expires_at, tz=UTC)
                if f.expires_at <= now_ts:
                    f.expiration_status = "expired"
                elif f.expires_at <= now_ts + 86400 * 14:  # within 2 weeks
                    f.expiration_status = "expiring_soon"
                else:
                    f.expiration_status = "active"
            else:
                f.expires_dt = None
                f.expiration_status = "never"

        context["files"] = files
        return context
