from datetime import UTC, datetime

from django.db.models import Q
from django.utils import timezone
from django.views.generic import TemplateView

from management.models import Batch
from management.views.base import BaseAqueductView

INPUT_FILE_ID_PREVIEW_THRESHOLD = 12


class UserBatchesView(BaseAqueductView, TemplateView):
    """
    Displays a list of Batch instances accessible by the current user.
    """

    template_name = "management/batches.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        batches = Batch.objects.filter(
            Q(token__user=user) | Q(token__service_account__team__in=teams)
        ).order_by("-created_at")

        now_ts = int(timezone.now().timestamp())

        # Convert Unix timestamp to datetime for template date filter
        for b in batches:
            b.created_dt = datetime.fromtimestamp(b.created_at, tz=UTC)
            # input file preview (first lines)
            b.input_file_preview = b.input_file.preview or ""
            # short preview of file id for display
            raw_id = b.input_file.id
            if len(raw_id) > INPUT_FILE_ID_PREVIEW_THRESHOLD:
                b.input_file_id_preview = f"{raw_id[:7]}...{raw_id[-5:]}"
            else:
                b.input_file_id_preview = raw_id
            # output file preview
            if b.output_file:
                b.output_file_preview = b.output_file.preview or ""
                b.output_file_has_preview = bool(b.output_file_preview)
                b.output_file_id_preview = b.output_file.id
            else:
                b.output_file_preview = ""
                b.output_file_has_preview = False
                b.output_file_id_preview = "-"
            # error file preview
            if b.error_file:
                b.error_file_preview = b.error_file.preview or ""
                b.error_file_has_preview = bool(b.error_file_preview)
                b.error_file_id_preview = b.error_file.id
            else:
                b.error_file_preview = ""
                b.error_file_has_preview = False
                b.error_file_id_preview = "-"
            # expiration display
            if b.expires_at is not None:
                b.expires_dt = datetime.fromtimestamp(b.expires_at, tz=UTC)
                if b.expires_at <= now_ts:
                    b.expiration_status = "expired"
                elif b.expires_at <= now_ts + 86400 * 14:  # within 2 weeks
                    b.expiration_status = "expiring_soon"
                else:
                    b.expiration_status = "active"
            else:
                b.expires_dt = None
                b.expiration_status = "never"

        context["batches"] = batches
        return context
