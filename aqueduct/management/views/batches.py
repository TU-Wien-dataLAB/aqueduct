from datetime import datetime

from django.db.models import Q
from django.views.generic import TemplateView

from ..models import Batch
from .base import BaseAqueductView


class UserBatchesView(BaseAqueductView, TemplateView):
    """
    Displays a list of Batch instances accessible by the current user.
    """

    template_name = "management/batches.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        batches = Batch.objects.filter(
            Q(input_file__token__user=user) | Q(input_file__token__service_account__team__in=teams)
        ).order_by("-created_at")

        # Convert Unix timestamp to datetime for template date filter
        for b in batches:
            b.created_dt = datetime.fromtimestamp(b.created_at)
            # input file preview (first lines)
            b.input_file_preview = b.input_file.preview or ""
            # short preview of file id for display
            raw_id = b.input_file.id
            if len(raw_id) > 12:
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

        context["batches"] = batches
        return context
