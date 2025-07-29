from django.conf import settings
from django.db.models import Q
from django.views.generic import TemplateView
from datetime import datetime, timedelta

from .base import BaseAqueductView
from ..models import Batch


class UserBatchesView(BaseAqueductView, TemplateView):
    """
    Displays a list of Batch instances accessible by the current user.
    """
    template_name = 'management/batches.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        batches = Batch.objects.filter(
            Q(input_file__token__user=user) |
            Q(input_file__token__service_account__team__in=teams)
        ).order_by('-created_at')

        # Convert Unix timestamp to datetime for template date filter
        from datetime import datetime
        for b in batches:
            b.created_dt = datetime.fromtimestamp(b.created_at)
            # input file preview (first lines)
            b.input_file_preview = b.input_file.preview()
            # short preview of file id for display
            raw_id = b.input_file.id
            if len(raw_id) > 12:
                b.input_file_id_preview = f"{raw_id[:7]}...{raw_id[-5:]}"
            else:
                b.input_file_id_preview = raw_id
            # output file preview and id
            if b.output_file:
                b.output_file_preview = b.output_file.preview()
                raw_out = b.output_file.id
                if len(raw_out) > 12:
                    b.output_file_id_preview = f"{raw_out[:7]}...{raw_out[-5:]}"
                else:
                    b.output_file_id_preview = raw_out
            else:
                b.output_file_preview = ''
                b.output_file_id_preview = '-'
            # error file preview and id
            if b.error_file:
                b.error_file_preview = b.error_file.preview()
                raw_err = b.error_file.id
                if len(raw_err) > 12:
                    b.error_file_id_preview = f"{raw_err[:7]}...{raw_err[-5:]}"
                else:
                    b.error_file_id_preview = raw_err
            else:
                b.error_file_preview = ''
                b.error_file_id_preview = '-'

        context['batches'] = batches
        crontab = settings.AQUEDUCT_BATCH_PROCESSING_CRONTAB
        # Total seconds until the next batch run
        context['next_run_in_seconds'] = crontab.remaining_estimate(
            datetime.now() - timedelta(seconds=1)
        ).seconds
        return context
