from django.db.models import Q
from django.views.generic import TemplateView

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

        context['batches'] = batches
        return context
