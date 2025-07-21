from django.db.models import Q
from django.views.generic import TemplateView

from .base import BaseAqueductView
from ..models import FileObject


class UserFilesView(BaseAqueductView, TemplateView):
    """
    Displays a list of FileObject instances accessible by the current user.
    """
    template_name = 'management/files.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        files = FileObject.objects.filter(
            Q(token__user=user) |
            Q(token__service_account__team__in=teams)
        ).order_by('-created_at')

        # Convert Unix timestamp to datetime for template date filter
        from datetime import datetime
        for f in files:
            f.created_dt = datetime.fromtimestamp(f.created_at)

        context['files'] = files
        return context
