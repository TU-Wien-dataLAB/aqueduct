# token_administration/views/org.py

from django.views.generic import TemplateView
from .base import BaseAqueductView  # Import from within the views package


class OrgView(BaseAqueductView, TemplateView):
    """Displays the organization dashboard showing accessible teams."""
    template_name = 'token_administration/org.html'  # Adjust path if needed

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Pass org admin status and list of teams to the template
        context['is_org_admin'] = self.is_org_admin()
        context['teams'] = self.get_teams_for_user()
        context['org_object'] = self.org  # Add the org object itself
        return context
