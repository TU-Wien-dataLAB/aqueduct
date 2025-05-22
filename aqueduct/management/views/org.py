# management/views/org.py

from django.views.generic import TemplateView, View
from django.contrib import messages
from django.shortcuts import redirect, get_object_or_404
from django.urls import reverse
from .base import BaseAqueductView
from ..models import Org

class OrgView(BaseAqueductView, TemplateView):
    """Displays the organization dashboard showing accessible teams."""
    template_name = 'management/org.html'  # Adjust path if needed

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Pass org admin status and list of teams to the template
        context['is_org_admin'] = self.is_org_admin()
        context['teams'] = self.get_teams_for_user()
        context['org_object'] = self.org
        # Add all orgs for admin users
        if self.profile.group == 'admin':
            context['all_orgs'] = Org.objects.all()
        return context

class OrgSwitchView(BaseAqueductView, View):
    """
    Allows admin users to switch their current organization.
    """
    def post(self, request, *args, **kwargs):
        # Only allow if user is a global admin
        if self.profile.group != 'admin':
            messages.error(request, "Only admin users can switch organizations.")
            return redirect(reverse('org'))

        org_id = request.POST.get('org_id')
        if not org_id:
            messages.error(request, "No organization selected.")
            return redirect(reverse('org'))

        org = get_object_or_404(Org, pk=org_id)
        self.profile.org = org
        self.profile.save(update_fields=['org'])
        messages.success(request, f"Switched to organization: {org.name}")
        return redirect(reverse('org'))
