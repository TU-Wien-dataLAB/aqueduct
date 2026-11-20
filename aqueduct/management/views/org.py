from urllib.parse import urlsplit

from django.contrib import messages
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import Resolver404, resolve, reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext
from django.views.generic import TemplateView, View

from management.access import WholeOrg
from management.models import Org
from management.org_choice import switch, without_org_params
from management.views.base import BaseAqueductView


class OrgView(BaseAqueductView, TemplateView):
    """Displays the organization dashboard showing accessible teams."""

    template_name = "management/org.html"  # Adjust path if needed

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        # Pass org admin status and list of teams to the template
        context["is_org_admin"] = self.is_org_admin()
        context["teams"] = self.get_teams_for_user()
        context["org_object"] = self.org
        # Add all orgs for admin users
        if self.profile.group == "admin":
            context["all_orgs"] = Org.objects.all()
        return context


class OrgSwitchView(BaseAqueductView, View):
    def post(self, request, *args, **kwargs) -> HttpResponse:
        if not self.viewer.can_switch_org:
            raise Http404

        choice = switch(request, request.POST.get("org_id", ""))
        messages.success(
            request,
            gettext("Viewing %(name)s.") % {"name": choice.org.name}
            if isinstance(choice, WholeOrg)
            else gettext("Viewing all organizations."),
        )
        back = request.POST.get("next", "")
        if url_has_allowed_host_and_scheme(back, {request.get_host()}, request.is_secure()):
            back = without_org_params(back)
            try:
                bound = bool(resolve(urlsplit(back).path).kwargs)
            except Resolver404:
                bound = True
            # A page about one org's team, key or settings would keep editing the old org.
            if not bound:
                return redirect(back)
        return redirect(reverse("org"))
