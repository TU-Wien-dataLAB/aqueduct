# management/views/org.py

from datetime import timedelta

from django.contrib import messages
from django.db.models import Count, QuerySet, Sum
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.generic import TemplateView, UpdateView, View

from management.forms import OrgLimitsForm
from management.models import Org
from management.views.base import ORG_SESSION_KEY, BaseAqueductView


class OrgView(BaseAqueductView, TemplateView):
    """Displays the organization dashboard showing accessible teams."""

    template_name = "management/org.html"  # Adjust path if needed

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        # Pass org admin status and list of teams to the template
        context["is_org_admin"] = self.is_org_admin()
        teams = list(self.get_teams_for_user())
        week = (
            self.profile.visible_requests(self.org)
            .filter(token__service_account__team__in=teams)
            .filter(timestamp__gte=timezone.now() - timedelta(days=7))
            .values("token__service_account__team")
            .annotate(
                requests=Count("id"),
                tokens=Sum("input_tokens", default=0) + Sum("output_tokens", default=0),
            )
            .order_by()
        )
        by_team = {row["token__service_account__team"]: row for row in week}
        for team in teams:
            team.week = by_team.get(team.id)  # type: ignore[attr-defined]
        context["teams"] = teams
        context["org_object"] = self.org
        return context


class OrgLimitsEditView(BaseAqueductView, UpdateView):
    """The default limits of an org, for its admins."""

    form_class = OrgLimitsForm
    template_name = "management/edit/org.html"
    pk_url_kwarg = "id"
    context_object_name = "org_object"
    success_url = reverse_lazy("org")

    def get_queryset(self) -> QuerySet[Org]:
        return self.profile.administered_orgs()

    def form_valid(self, form: OrgLimitsForm) -> HttpResponse:
        messages.success(self.request, f"Limits of {form.instance.name} saved.")
        return super().form_valid(form)

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(view_title=f"Limits of {self.object.name}", **kwargs)


class OrgSwitchView(BaseAqueductView, View):
    """
    Allows admin users to switch their current organization.
    """

    def post(self, request, *args, **kwargs) -> HttpResponse:
        # Only allow if user is a global admin
        if self.profile.group != "admin":
            messages.error(request, "Only admin users can switch organizations.")
            return redirect(reverse("org"))

        org_id = request.POST.get("org_id")
        if not org_id:
            messages.error(request, "No organization selected.")
            return redirect(reverse("org"))

        org = get_object_or_404(Org, pk=org_id)
        # The session keeps the choice: moving the admin's profile would end their
        # memberships in the teams of their own org.
        request.session[ORG_SESSION_KEY] = org.pk
        messages.success(request, f"Switched to organization: {org.name}")
        # The sidebar's switch sends the page it was on, and the admin stays there.
        back = request.POST.get("next", "")
        if url_has_allowed_host_and_scheme(back, {request.get_host()}, request.is_secure()):
            return redirect(back)
        return redirect(reverse("org"))
