from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from django.contrib import messages
from django.db.models import Count
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import Resolver404, resolve, reverse, reverse_lazy
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
from django.views.generic import TemplateView, UpdateView, View

from management.access import EveryOrg, TeamAct, WholeOrg
from management.forms import OrgAvailabilityForm, OrgLimitsForm
from management.models import Org
from management.org_choice import switch, without_org_params
from management.org_overview import org_page
from management.team_overview import team_rows
from management.views.base import BaseAqueductView

if TYPE_CHECKING:
    from django.utils.functional import _StrPromise


class OrgView(BaseAqueductView, TemplateView):
    template_name = "management/org.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        query = self.request.GET.get("q", "").strip()
        viewer = self.viewer
        is_org_admin = viewer.administers(viewer.org)
        teams = viewer.teams(TeamAct.SEE)
        context.update(
            is_org_admin=is_org_admin,
            all_orgs=isinstance(viewer.chosen, EveryOrg),
            in_no_team=not is_org_admin and not teams.exists(),
            query=query,
            teams=team_rows(
                teams, viewer.admin_of, query, self.request.GET.get("page", 1), timezone.now()
            ),
            org_object=viewer.chosen.org if isinstance(viewer.chosen, WholeOrg) else viewer.org,
        )
        return context


class OrganizationView(BaseAqueductView, TemplateView):
    def get_template_names(self) -> list[str]:
        if isinstance(self.viewer.chosen, EveryOrg):
            return ["management/organizations.html"]
        return ["management/organization.html"]

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        viewer = self.viewer
        if isinstance(viewer.chosen, EveryOrg):
            context["orgs"] = viewer.orgs().annotate(team_count=Count("teams")).order_by("name")
        else:
            org = get_object_or_404(viewer.orgs(), pk=viewer.chosen.org.pk)
            context.update(org_object=org, page=org_page(org))
        return context


class OrgEditView(BaseAqueductView, UpdateView):
    model = Org
    pk_url_kwarg = "id"
    template_name = "management/edit/org.html"
    context_object_name = "org_object"
    success_url = reverse_lazy("organization")
    view_title: "_StrPromise"
    saved: "_StrPromise"

    def get_object(self, queryset=None) -> Org:
        return get_object_or_404(self.viewer.orgs(), pk=self.kwargs[self.pk_url_kwarg])

    def form_valid(self, form) -> HttpResponse:
        response = super().form_valid(form)
        messages.success(self.request, self.saved % {"name": self.object.name})
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(view_title=self.view_title, **kwargs)


class OrgLimitsEditView(OrgEditView):
    form_class = OrgLimitsForm
    view_title = _("Limits")
    saved = _("Limits of %(name)s saved.")


class OrgAvailabilityEditView(OrgEditView):
    form_class = OrgAvailabilityForm
    view_title = _("Availability")
    saved = _("Availability of %(name)s saved.")


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
