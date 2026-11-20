from urllib.parse import urlsplit

from django.contrib import messages
from django.http import Http404, HttpResponse
from django.shortcuts import redirect
from django.urls import Resolver404, resolve, reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext
from django.views.generic import TemplateView, View

from management.access import EveryOrg, TeamAct, WholeOrg
from management.org_choice import switch, without_org_params
from management.team_overview import team_rows
from management.views.base import BaseAqueductView



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
