from collections.abc import Mapping
from enum import StrEnum
from functools import cached_property
from typing import Final

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.http import Http404, HttpResponse, QueryDict
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.translation import gettext
from django.views.generic import CreateView, DeleteView, DetailView, UpdateView

from management.access import KeyAct, TeamAct, Viewer, WholeOrg
from management.deletion import Crumb, deletion
from management.forms import MemberEmailForm, TeamAvailabilityForm, TeamCreateForm, TeamLimitsForm
from management.ids import parse_id
from management.models import Org, ServiceAccount, Team, UserProfile
from management.team_overview import team_page
from management.views.base import BaseAqueductView, BaseTeamView


class TeamCreateView(BaseAqueductView, CreateView):
    model = Team
    form_class = TeamCreateForm
    template_name = "management/create/team.html"
    success_url = reverse_lazy("org")

    @cached_property
    def target_org(self) -> Org:
        chosen = self.viewer.chosen
        pk = (chosen.org if isinstance(chosen, WholeOrg) else self.viewer.org).pk
        if "org" in self.request.POST:
            pk = parse_id(self.request.POST["org"])
        return get_object_or_404(self.viewer.orgs(), pk=pk)

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["org"] = self.target_org
        return kwargs

    def form_valid(self, form) -> HttpResponse:
        form.instance.org = self.target_org
        response = super().form_valid(form)
        messages.success(
            self.request, gettext("Team %(name)s created.") % {"name": form.instance.name}
        )
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = gettext("Create a team")
        context["target_org"] = self.target_org
        context["names_target_org"] = self.viewer.can_switch_org
        return context


class TeamUpdateView(BaseTeamView, UpdateView):
    model = Team
    template_name = "management/edit/team.html"
    context_object_name = "team"
    team_act = TeamAct.MANAGE

    def manages_team(self) -> bool:
        return self.viewer.may_team(TeamAct.ADMINISTER, self.team)

    def get_form_class(self) -> type[TeamCreateForm]:
        return TeamLimitsForm if self.manages_team() else TeamCreateForm

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["org"] = self.team.org
        return kwargs

    def form_valid(self, form) -> HttpResponse:
        response = super().form_valid(form)
        messages.success(self.request, gettext("Team %(name)s saved.") % {"name": self.object.name})
        return response

    def get_success_url(self) -> str:
        return reverse("team", kwargs={"id": self.object.id})

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = gettext("Edit team")
        context["cancel_url"] = self.get_success_url()
        context["can_delete"] = self.manages_team()
        return context


class TeamAvailabilityEditView(BaseTeamView, UpdateView):
    model = Team
    form_class = TeamAvailabilityForm
    template_name = "management/edit/team.html"
    context_object_name = "team"
    team_act = TeamAct.ADMINISTER

    def form_valid(self, form) -> HttpResponse:
        response = super().form_valid(form)
        messages.success(
            self.request, gettext("Availability of %(name)s saved.") % {"name": self.object.name}
        )
        return response

    def get_success_url(self) -> str:
        return reverse("team", kwargs={"id": self.object.id})

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(
            view_title=gettext("Availability"),
            breadcrumb=gettext("Availability"),
            cancel_url=self.get_success_url(),
            **kwargs,
        )


class TeamDeleteView(BaseTeamView, DeleteView):
    model = Team
    team_act = TeamAct.ADMINISTER
    template_name = "management/common/confirm_delete.html"
    success_url = reverse_lazy("org")

    def form_valid(self, form) -> HttpResponse:
        team_name = self.team.name
        response = super().form_valid(form)
        messages.success(self.request, gettext("Team %(name)s deleted.") % {"name": team_name})
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(
            confirmation=deletion(self.team),
            cancel_url=reverse("team", kwargs={"id": self.team.id}),
            trail=[
                Crumb(gettext("Teams"), reverse("org")),
                Crumb(self.team.name, reverse("team", kwargs={"id": self.team.id})),
            ],
            **kwargs,
        )


def member_form(team: Team, data=None) -> MemberEmailForm:
    return MemberEmailForm(
        data,
        candidates=team.org.user_profiles.all(),
        not_found=lambda email: (
            gettext("No one in %(org)s uses %(email)s.") % {"org": team.org.name, "email": email}
        ),
    )


def team_context(
    viewer: Viewer, team: Team, params: QueryDict, add_form: MemberEmailForm
) -> dict[str, object]:
    can_administer = viewer.may_team(TeamAct.ADMINISTER, team)
    member_query = params.get("q", "").strip()
    page = team_page(
        team,
        viewer.profile,
        can_administer,
        member_query,
        params.get("page", 1),
        timezone.now(),
        lambda token: viewer.may_key(KeyAct.OPEN, token),
    )
    max_service_accounts = getattr(settings, "MAX_SERVICE_ACCOUNTS_PER_TEAM", 10)
    return {
        "team": team,
        "add_form": add_form,
        "page": page,
        "member_query": member_query,
        "is_team_admin": viewer.may_team(TeamAct.MANAGE, team),
        "can_administer": can_administer,
        "max_service_accounts": max_service_accounts,
        "can_add_service_account": len(page.service_accounts) < max_service_accounts,
    }


class TeamDetailView(BaseTeamView, DetailView):
    model = Team
    template_name = "management/team.html"
    context_object_name = "team"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        team = self.team
        context.update(team_context(self.viewer, team, self.request.GET, member_form(team)))
        return context


class MemberAction(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    GRANT_ADMIN = "grant_admin"
    REVOKE_ADMIN = "revoke_admin"


MEMBER_ACTIONS: Final[Mapping[MemberAction, TeamAct]] = {
    MemberAction.ADD: TeamAct.MANAGE,
    MemberAction.REMOVE: TeamAct.MANAGE,
    MemberAction.GRANT_ADMIN: TeamAct.ADMINISTER,
    MemberAction.REVOKE_ADMIN: TeamAct.ADMINISTER,
}


class TeamMembersView(BaseAqueductView):
    """A membership change: the team is fetched for the act `MEMBER_ACTIONS` names."""

    @cached_property
    def action(self) -> MemberAction:
        try:
            return MemberAction(self.request.POST.get("action", ""))
        except ValueError:
            raise Http404 from None

    @cached_property
    def team(self) -> Team:
        return get_object_or_404(
            self.viewer.teams(MEMBER_ACTIONS[self.action]).select_related("org"),
            pk=self.kwargs["id"],
        )

    def get(self, request, *args, **kwargs) -> HttpResponse:
        """The team page's relative links (search, pager) after a re-rendered add form."""
        team_page = reverse("team", kwargs={"id": kwargs["id"]})
        return redirect(f"{team_page}?{request.GET.urlencode()}" if request.GET else team_page)

    @transaction.atomic
    def post(self, request, *args, **kwargs) -> HttpResponse:
        team = self.team
        match self.action:
            case MemberAction.ADD:
                return self.add(team)
            case MemberAction.REMOVE:
                return self.remove(team)
            case MemberAction.GRANT_ADMIN | MemberAction.REVOKE_ADMIN:
                return self.change_admin(team, self.action is MemberAction.GRANT_ADMIN)

    def add(self, team: Team) -> HttpResponse:
        form = member_form(team, self.request.POST)
        if form.is_valid() and team.member_profiles.filter(id=form.profile.id).exists():
            form.add_error(
                "email",
                gettext("%(email)s is already in this team.") % {"email": form.profile.user.email},
            )
        if not form.is_valid():
            context = team_context(self.viewer, team, self.request.GET, form)
            return TemplateResponse(self.request, TeamDetailView.template_name, context)
        team.member_profiles.add(form.profile)
        messages.success(
            self.request, gettext("%(email)s added.") % {"email": form.profile.user.email}
        )
        return redirect("team", id=team.id)

    def remove(self, team: Team) -> HttpResponse:
        viewer, request = self.viewer, self.request
        leaving = team.member_profiles.filter(id=parse_id(request.POST.get("profile_id"))).first()
        if leaving is None:
            messages.error(request, gettext("Member not found."))
        elif ServiceAccount.objects.filter(team=team, token__user__profile=leaving).exists():
            messages.error(
                request,
                gettext("Transfer or delete %(email)s's service accounts first.")
                % {"email": leaving.user.email},
            )
        else:
            team.member_profiles.remove(leaving)
            messages.success(request, gettext("%(email)s removed.") % {"email": leaving.user.email})
            if leaving == viewer.profile and not viewer.leaving(team).may_team(TeamAct.OPEN, team):
                return redirect("org")
        return redirect("team", id=team.id)

    def change_admin(self, team: Team, is_admin: bool) -> HttpResponse:
        request = self.request
        back = redirect("team", id=team.id)
        profile = get_object_or_404(
            UserProfile, id=parse_id(request.POST.get("profile_id")), org=team.org
        )
        email = profile.user.email
        if profile == self.viewer.profile:
            messages.warning(request, gettext("Org admins already have team admin rights."))
            return back
        membership = team.teammembership_set.filter(user_profile=profile).first()
        if membership is None:
            messages.error(request, gettext("%(email)s is not in this team.") % {"email": email})
        elif membership.is_admin == is_admin:
            already = (
                gettext("%(email)s is already a team admin.")
                if is_admin
                else gettext("%(email)s is not a team admin.")
            )
            messages.info(request, already % {"email": email})
        else:
            membership.is_admin = is_admin
            membership.save(update_fields=["is_admin"])
            changed = (
                gettext("%(email)s is now a team admin.")
                if is_admin
                else gettext("%(email)s is no longer a team admin.")
            )
            messages.success(request, changed % {"email": email})
        return back
