from functools import cached_property
from typing import Any

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages import get_messages
from django.contrib.messages.storage.base import Message
from django.db.models import QuerySet
from django.http import HttpRequest
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views import View

from management.access import TeamAct, Viewer, WholeOrg
from management.models import Org, ServiceAccount, Team, UserProfile
from management.org_choice import viewer_of

NEW_KEY_TAG = "token-regenerated-key"


def is_new_key(message: Message) -> bool:
    return (message.extra_tags or "").split(" ")[0] == NEW_KEY_TAG


def show_new_key(request: HttpRequest, key_id: int, secret: str) -> None:
    """Hand `secret` to the page of the key it belongs to, which shows it once."""
    messages.info(request, secret, extra_tags=f"{NEW_KEY_TAG} {key_id}")


def take_new_key(request: HttpRequest, key_id: int) -> str | None:
    """The secret `show_new_key` left for this key, read once; any page that renders drops it."""
    wanted = f"{NEW_KEY_TAG} {key_id}"
    return next(
        (str(message.message) for message in get_messages(request) if message.extra_tags == wanted),
        None,
    )


class BaseAqueductView(LoginRequiredMixin, View):
    """
    Base view providing user profile/org context for logged-in users.
    """

    login_url = reverse_lazy("sso")
    redirect_field_name = "next"

    # Caching profile lookup per request avoids repeated DB hits via user.profile
    _profile: UserProfile | None = None

    @property
    def profile(self) -> UserProfile:
        """
        Returns the UserProfile for the logged-in user.
        This assumes a UserProfile always exists for an authenticated user,
        typically enforced via signals or user creation logic.
        """
        # LoginRequiredMixin ensures self.request.user is not AnonymousUser.
        # The OneToOneField relationship ensures the 'profile' attribute exists.
        # If the related UserProfile object *doesn't* exist for some reason
        # (data inconsistency), user.profile access will raise RelatedObjectDoesNotExist.
        # We let that happen as it indicates a deeper issue.
        if self._profile is None:
            self._profile = self.request.user.profile
        return self._profile

    @property
    def viewer(self) -> Viewer:
        viewer = viewer_of(self.request)
        if viewer is None:
            raise UserProfile.DoesNotExist
        return viewer

    @property
    def org(self) -> Org:
        chosen = self.viewer.chosen
        return chosen.org if isinstance(chosen, WholeOrg) else self.profile.org

    def is_admin(self) -> bool:
        """Checks if the current user is a global admin."""
        return self.profile.is_admin()

    def is_org_admin(self) -> bool:
        """Checks if the current user is an admin of their associated organization."""
        # Relies on the is_org_admin method defined in the UserProfile model.
        # Assumes self.profile and self.org properties correctly return non-None objects.
        return self.profile.is_org_admin(self.org)

    def get_teams_for_user(self) -> QuerySet[Team]:
        """
        Returns a queryset of teams accessible by the current user.
        - Org admins see all teams in their org.
        - Regular users see only teams they are members of.
        """
        # profile and org are guaranteed by the properties above for logged-in users.
        if self.is_org_admin():
            # Org admin sees all teams in their organization.
            return self.org.teams.all()
        # Regular user sees only teams they are a member of via the ManyToManyField.
        # The 'teams' attribute is guaranteed to exist due to the M2M definition.
        return self.profile.teams.all()


class BaseTeamView(BaseAqueductView):
    """A view about the team in the URL; a team outside `viewer.teams(team_act)` is a 404."""

    pk_url_kwarg: str = "id"
    team_act: TeamAct = TeamAct.OPEN

    @cached_property
    def team(self) -> Team:
        return get_object_or_404(
            self.viewer.teams(self.team_act).select_related("org"),
            pk=self.kwargs[self.pk_url_kwarg],
        )

    def get_object(self, queryset: QuerySet[Any] | None = None) -> Team | ServiceAccount:
        return self.team


class BaseServiceAccountView(BaseAqueductView):
    """A view about the service account in the URL; acting on it is managing its team."""

    pk_url_kwarg = "service_account_id"

    @cached_property
    def service_account(self) -> ServiceAccount:
        return get_object_or_404(
            ServiceAccount.objects.filter(
                team__in=self.viewer.teams(TeamAct.MANAGE)
            ).select_related("team__org"),
            pk=self.kwargs[self.pk_url_kwarg],
        )

    @property
    def team(self) -> Team:
        return self.service_account.team

    def get_object(self, queryset: QuerySet[Any] | None = None) -> ServiceAccount:
        return self.service_account
