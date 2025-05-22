from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.core.exceptions import ImproperlyConfigured
from django.db.models import QuerySet
from django.http import Http404  # Removed HttpResponseForbidden as redirect is used
from django.shortcuts import redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.views import View
# from django.conf import settings # No longer needed in this file

# Import your models (assuming they are in '..models')
from ..models import Team, UserProfile, ServiceAccount, Org


class BaseAqueductView(LoginRequiredMixin, View):
    """
    Base view providing user profile/org context for logged-in users.
    """
    login_url = reverse_lazy('sso')
    redirect_field_name = 'next'

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
    def org(self) -> Org:
        """
        Returns the Organization associated with the user's profile.
        """
        return self.profile.org

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
        else:
            # Regular user sees only teams they are a member of via the ManyToManyField.
            # The 'teams' attribute is guaranteed to exist due to the M2M definition.
            return self.profile.teams.all()


# --- Mixins for Permission Checks ---

class OrgAdminRequiredMixin:
    """
    Mixin requiring the user to be an administrator of their own organization.
    Must be used *after* LoginRequiredMixin and typically with BaseAqueductView
    or a view providing an `is_org_admin` method.
    """

    def dispatch(self, request, *args, **kwargs):
        # We rely on the inheriting view (like BaseAqueductView) providing is_org_admin.
        # This check ensures the method exists before calling it, making the mixin
        # slightly more robust if used outside the intended pattern, although
        # it's primarily designed to work with BaseAqueductView context.
        if not hasattr(self, 'is_org_admin') or not self.is_org_admin():
            messages.error(request, "You do not have permission to perform this action (Organization Admin required).")
            # Redirect to a sensible default page, like an org overview or dashboard.
            # Ensure 'org_dashboard' or similar exists in your URLs.
            # Using 'org' from the original code as a placeholder.
            return redirect(reverse_lazy('org'))
        return super().dispatch(request, *args, **kwargs)


class TeamAdminRequiredMixin:
    """
    Mixin requiring the user to be an admin of a *specific* team.
    Requires the view using this mixin to:
    1. Inherit from BaseAqueductView (or provide `self.profile`).
    2. Implement `get_team_object()` returning the relevant Team instance.
    """
    team_object: Team | None = None  # Cache the team object for the request duration

    def get_team_object(self) -> Team:
        """
        Placeholder method: Views using this mixin MUST override this method.
        It should fetch the specific Team instance based on URL kwargs (e.g., pk, slug)
        and raise Http404 if not found.
        Example implementation in a view:
            def get_team_object(self):
                team_id = self.kwargs.get('team_id') # Or other URL parameter
                return get_object_or_404(Team, pk=team_id, org=self.org) # Optionally scope to user's org
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_team_object()"
        )

    def dispatch(self, request, *args, **kwargs):
        # Assumes LoginRequiredMixin ran first, so request.user is authenticated.
        # Assumes BaseAqueductView context provides self.profile.

        try:
            # Fetch the specific team this view operates on.
            # Relies on the inheriting view's implementation of get_team_object.
            self.team_object = self.get_team_object()
            # Note: get_object_or_404 in get_team_object should handle not found cases.
            # A direct None check here is belt-and-suspenders.
            if not self.team_object:
                raise Http404("Team object could not be determined by get_team_object().")

        except Http404:
            messages.error(request, "The requested team was not found.")
            # Redirect to a safe fallback, like the organization dashboard.
            return redirect(reverse_lazy('org'))  # TODO: Update 'org' if needed

        # Check if the user has admin rights for *this specific* team.
        # Relies on the is_team_admin method in UserProfile model.
        # Assumes self.profile is available from BaseAqueductView.
        if not self.profile.is_team_admin(self.team_object):
            messages.error(request, f"You do not have permission to manage the team '{self.team_object.name}'.")
            # Attempt to redirect back to the team's detail view if possible.
            # Fallback to the org dashboard if the team view URL fails.
            try:
                # Ensure 'team_detail' or similar exists and accepts the team's ID/PK.
                team_url = reverse('team', kwargs={
                    'id': self.team_object.id})  # TODO: Update 'team' to your team detail URL name
                return redirect(team_url)
            except Exception:  # Catch potential NoReverseMatch or other errors
                return redirect(reverse_lazy('org'))  # TODO: Update 'org' if needed

        # Permission granted, proceed with the original view's dispatch.
        return super().dispatch(request, *args, **kwargs)


class BaseTeamView(BaseAqueductView):
    """
    Base view providing context for a specific Team instance fetched via URL kwarg.

    Inheriting views must:
    - Set `pk_url_kwarg` to the name of the URL keyword argument containing the Team's primary key (defaults to 'id').
    - Ensure the corresponding URL pattern captures this keyword argument.
    """
    pk_url_kwarg: str = 'id'  # Default URL kwarg name for the team's primary key
    _team: Team | None = None # Cache the team object for the request

    def get_team_object(self) -> Team:
        """
        Fetches and caches the Team instance based on pk_url_kwarg.
        Raises Http404 if not found or if the URL kwarg is missing.
        Permissions should be checked by mixins or subclass implementations.
        """
        if self._team is None:
            team_pk = self.kwargs.get(self.pk_url_kwarg)
            if team_pk is None:
                raise Http404(
                    f"URL keyword argument '{self.pk_url_kwarg}' not found in URLconf for {self.__class__.__name__}."
                )
            # Fetch the team. Permission mixins (like TeamAdminRequiredMixin)
            # or view logic should handle authorization.
            # We don't filter by self.org here by default, allowing org admins
            # to potentially access teams outside their direct membership if permitted.
            # Subclasses or mixins can override or add checks if needed.
            # Consider adding select_related('org') if org is frequently accessed.
            self._team = get_object_or_404(Team.objects.select_related('org'), pk=team_pk)
        return self._team

    @property
    def team(self) -> Team:
        """Returns the cached Team instance, fetching it if necessary."""
        return self.get_team_object()

    @property
    def team_org(self) -> Org:
        """Returns the Organization associated with the fetched team."""
        # Assumes get_team_object() was successful and team is loaded.
        # The select_related in get_team_object helps avoid an extra query here.
        return self.team.org

    # --- Convenience for TeamAdminRequiredMixin ---
    # If a view inherits from BaseTeamView AND TeamAdminRequiredMixin,
    # the mixin's requirement for get_team_object() is automatically satisfied.

    # --- Convenience for Generic Views ---
    # Override get_object common in DetailView/UpdateView/DeleteView
    # to return the team object by default if not overridden.
    def get_object(self, queryset=None):
        """
        Default implementation for generic views to return the fetched Team object.
        """
        # Note: This might conflict if a generic view *also* sets model = Team
        # and expects default get_object behavior based on pk/slug in URL.
        # However, since we fetch based on pk_url_kwarg, this aligns.
        return self.team


class BaseServiceAccountView(TeamAdminRequiredMixin, BaseAqueductView):
    """
    Base view for operations on a specific ServiceAccount instance.

    Requires TeamAdmin privileges for the SA's team.
    Automatically fetches the ServiceAccount and its Team.

    Views inheriting from this must:
    - Set `pk_url_kwarg` to the name of the URL keyword argument
      containing the ServiceAccount's primary key.
    - Optionally set the `model` attribute to ServiceAccount (often handled by Generic Views).
    - Inherit from a relevant generic view (e.g., View, DetailView, DeleteView)
      or implement appropriate methods (get, post, etc.).
    """
    model = ServiceAccount
    pk_url_kwarg: str | None = None  # Must be set by inheriting views (e.g., 'service_account_id')
    service_account_object: ServiceAccount | None = None  # Cache the fetched SA

    def get_service_account_object(self) -> ServiceAccount:
        """
        Fetches and caches the ServiceAccount instance based on pk_url_kwarg.
        Raises Http404 if not found.
        Called internally by get_team_object during dispatch.
        """
        if self.service_account_object is None:
            if not self.pk_url_kwarg:
                raise ImproperlyConfigured(
                    f"{self.__class__.__name__} must define 'pk_url_kwarg'."
                )
            sa_id = self.kwargs.get(self.pk_url_kwarg)
            if sa_id is None:
                raise Http404(
                    f"URL keyword argument '{self.pk_url_kwarg}' not found in URLconf."
                )
            # Fetch the SA. Team permission check happens later in the mixin's dispatch.
            self.service_account_object = get_object_or_404(self.model, pk=sa_id)
        return self.service_account_object

    # --- Implementation for TeamAdminRequiredMixin ---
    def get_team_object(self) -> Team:
        """
        Implementation for TeamAdminRequiredMixin.
        Fetches the team associated with the Service Account identified by pk_url_kwarg.
        """
        # Fetch the SA first. get_object_or_404 handles not found.
        sa = self.get_service_account_object()
        # ServiceAccount.team is a non-nullable ForeignKey, so sa.team will exist.
        return sa.team

    # --- Convenience for inheriting views ---
    def get_object(self, queryset=None):
        """
        Default implementation for generic views (like DetailView, DeleteView)
        to return the fetched Service Account object after permissions are checked.
        """
        # The SA is fetched and the team permission checked during dispatch.
        # We just return the cached object.
        return self.get_service_account_object()
