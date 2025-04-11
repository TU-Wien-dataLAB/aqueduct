from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import QuerySet
from django.http import HttpResponse, HttpResponseForbidden
from django.conf import settings
from django.views.generic import TemplateView, View, DetailView, CreateView
from django.shortcuts import redirect, render, get_object_or_404
from django.urls import reverse_lazy

from .forms import TeamCreateForm
from .models import Team, UserProfile


class SSOTemplateView(TemplateView):
    template_name = 'token_administration/login.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            # Redirect the logged-in user to the desired page
            return redirect('/')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add setting with fallback default
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context


class BaseTeamView(LoginRequiredMixin, View):
    """
    Base view for token administration, requiring login and providing
    common methods for team data and authorization checks.
    """
    login_url = reverse_lazy('sso')  # Or your actual login URL name
    redirect_field_name = 'next'

    @property
    def profile(self) -> UserProfile:
        """Helper method to get the user's profile."""
        # Add error handling if profile might not exist
        return self.request.user.profile

    def is_org_admin(self) -> bool:
        """Checks if the current user is an admin of their organization."""
        profile = self.profile
        if profile and profile.org:
            return profile.is_org_admin(profile.org)
        return False

    def get_teams_for_user(self) -> QuerySet[Team]:
        """
        Returns a queryset of teams accessible by the current user.
        Org admins see all teams in the org, others see only their teams.
        """
        profile = self.profile

        if self.is_org_admin():
            return profile.org.teams.all()
        else:
            # User is not org admin, return teams they are directly associated with
            return profile.teams.all()


class TokensView(BaseTeamView):
    """
    A simple view replacing the original 'tokens' function.
    Inherits login requirement from BaseTeamView.
    """

    def get(self, request, *args, **kwargs):
        # You can access methods from BaseTeamView if needed, e.g.:
        # teams = self.get_teams_for_user()
        return HttpResponse("Hello, world. You're at the tokens index (CBV).")


class OrgView(BaseTeamView, TemplateView):
    """
    Displays organization-level information.
    Inherits login requirement and provides org admin status to the template.
    """
    template_name = 'token_administration/org.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_org_admin'] = self.is_org_admin()
        context['teams'] = self.get_teams_for_user()
        return context


class TeamCreateView(BaseTeamView, CreateView):
    """
    Handles the creation of a new Team using a generic CreateView
    with custom form validation for uniqueness within the organization.
    """
    model = Team
    form_class = TeamCreateForm
    template_name = 'token_administration/create/team.html'
    success_url = reverse_lazy('org')

    # Store fetched org to avoid fetching twice
    org = None

    def dispatch(self, request, *args, **kwargs):
        """
        Check for org admin privileges and fetch the organization.
        Store the organization on the view instance.
        """
        profile = self.profile

        if not profile.is_org_admin(profile.org):
            return HttpResponseForbidden('You do not have permission to create teams.')

        self.org = profile.org
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        """
        Pass additional keyword arguments to the form's __init__ method.
        We pass the org fetched in dispatch.
        """
        kwargs = super().get_form_kwargs()
        # Add the organization to the kwargs dictionary
        kwargs['org'] = self.org
        return kwargs

    def form_valid(self, form):
        """
        Assign the organization to the model instance before saving.
        The uniqueness check has already passed in form.clean().
        """
        # Assign the organization (fetched in dispatch) to the instance
        form.instance.org = self.org
        # Let CreateView handle the actual saving and redirect
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Create New Team'
        return context


class TeamDetailView(BaseTeamView, DetailView):
    """
    Displays details for a specific team.
    Provides team admin status to the template.
    """
    model = Team
    template_name = 'token_administration/team.html'
    context_object_name = 'team'  # Use 'team' in the template (default is 'object' or 'team')
    pk_url_kwarg = 'id'  # Use 'id' from URL pattern as the primary key lookup

    def get_queryset(self):
        """
        Ensure users can only view details for teams they have access to,
        according to the logic in get_teams_for_user.
        """
        # Optimization: Filter the initial queryset based on user access.
        # This prevents users from accessing team detail pages via URL manipulation
        # if they shouldn't see the team at all.
        return self.get_teams_for_user()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # self.object is the Team instance fetched by DetailView
        team_instance = self.object
        profile = self.profile

        # Check if the profile exists and the team instance was successfully fetched
        if profile and team_instance:
            context['is_team_admin'] = profile.is_team_admin(team_instance)
        else:
            context['is_team_admin'] = False  # Default if something went wrong

        return context
