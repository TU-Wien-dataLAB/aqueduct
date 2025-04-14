from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import QuerySet
from django.http import HttpResponse, HttpResponseForbidden, Http404
from django.conf import settings
from django.views.generic import TemplateView, View, DetailView, CreateView, DeleteView
from django.shortcuts import redirect, render, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.db import transaction
from django.views.decorators.http import require_POST

from .forms import TeamCreateForm, ServiceAccountForm
from .models import Team, UserProfile, ServiceAccount, Token


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


class BaseAqueductView(LoginRequiredMixin, View):
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
        """Checks if the current user is an admin of their org."""
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


class TokensView(BaseAqueductView):
    """
    A simple view replacing the original 'tokens' function.
    Inherits login requirement from BaseTeamView.
    """

    def get(self, request, *args, **kwargs):
        # You can access methods from BaseTeamView if needed, e.g.:
        # teams = self.get_teams_for_user()
        return HttpResponse("Hello, world. You're at the tokens index (CBV).")


class OrgView(BaseAqueductView, TemplateView):
    """
    Displays org-level information.
    Inherits login requirement and provides org admin status to the template.
    """
    template_name = 'token_administration/org.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_org_admin'] = self.is_org_admin()
        context['teams'] = self.get_teams_for_user()
        return context


class TeamCreateView(BaseAqueductView, CreateView):
    """
    Handles the creation of a new Team using a generic CreateView
    with custom form validation for uniqueness within the org.
    """
    model = Team
    form_class = TeamCreateForm
    template_name = 'token_administration/create/team.html'
    success_url = reverse_lazy('org')

    # Store fetched org to avoid fetching twice
    org = None

    def dispatch(self, request, *args, **kwargs):
        """
        Check for org admin privileges and fetch the org.
        Store the org on the view instance.
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
        # Add the org to the kwargs dictionary
        kwargs['org'] = self.org
        return kwargs

    def form_valid(self, form):
        """
        Assign the org to the model instance before saving.
        The uniqueness check has already passed in form.clean().
        """
        # Assign the org (fetched in dispatch) to the instance
        form.instance.org = self.org
        # Let CreateView handle the actual saving and redirect
        response = super().form_valid(form)
        # Add a success message
        messages.success(self.request, f"Team '{form.instance.name}' has been created successfully.")
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Create New Team'
        return context


class TeamDeleteView(BaseAqueductView, DeleteView):
    """
    Handles the deletion of a specific Team instance after confirmation.
    Requires org admin privileges.
    """
    model = Team
    pk_url_kwarg = 'id'
    template_name = 'token_administration/common/confirm_delete.html'  # Standard confirmation template
    success_url = reverse_lazy('org')  # Redirect back to org view after deletion

    org = None
    object: Team = None

    def dispatch(self, request, *args, **kwargs):
        """
        Verify user is logged in and is an org admin before proceeding.
        Also fetches the org context.
        """
        profile = self.profile

        if not profile.is_org_admin(profile.org):
            # Return Forbidden if user isn't org admin, even if logged in
            return HttpResponseForbidden('You do not have permission to delete teams.')

        self.org = profile.org

        # Important: Call DeleteView's get_object *here* within dispatch
        # to ensure the object exists and belongs to the admin's org
        # *before* allowing GET (confirmation page) or POST (deletion).
        try:
            self.object = self.get_object()
        except Http404:
            messages.error(request, "Team not found or you do not have permission to access it.")
            return redirect(self.success_url)  # Redirect back if object not accessible

        return super(BaseAqueductView, self).dispatch(request, *args, **kwargs)

    def get_queryset(self):
        """
        Ensure that the org admin can only see/delete teams
        belonging to their own org.
        """
        # This is crucial for security. It prevents an admin from Org A
        # deleting a team from Org B by guessing the URL/ID.
        if self.org:
            return Team.objects.filter(org=self.org)
        else:
            # Should not happen due to dispatch check, but defensively return none
            return Team.objects.none()

    def get_object(self, queryset=None):
        """
        Override to ensure the object fetched belongs to the admin's org.
        get_queryset() already filters, but this adds an explicit layer.
        Called by dispatch now.
        """
        # Use the filtered queryset
        if queryset is None:
            queryset = self.get_queryset()

        # Fetch the object using the default DetailView/DeleteView logic
        # which uses pk_url_kwarg ('id')
        obj = super(TeamDeleteView, self).get_object(queryset=queryset)  # Use DeleteView's get_object
        return obj

    def form_valid(self, form):
        """
        Called when the POST request is confirmed (delete occurs here).
        Add a success message.
        """
        team_name = self.object.name  # Get name before deleting
        response = super().form_valid(form)
        messages.success(self.request, f"Team '{team_name}' has been deleted successfully.")
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['object_type_name'] = self.model._meta.verbose_name  # e.g., "Team"
        # Pass object name explicitly if needed (though {{ object.name }} often works)
        context['object_name'] = str(self.object)
        context['cancel_url'] = self.get_success_url()  # Pass cancel URL explicitly
        return context


class TeamDetailView(BaseAqueductView, DetailView):
    """
    Displays details for a specific team.
    Provides team admin status to the template.
    """
    model = Team
    template_name = 'token_administration/team.html'
    context_object_name = 'team'  # Use 'team' in the template (default is 'object' or 'team')
    pk_url_kwarg = 'id'  # Use 'id' from URL pattern as the primary key lookup

    def get_queryset(self):
        return self.get_teams_for_user()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        team_instance = self.get_object()
        profile = self.profile  # UserProfile of the logged-in user
        is_admin = profile.is_team_admin(team_instance) if profile and hasattr(profile, 'is_team_admin') else False
        context['is_team_admin'] = is_admin

        # --- Get Service Accounts for this Team ---
        # Fetch Service Accounts, prefetch related Token and its User for efficiency in template
        context['service_accounts'] = team_instance.service_accounts.select_related(
            'token__user'  # Select token and its user in one go
        ).prefetch_related(
            'token__user__profile'  # Also prefetch the profile related to the user
            # Use prefetch if UserProfile is accessed often or has complex lookups
            # Otherwise, select_related('token__user__profile') might work if 1:1 path is simple
        ).all()

        # --- Get Profile IDs of users whose tokens are linked to service accounts IN THIS TEAM ---
        # Path: Team -> ServiceAccount -> Token -> User -> UserProfile -> ID
        profile_ids_owning_service_accounts = set(
            team_instance.service_accounts
            .filter(
                token__isnull=False,
                token__user__isnull=False,
                token__user__profile__isnull=False  # Ensure User has related_name 'profile'
            )
            .values_list('token__user__profile__id', flat=True)  # Path to UserProfile ID
        )
        context['profile_ids_owning_service_accounts'] = profile_ids_owning_service_accounts

        # Filter available users (Profiles not already in the team)
        if profile and profile.org and team_instance:  # Assuming profile has 'org' attribute
            current_member_profile_ids = team_instance.member_profiles.values_list('id', flat=True)
            # Assuming org has 'user_profiles' related manager
            available_profiles = profile.org.user_profiles.exclude(id__in=current_member_profile_ids)
            context['available_profiles'] = available_profiles
        else:
            context['available_profiles'] = UserProfile.objects.none()  # Return an empty queryset

        return context

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        team = self.get_object()
        profile = self.profile  # UserProfile of the logged-in user

        # --- Security Check: Must be Team Admin ---
        if not profile or not hasattr(profile, 'is_team_admin') or not profile.is_team_admin(team):
            messages.error(request, "You do not have permission to modify this team's membership.")
            return redirect(reverse('team', kwargs={'id': team.id}))

        user_profile_id_to_add = request.POST.get('user_profile_to_add_id')
        user_profile_id_to_remove = request.POST.get('user_profile_to_remove_id')

        # --- Action: Add User ---
        if user_profile_id_to_add:
            try:
                # Ensure adding a profile from the same organization
                profile_to_add = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_add,
                    org=profile.org  # Assuming UserProfile has an 'org' FK/attribute
                )

                # Check if already a member (using the M2M relationship on Team model)
                if team.member_profiles.filter(id=profile_to_add.id).exists():
                    messages.warning(request, f"{profile_to_add.user.email} is already in the team.")
                else:
                    team.member_profiles.add(profile_to_add)
                    messages.success(request,
                                     f"Successfully added {profile_to_add.user.email} to the team {team.name}.")

            except UserProfile.DoesNotExist:
                messages.error(request,
                               "The selected user profile to add could not be found or does not belong to your organization.")
            except AttributeError:
                messages.error(request,
                               "Could not verify organization membership for adding user.")  # If profile.org doesn't exist
            except Exception as e:
                messages.error(request, f"An error occurred while adding the user: {e}")

        # --- Action: Remove User ---
        elif user_profile_id_to_remove:
            try:
                profile_to_remove = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_remove
                    # No org check needed here explicitly, as we first check if they are in *this* team's members
                )

                # *** CORRECTED CHECK: Does this user profile own tokens linked to SAs in THIS team? ***
                # Path: ServiceAccount -> token -> user -> profile
                owns_sa_in_team = ServiceAccount.objects.filter(
                    team=team,  # Filter for SAs in the current team
                    token__isnull=False,  # Ensure the SA has a token
                    token__user__profile=profile_to_remove  # Check if the token's user's profile matches
                    # Assumes UserProfile has a OneToOne back to User,
                    # and User.profile reverse accessor exists.
                ).exists()  # Use exists() for efficiency

                if owns_sa_in_team:
                    messages.error(request,
                                   f"Cannot remove {profile_to_remove.user.email}. They own tokens linked to Service Accounts in this team. Please transfer or delete the relevant tokens/service accounts first.")
                    return redirect(reverse('team', kwargs={'id': team.id}))
                # *** END CORRECTED CHECK ***

                # Ensure the profile is actually a member before trying to remove
                if team.member_profiles.filter(id=profile_to_remove.id).exists():
                    # Optional: Prevent removing the last administrator
                    is_last_admin = False
                    if hasattr(profile_to_remove, 'is_team_admin') and hasattr(team, 'get_admin_count'):
                        if profile_to_remove.is_team_admin(team) and team.get_admin_count() <= 1:
                            is_last_admin = True

                    if is_last_admin:
                        messages.error(request, "Cannot remove the last administrator from the team.")
                        return redirect(reverse('team', kwargs={'id': team.id}))

                    # Proceed with removal
                    team.member_profiles.remove(profile_to_remove)
                    messages.success(request,
                                     f"Successfully removed {profile_to_remove.user.email} from the team {team.name}.")
                else:
                    # This case might happen if the page state is old
                    messages.warning(request, f"{profile_to_remove.user.email} is not currently a member of this team.")

            except UserProfile.DoesNotExist:
                messages.error(request, "The selected user profile to remove could not be found.")
            except Exception as e:
                # Log the exception for debugging
                # logger.error(f"Error removing user {user_profile_id_to_remove} from team {team.id}: {e}", exc_info=True)
                messages.error(request, f"An error occurred while removing the user: {e}")

        # --- No valid action identified ---
        elif not user_profile_id_to_add and not user_profile_id_to_remove:
            # This might happen if the form is submitted unexpectedly empty
            messages.warning(request, "No add or remove action specified.")

        # --- Redirect back to the team detail page ---
        return redirect(reverse('team', kwargs={'id': team.id}))


class ServiceAccountCreateView(BaseAqueductView, CreateView):
    model = ServiceAccount
    form_class = ServiceAccountForm
    template_name = 'token_administration/create/service_account.html'
    team = None  # Store the team

    def dispatch(self, request, *args, **kwargs):
        team_id = self.kwargs.get('id')
        self.team = get_object_or_404(Team, pk=team_id)

        # --- Security Check: Must be Team Admin ---
        if not self.profile or not hasattr(self.profile, 'is_team_admin') or not self.profile.is_team_admin(self.team):
            messages.error(request, "You do not have permission to modify this team's membership.")
            return redirect(reverse('team', kwargs={'id': self.team.id}))

        return super().dispatch(request, *args, **kwargs)

    # Pass the team to the form
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['team'] = self.team  # Pass the fetched team object
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['team'] = self.team
        context['view_title'] = f"Add Service Account to {self.team.name}"
        context['cancel_url'] = reverse('team', kwargs={'id': self.team.id})
        return context

    def form_valid(self, form):
        # Assign the team *before* saving (though validation already happened)
        form.instance.team = self.team

        # Use transaction for saving SA and creating Token
        try:
            with transaction.atomic():
                self.object = form.save()  # Save ServiceAccount

                user = self.request.user
                token_name = f"Token for {self.object.name}"
                new_token = Token.objects.create(
                    name=token_name,
                    user=user,
                    service_account=self.object
                )
                messages.success(self.request,
                                 f"Service Account '{self.object.name}' created successfully for team '{self.team.name}'.")
                messages.info(self.request, f"Associated Token '{new_token.name}' created. Key: {new_token.key}",
                              extra_tags='token-key-info')

        except Exception as e:
            messages.error(self.request, f"An error occurred: {e}")
            form.add_error(None, "An unexpected error occurred during creation.")
            return self.form_invalid(form)

        return redirect(self.get_success_url())

    def get_success_url(self):
        return reverse('team', kwargs={'id': self.team.id})


class ServiceAccountDeleteView(BaseAqueductView, DeleteView):
    """
    Handles the confirmation and deletion of a ServiceAccount.
    Requires the user to be an admin of the associated team.
    """
    model = ServiceAccount
    pk_url_kwarg = 'service_account_id'  # Match the URL parameter name from your urls.py
    template_name = 'token_administration/common/confirm_delete.html'
    context_object_name = 'object'  # The object will be available as 'object' in the template

    team = None  # To store team for redirection

    def dispatch(self, request, *args, **kwargs):
        """
        Verify user is logged in and has permission (is team member)
        before showing the confirmation page or processing deletion.
        """
        # Fetch the object first to check its team association.
        try:
            # Ensure get_queryset is appropriate or override get_object carefully.
            sa_object: ServiceAccount = self.get_object()
            self.team = sa_object.team

            if not self.team:
                messages.error(request, "Service Account is not associated with a team.")
                return redirect('org')

        except Http404:
            messages.error(request, "Service Account not found.")
            # Redirect to a safe fallback page if object doesn't exist
            return redirect('org')

        # --- Permission Check ---
        if not self.profile or not hasattr(self.profile, 'is_team_admin') or not self.profile.is_team_admin(self.team):
            messages.error(request, "You do not have permission to delete service accounts for this team.")
            return redirect(reverse('team', kwargs={'id': self.team.id}))

        return super().dispatch(request, *args, **kwargs)

    def get_success_url(self):
        """
        Determine the URL to redirect to after successful deletion.
        """
        if self.team:
            # Redirect to the detail page of the team the SA belonged to
            return reverse('team', kwargs={'id': self.team.id})
        else:
            # Fallback URL if team_id wasn't available (shouldn't normally happen after dispatch)
            messages.warning(self.request, "Could not determine team context, redirecting to dashboard.")
            return reverse('org')

    def get_context_data(self, **kwargs):
        """
        Add extra context needed by the confirmation template.
        """
        context = super().get_context_data(**kwargs)
        context['object_type_name'] = "Service Account"  # Name for the template
        context['object_name'] = str(self.object)  # String representation of the SA
        context['cancel_url'] = self.get_success_url()  # URL for the 'Cancel' button
        return context

    # Override delete() method to add success message (more robust than form_valid for DeleteView)
    def delete(self, request, *args, **kwargs):
        """
        Handles the POST request; deletes the object and then redirects.
        Adds a success message.
        """
        object_to_delete = self.get_object()  # Get object again safely
        object_name = str(object_to_delete)
        # Deleting ServiceAccount cascades to Token via on_delete=models.CASCADE

        # Call the parent class's delete method to perform the actual deletion
        response = super().delete(request, *args, **kwargs)

        # Add success message *after* successful deletion
        messages.success(self.request, f"Service Account '{object_name}' and its associated token have been deleted.")

        return response


# --- Function-Based View for Transfer ---

@login_required
@require_POST
def service_account_transfer(request, service_account_id):
    """
    Handles transferring the ownership of a ServiceAccount's token
    to the currently logged-in user. (No confirmation step).
    """
    sa = get_object_or_404(ServiceAccount, pk=service_account_id)
    team = sa.team  # Get the team for permission checks and redirection
    profile: UserProfile = request.user.profile

    # Determine the redirect URL early (used for success and failure cases)
    if team:
        redirect_url = reverse('team', kwargs={'id': team.id})
    else:
        redirect_url = reverse('org')

    # --- Permission Check ---
    if not profile.is_team_admin(team):
        messages.error(request, "You must be a team administrator to transfer ownership of this service account.")
        return redirect(reverse('team', kwargs={'id': team.id}))

    try:
        token = get_object_or_404(Token, service_account=sa)

        # Check if the user already owns the token
        if token.user == request.user:
            messages.info(request, f"You already own the token for service account '{sa.name}'.")
        else:
            # Transfer ownership
            token.user = request.user
            token.save(update_fields=['user'])  # Efficiently update only the user field
            messages.success(request, f"Ownership of service account '{sa.name}' token transferred to you.")

    except Token.DoesNotExist:
        messages.error(request,
                       f"Could not find a token associated with service account '{sa.name}'. Cannot transfer ownership.")
    except Exception as e:
        messages.error(request, f"An error occurred while transferring ownership: {e}")

    return redirect(redirect_url)
