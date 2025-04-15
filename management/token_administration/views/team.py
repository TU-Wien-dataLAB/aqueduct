# token_administration/views/team.py

from django.views.generic import CreateView, DeleteView, DetailView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect, get_object_or_404
# Removed Http404 as it wasn't used directly

# Import base views/mixins and models/forms
from .base import BaseAqueductView, OrgAdminRequiredMixin
from ..models import Team, UserProfile, ServiceAccount, Token
from ..forms import TeamCreateForm


class TeamCreateView(OrgAdminRequiredMixin, BaseAqueductView, CreateView):
    """
    Handles creation of new Teams. Requires Org Admin privileges.
    """
    model = Team
    form_class = TeamCreateForm
    template_name = 'token_administration/create/team.html'
    success_url = reverse_lazy('org') # Redirects to organization dashboard

    # Permission checking is handled by OrgAdminRequiredMixin in dispatch.

    def get_form_kwargs(self):
        """Pass the user's organization to the form."""
        kwargs = super().get_form_kwargs()
        # self.org is guaranteed by BaseAqueductView for authenticated users.
        kwargs['org'] = self.org
        return kwargs

    def form_valid(self, form):
        """Assign the organization to the team instance before saving."""
        # self.org is guaranteed by BaseAqueductView.
        form.instance.org = self.org
        response = super().form_valid(form)
        messages.success(self.request, f"Team '{form.instance.name}' created successfully.")
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Create New Team'
        return context


class TeamDeleteView(OrgAdminRequiredMixin, BaseAqueductView, DeleteView):
    """
    Handles deletion of a Team. Requires Org Admin privileges.
    Ensures the team belongs to the admin's organization via get_queryset.
    """
    model = Team
    pk_url_kwarg = 'id'
    template_name = 'token_administration/common/confirm_delete.html'
    success_url = reverse_lazy('org')

    # Permission checking is handled by OrgAdminRequiredMixin in dispatch.

    def get_queryset(self):
        """Ensure deletion is restricted to teams within the user's org."""
        # self.org is guaranteed by BaseAqueductView & OrgAdminRequiredMixin.
        # The mixin ensures the user is an org admin, and the base view sets self.org.
        return Team.objects.filter(org=self.org)

    # get_object uses get_queryset, so team is implicitly scoped to the user's org.

    def form_valid(self, form):
        """Adds a success message upon successful deletion."""
        team_name = self.object.name # Get name before object is deleted
        response = super().form_valid(form)
        messages.success(self.request, f"Team '{team_name}' deleted successfully.")
        return response

    def get_context_data(self, **kwargs):
        """Provide context for the confirmation template."""
        context = super().get_context_data(**kwargs)
        context['object_type_name'] = self.model._meta.verbose_name
        context['object_name'] = str(self.object)
        context['cancel_url'] = self.get_success_url()
        return context


class TeamDetailView(BaseAqueductView, DetailView):
    """
    Displays team details, members, and service accounts.
    Allows Team Admins (or Org Admins) to manage membership via POST requests.
    """
    model = Team
    template_name = 'token_administration/team.html'
    context_object_name = 'team'
    pk_url_kwarg = 'id'

    def get_queryset(self):
        """
        Users can only view details for teams they have access to.
        Relies on get_teams_for_user() from the BaseAqueductView (or similar)
        to filter teams based on membership or org admin status.
        """
        # Assuming BaseAqueductView provides this method to filter teams
        # appropriate for the current user (e.g., member of, or org admin).
        return self.get_teams_for_user()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        team_instance = self.get_object() # Fetched via get_queryset
        # self.profile is guaranteed by BaseAqueductView for authenticated users.
        profile = self.profile

        # Determine if the current user is an admin of *this* specific team.
        # UserProfile.is_team_admin handles org admin checks internally.
        context['is_team_admin'] = profile.is_team_admin(team_instance)

        # Get associated service accounts and related user profiles efficiently.
        context['service_accounts'] = team_instance.service_accounts.select_related(
            'token__user__profile'
        ).all()

        # Get IDs of profiles owning Service Account tokens within this team.
        context['profile_ids_owning_service_accounts'] = set(
            team_instance.service_accounts
            .filter(token__user__profile__isnull=False)
            .values_list('token__user__profile__id', flat=True)
        )

        # Get available profiles from the same org who are not already members.
        # self.profile and self.profile.org are guaranteed non-None by BaseAqueductView
        # and the UserProfile model structure.
        current_member_profile_ids = team_instance.member_profiles.values_list('id', flat=True)
        context['available_profiles'] = profile.org.user_profiles.exclude(
            id__in=current_member_profile_ids
        )

        return context

    @transaction.atomic # Ensure membership changes are atomic.
    def post(self, request, *args, **kwargs):
        """Handles adding/removing team members. Requires Team Admin privileges."""
        team = self.get_object() # Fetched via get_queryset, ensuring user has basic access.
        # self.profile is guaranteed by BaseAqueductView.
        profile = self.profile
        redirect_url = reverse('team', kwargs={'id': team.id})

        # --- Security Check: Verify user is admin for *this specific team* ---
        # UserProfile.is_team_admin checks both explicit team admin status
        # and if the user is an admin of the team's organization.
        if not profile.is_team_admin(team):
            messages.error(request, "You do not have permission to modify this team's membership.")
            return redirect(redirect_url)

        user_profile_id_to_add = request.POST.get('user_profile_to_add_id')
        user_profile_id_to_remove = request.POST.get('user_profile_to_remove_id')

        # --- Action: Add User ---
        if user_profile_id_to_add:
            try:
                # Verify the profile exists and belongs to the *same organization*
                # as the current user (and thus the team). This prevents adding
                # users from other orgs even if their ID was somehow submitted.
                profile_to_add = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_add,
                    org=profile.org # Ensures org consistency
                )
                if team.member_profiles.filter(id=profile_to_add.id).exists():
                    messages.warning(request, f"{profile_to_add.user.email} is already in the team.")
                else:
                    team.member_profiles.add(profile_to_add)
                    # Note: TeamMembership object (with is_admin=False) is created automatically.
                    messages.success(request, f"Added {profile_to_add.user.email} to team {team.name}.")
            except UserProfile.DoesNotExist:
                # This message covers both not found and wrong org cases due to the filter.
                messages.error(request, "User profile to add not found or not in your organization.")
            except Exception as e:
                 # Log the exception here for debugging
                 # logger.error(f"Error adding user to team {team.id}: {e}", exc_info=True)
                 messages.error(request, f"An unexpected error occurred while adding the user.")


        # --- Action: Remove User ---
        elif user_profile_id_to_remove:
            try:
                # We only need the ID to find the profile, org check isn't strictly
                # needed here as we verify membership below, but doesn't hurt.
                profile_to_remove = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_remove,
                    # org=profile.org # Optional: Could add for extra check
                )

                # --- Critical Check: Prevent removal if user owns SA tokens in this team ---
                owns_sa_in_team = ServiceAccount.objects.filter(
                    team=team,
                    token__user__profile=profile_to_remove
                ).exists()

                if owns_sa_in_team:
                    messages.error(request, f"Cannot remove {profile_to_remove.user.email}. They own Service Account tokens in this team. Transfer or delete these tokens first.")
                    return redirect(redirect_url) # Stop processing removal

                # Verify the user is actually a member before attempting removal.
                if team.member_profiles.filter(id=profile_to_remove.id).exists():

                    # --- Optional Check: Prevent removing the last administrator ---
                    # This assumes UserProfile has `is_team_admin` and Team has `get_admin_count`
                    # Make sure these methods exist and are reliable.
                    is_last_admin = False
                    try:
                        # Check if the user being removed is an admin of this team *and*
                        # if the team would have zero admins left after removal.
                        if profile_to_remove.is_team_admin(team) and team.get_admin_count() <= 1:
                           is_last_admin = True
                    except AttributeError:
                        # Handle cases where is_team_admin or get_admin_count might not exist
                        # Log this issue during development/testing
                        # logger.warning(f"Could not perform 'last admin' check for team {team.id} removal.", exc_info=True)
                        pass # Default to allowing removal if check fails

                    if is_last_admin:
                        messages.error(request, "Cannot remove the last administrator from the team.")
                    else:
                        team.member_profiles.remove(profile_to_remove)
                        messages.success(request, f"Removed {profile_to_remove.user.email} from team {team.name}.")
                else:
                    # User wasn't a member anyway.
                    messages.warning(request, f"{profile_to_remove.user.email} is not currently a member of this team.")

            except UserProfile.DoesNotExist:
                 messages.error(request, "User profile to remove not found.")
            except Exception as e:
                 # Log the exception here for debugging
                 # logger.error(f"Error removing user from team {team.id}: {e}", exc_info=True)
                 messages.error(request, f"An unexpected error occurred while removing the user.")

        # --- No valid action ---
        elif not user_profile_id_to_add and not user_profile_id_to_remove:
            # Only show warning if the POST request was intended for modification
            # (e.g., check for a specific button name if necessary)
            # For simplicity, showing warning if neither ID is present.
            messages.warning(request, "No add or remove action specified.")

        return redirect(redirect_url)