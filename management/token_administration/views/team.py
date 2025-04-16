# token_administration/views/team.py
from django.http import Http404
from django.views.generic import CreateView, DeleteView, DetailView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect, get_object_or_404
from django.views import View
from django.utils.decorators import method_decorator
# Removed Http404 as it wasn't used directly

# Import base views/mixins and models/forms
from .base import BaseAqueductView, OrgAdminRequiredMixin, BaseTeamView
from ..models import Team, UserProfile, ServiceAccount, Token, Org
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


class TeamDeleteView(OrgAdminRequiredMixin, BaseTeamView, DeleteView):
    """
    Handles deletion of a Team. Requires Org Admin privileges.
    Ensures the team belongs to the admin's organization implicitly via BaseTeamView fetching
    and OrgAdminRequiredMixin checks on self.org (user's org).
    """
    model = Team
    pk_url_kwarg = 'id'
    template_name = 'token_administration/common/confirm_delete.html'
    success_url = reverse_lazy('org')

    # Permission checking is handled by OrgAdminRequiredMixin and BaseTeamView fetches team.
    # No need for get_queryset to filter by org, as OrgAdminRequiredMixin ensures user is admin of their own org.
    # get_object() inherited from BaseTeamView returns self.team

    def form_valid(self, form):
        """Adds a success message upon successful deletion."""
        team_name = self.team.name # Use self.team from BaseTeamView
        response = super().form_valid(form)
        messages.success(self.request, f"Team '{team_name}' deleted successfully.")
        return response

    def get_context_data(self, **kwargs):
        """Provide context for the confirmation template."""
        context = super().get_context_data(**kwargs)
        context['object_type_name'] = self.model._meta.verbose_name
        context['object_name'] = str(self.team) # Use self.team
        context['cancel_url'] = self.get_success_url()
        # context['object'] is implicitly set to self.team by DeleteView calling get_object()
        return context


class TeamAdminManagementView(OrgAdminRequiredMixin, BaseTeamView, View):
    """
    Allows org-admins to add or remove team admin status for team members.
    Only org-admins can use this view. POST to add or remove.
    """
    pk_url_kwarg = 'id' # Tell BaseTeamView how to find the team

    def post(self, request, *args, **kwargs):
        # self.team is fetched by BaseTeamView
        team = self.team
        # self.profile and self.org (user's org) are available from BaseAqueductView/OrgAdminRequiredMixin
        profile_id = request.POST.get('profile_id')
        action = request.POST.get('action')
        # Ensure the profile being managed belongs to the same org as the team (and the admin)
        profile = get_object_or_404(UserProfile, id=profile_id, org=self.team_org) # Use team_org for consistency

        # OrgAdminRequiredMixin already confirmed the requesting user is an org admin.
        # Check if the target profile is the org admin themself (no-op)
        if profile == self.profile:
            messages.warning(request, "As an org-admin, you already have all team admin rights.")
            return redirect(reverse('team', kwargs={'id': team.id}))

        # Ensure the target profile is actually a member of the team before changing admin status
        # Although get_or_create handles creation, it might be confusing if they weren't explicitly added first.
        # Let's find the membership first.
        try:
            membership = team.teammembership_set.get(user_profile=profile)
        except team.teammembership_set.model.DoesNotExist:
            messages.error(request, f"{profile.user.email} is not a member of team {team.name}. Cannot modify admin status.")
            return redirect(reverse('team', kwargs={'id': team.id}))

        # membership, created = team.teammembership_set.get_or_create(user_profile=profile, team=team) # Old logic

        if action == 'add':
            if membership.is_admin:
                messages.info(request, f"{profile.user.email} is already a team admin.")
            else:
                membership.is_admin = True
                membership.save(update_fields=['is_admin'])
                messages.success(request, f"{profile.user.email} is now a team admin for {team.name}.")
        elif action == 'remove':
            if membership.is_admin:
                membership.is_admin = False
                membership.save(update_fields=['is_admin'])
                messages.success(request, f"{profile.user.email} is no longer a team admin for {team.name}.")
            else:
                messages.info(request, f"{profile.user.email} is not a team admin.")
        else:
            messages.error(request, "Invalid action.")
        return redirect(reverse('team', kwargs={'id': team.id}))


class TeamDetailView(BaseTeamView, DetailView):
    """
    Displays team details, members, and service accounts.
    Allows Team Admins (or Org Admins) to manage membership via POST requests.
    """
    model = Team
    template_name = 'token_administration/team.html'
    context_object_name = 'team'
    pk_url_kwarg = 'id'

    def dispatch(self, request, *args, **kwargs):
        # Fetch team first using BaseTeamView's mechanism
        # self.team will be set if successful, or Http404 raised
        try:
            # Trigger team fetching via property access
            _ = self.team
        except Http404:
            messages.error(request, "The requested team was not found.")
            return redirect(reverse_lazy('org')) # Redirect if team doesn't exist

        # Now check if the user has permission to view this team
        # Users should see teams they are members of OR if they are admin of the team's org.
        if not (self.team in self.get_teams_for_user()):
             messages.error(request, f"You do not have permission to view the team '{self.team.name}'.")
             return redirect(reverse_lazy('org')) # Redirect to org dashboard

        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # self.team and self.team_org are available from BaseTeamView
        # self.profile is available from BaseAqueductView
        # context['team'] is set automatically by DetailView using get_object() -> self.team

        context['is_org_admin'] = self.profile.is_org_admin(self.team_org) # Check against team's org
        # is_team_admin checks org admin status implicitly
        context['is_team_admin'] = self.profile.is_team_admin(self.team)

        context['service_accounts'] = self.team.service_accounts.select_related(
            'token__user__profile'
        ).all()

        context['profile_ids_owning_service_accounts'] = set(
            self.team.service_accounts
            .filter(token__user__profile__isnull=False)
            .values_list('token__user__profile__id', flat=True)
        )

        current_member_profile_ids = self.team.member_profiles.values_list('id', flat=True)
        # Show profiles from the TEAM's org, not necessarily the user's org if different (though unlikely)
        context['available_profiles'] = self.team_org.user_profiles.exclude(
            id__in=current_member_profile_ids
        )

        # --- Build member_badges: list of dicts for each member with badge info ---
        member_badges = []
        for member in self.team.member_profiles.all():
            badge = {
                'profile': member,
                'group': member.group,
                'is_team_admin': member.is_team_admin(self.team),
                'is_org_admin': member.is_org_admin(self.team_org), # Check against team's org
            }
            member_badges.append(badge)
        context['member_badges'] = member_badges
        return context

    @transaction.atomic # Ensure membership changes are atomic.
    def post(self, request, *args, **kwargs):
        """Handles adding/removing team members. Requires Team Admin privileges."""
        # self.team is fetched by BaseTeamView and verified in dispatch
        # self.profile is guaranteed by BaseAqueductView.
        team = self.team
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
                # as the team (which should match the admin's org unless they are superuser).
                profile_to_add = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_add,
                    org=self.team_org # Ensures org consistency with the team
                )
                if team.member_profiles.filter(id=profile_to_add.id).exists():
                    messages.warning(request, f"{profile_to_add.user.email} is already in the team.")
                else:
                    team.member_profiles.add(profile_to_add)
                    # Note: TeamMembership object (with is_admin=False) is created automatically.
                    messages.success(request, f"Added {profile_to_add.user.email} to team {team.name}.")
            except UserProfile.DoesNotExist:
                # This message covers both not found and wrong org cases due to the filter.
                messages.error(request, "User profile to add not found or not in this team's organization.")
            except Exception as e:
                 # Log the exception here for debugging
                 # logger.error(f"Error adding user to team {team.id}: {e}", exc_info=True)
                 messages.error(request, f"An unexpected error occurred while adding the user.")


        # --- Action: Remove User ---
        elif user_profile_id_to_remove:
            try:
                # Verify the profile exists and is part of the team's org before checking membership
                profile_to_remove = get_object_or_404(
                    UserProfile,
                    id=user_profile_id_to_remove,
                    org=self.team_org # Ensures they are in the correct org
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
                    team.member_profiles.remove(profile_to_remove)
                    messages.success(request, f"Removed {profile_to_remove.user.email} from team {team.name}.")
                else:
                    # User wasn't a member anyway.
                    messages.warning(request, f"{profile_to_remove.user.email} is not currently a member of this team.")

            except UserProfile.DoesNotExist:
                 # Covers not found or wrong org
                 messages.error(request, "User profile to remove not found or not in this team's organization.")
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