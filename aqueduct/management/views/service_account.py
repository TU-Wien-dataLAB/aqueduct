# management/views/service_account.py

from django.views.generic import CreateView, DeleteView, UpdateView
from django.views import View
from django.utils.decorators import method_decorator
from django.urls import reverse
from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect, get_object_or_404
from django.http import Http404, HttpResponseRedirect
from django.core.exceptions import ImproperlyConfigured  # Import for base view
from django.contrib.auth import get_user_model

# Import base views/mixins and models/forms
# Assuming base.py is in the same directory level
from .base import BaseAqueductView, TeamAdminRequiredMixin, BaseServiceAccountView, BaseTeamView
from ..models import Team, ServiceAccount, Token, UserProfile # Added UserProfile
from ..forms import ServiceAccountForm

User = get_user_model()


# --- Create View ---
# Uses TeamAdminRequiredMixin to ensure the user has admin rights on that team.
# Uses BaseTeamView to get the team context for creation.
class ServiceAccountCreateView(BaseTeamView, TeamAdminRequiredMixin, CreateView):
    """
    Handles creation of Service Accounts within a specific Team.
    Requires Team Admin privileges for that team.
    Also creates an initial Token owned by the creator.
    """
    model = ServiceAccount
    form_class = ServiceAccountForm
    template_name = 'management/create/service_account.html'
    pk_url_kwarg = 'id'  # Tells BaseTeamView which URL kwarg holds the team ID

    # TeamAdminRequiredMixin handles permission checks in dispatch, calling get_team_object from BaseTeamView.
    # get_team_object from BaseTeamView fetches the team based on pk_url_kwarg ('id').

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        # self.team is now provided by BaseTeamView
        kwargs['team'] = self.team
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # self.team available from BaseTeamView
        context['team'] = self.team
        context['view_title'] = f"Add Service Account to {self.team.name}"
        context['cancel_url'] = reverse('team', kwargs={'id': self.team.id})
        return context

    @transaction.atomic
    def form_valid(self, form):
        # self.team is available from BaseTeamView
        form.instance.team = self.team
        self.object = form.save()  # self.object is the new ServiceAccount

        try:
            user = self.request.user
            token_name = f"Initial token for {self.object.name}"

            # Create a new Token instance
            new_token = Token(
                name=token_name,
                user=user,
                service_account=self.object
            )
            # Generate and set key data on the new token instance
            secret_key = new_token._set_new_key()
            expires_at = form.cleaned_data.get('token_expires_at')
            if expires_at:
                new_token.expires_at = expires_at

            # Save the token (hash and preview are now set)
            new_token.save()

            messages.success(self.request,
                             f"Service Account '{self.object.name}' created for team '{self.team.name}'.")
            # Display the original secret key
            messages.info(self.request, f"{secret_key}",
                          extra_tags='token-regenerated-key')

        except Exception as e:
            # Log error e
            messages.error(self.request, f"Service Account saved, but failed to create associated token: {e}")
            # Consider if the SA should be rolled back if token creation fails
            # Since it's atomic, the SA save won't commit if token creation fails here.

        return redirect(self.get_success_url())

    def get_success_url(self):
        # self.team is guaranteed to exist if we reach here
        return reverse('team', kwargs={'id': self.team.id})


# --- Delete View ---
# Uses BaseServiceAccountView to fetch SA and handle permissions via TeamAdminRequiredMixin
class ServiceAccountDeleteView(BaseServiceAccountView, DeleteView):
    """
    Handles deletion of a Service Account. Requires Team Admin privileges for the SA's team.
    Inherits SA/Team fetching and permission checks from BaseServiceAccountView.
    """
    # model = ServiceAccount # Inherited from BaseServiceAccountView
    pk_url_kwarg = 'service_account_id'  # Tell base view how to find the SA ID
    template_name = 'management/common/confirm_delete.html'
    context_object_name = 'object'  # Used by DeleteView template

    # get_object is inherited from BaseServiceAccountView, which returns the fetched SA
    # get_team_object is inherited from BaseServiceAccountView for the mixin

    def get_success_url(self):
        """Redirect back to the team detail page after deletion."""
        # self.team_object is set by the mixin via BaseServiceAccountView's get_team_object
        return reverse('team', kwargs={'id': self.team_object.id})

    def get_context_data(self, **kwargs):
        """Provide context for the confirmation template."""
        context = super().get_context_data(**kwargs)
        # self.object is set by DeleteView using BaseServiceAccountView's get_object
        context['object_type_name'] = "Service Account"
        context['object_name'] = str(self.object)
        context['cancel_url'] = self.get_success_url()
        return context

    def delete(self, request, *args, **kwargs):
        """Handles the actual deletion on POST and adds a success message."""
        # self.object is populated by DeleteView calling get_object
        object_to_delete = self.get_object()
        object_name = str(object_to_delete)
        # self.team_object is populated during dispatch by the mixin
        team_name = self.team_object.name

        # Deleting ServiceAccount cascades to Token via on_delete=models.CASCADE
        response = super().delete(request, *args, **kwargs)  # Let DeleteView handle deletion

        messages.success(self.request,
                         f"Service Account '{object_name}' (from team '{team_name}') and its token deleted.")
        return response


# --- Update View ---
# Uses BaseServiceAccountView to fetch SA and handle permissions via TeamAdminRequiredMixin
class ServiceAccountUpdateView(BaseServiceAccountView, UpdateView):
    """
    Handles editing an existing Service Account (name, description).
    Requires Team Admin privileges for the SA's team.
    Uses ServiceAccountForm for validation.
    """
    # model = ServiceAccount # Inherited from BaseServiceAccountView
    form_class = ServiceAccountForm
    template_name = 'management/edit/service_account.html'
    pk_url_kwarg = 'service_account_id' # Tell base view how to find the SA ID
    context_object_name = 'service_account' # Match template usage

    def get_initial(self):
        initial = super().get_initial()
        try:
            token = Token.objects.get(service_account=self.get_object())
            if token.expires_at:
                # Format for datetime-local input
                initial['token_expires_at'] = token.expires_at.strftime('%Y-%m-%dT%H:%M')
        except Token.DoesNotExist:
            pass
        return initial

    # Permission checking is handled by TeamAdminRequiredMixin (via BaseServiceAccountView)

    def get_form_kwargs(self):
        """Pass the team instance to the form for validation (e.g., SA count limit)."""
        kwargs = super().get_form_kwargs()
        # self.team_object is guaranteed by the BaseServiceAccountView mixin
        kwargs['team'] = self.team_object
        return kwargs

    def form_valid(self, form):
        """Adds a success message upon successful update."""
        response = super().form_valid(form)
        # Update the token's expiration date if provided
        expires_at = form.cleaned_data.get('token_expires_at')
        try:
            token = Token.objects.get(service_account=self.object)
            if expires_at:
                token.expires_at = expires_at
            else:
                token.expires_at = None
            token.save(update_fields=['expires_at'])
        except Token.DoesNotExist:
            pass  # Optionally handle this case
        messages.success(self.request, f"Service Account '{self.object.name}' updated successfully.")
        return response

    def get_success_url(self):
        """Redirect back to the team detail page after successful edit."""
        # self.team_object is set by the mixin via BaseServiceAccountView's get_team_object
        return reverse('team', kwargs={'id': self.team_object.id})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = f'Edit Service Account: {self.object.name}'
        context['cancel_url'] = self.get_success_url() # Add cancel URL
        # context['service_account'] is set automatically by UpdateView
        context['team'] = self.team_object # Pass team for context if needed in template
        return context


# --- Transfer Ownership View ---
# Uses BaseServiceAccountView to fetch SA and handle permissions via TeamAdminRequiredMixin
class ServiceAccountTransferOwnershipView(BaseServiceAccountView, View):
    """
    Handles the transfer of a Service Account's Token ownership to another user within the team.
    Requires the requesting user to be an admin of the Service Account's team.
    GET displays the transfer form, POST processes the transfer.
    """
    template_name = 'management/transfer_service_account.html'
    pk_url_kwarg = 'service_account_id'  # Tell base view how to find the SA ID

    # get_object is inherited from BaseServiceAccountView, returns the SA
    # get_team_object is inherited from BaseServiceAccountView for the mixin

    def get_success_url(self) -> str:
        """Redirect back to the team detail page."""
        # self.team_object is set by the mixin via BaseServiceAccountView's get_team_object
        return reverse('team', kwargs={'id': self.team_object.id})

    def get_context_data(self, **kwargs):
        """Prepare context for the GET request (displaying the form)."""
        context = {}
        # self.service_account_object is set by BaseServiceAccountView's dispatch->get_object
        # self.team_object is set by BaseServiceAccountView's dispatch->get_team_object
        sa = self.service_account_object
        team = self.team_object

        context['service_account'] = sa
        context['team'] = team
        context['cancel_url'] = self.get_success_url()
        context['view_title'] = f'Transfer Ownership of {sa.name}'

        try:
            current_token = Token.objects.get(service_account=sa)
            current_owner = current_token.user
            context['current_owner_email'] = current_owner.email if current_owner else None
            # Eligible users are members of the team, excluding the current token owner
            eligible_profiles = team.member_profiles.exclude(user=current_owner).select_related('user')
        except Token.DoesNotExist:
            # This case should ideally not happen if SA exists, but handle gracefully
            context['current_owner_email'] = None
            eligible_profiles = team.member_profiles.all().select_related('user') # Show all members if no owner known
            messages.warning(self.request, f"Could not find the current token owner for {sa.name}.")
        except Exception as e:
            # Log error e
            messages.error(self.request, "An error occurred while determining eligible users.")
            eligible_profiles = UserProfile.objects.none() # Return empty queryset on error

        context['eligible_users'] = eligible_profiles
        return context

    def get(self, request, *args, **kwargs):
        """Handles GET requests: displays the transfer form."""
        # Fetch SA and Team objects using base view methods (called by dispatch)
        self.object = self.get_object() # Fetches SA via pk_url_kwarg
        # Permission check is implicitly done by TeamAdminRequiredMixin in dispatch via get_team_object

        context = self.get_context_data()
        return self.render_to_response(context)


    @method_decorator(transaction.atomic)
    def post(self, request, *args, **kwargs):
        """ Handles POST requests: processes the token transfer. """
        # Fetch SA and Team objects using base view methods (called by dispatch)
        sa = self.get_object()
        team = self.get_team_object() # Team context needed for validation/redirect
        requesting_profile = self.profile # From BaseAqueductView
        redirect_url = self.get_success_url()

        # Permission check (is_team_admin) is handled by TeamAdminRequiredMixin (via BaseServiceAccountView)

        target_profile_id = request.POST.get('target_profile_id')

        if not target_profile_id:
            messages.error(request, "No target user selected for transfer.")
            # Rerender the form if possible, or redirect back
            # For simplicity, redirecting back to the form page (GET) might be easiest
            # Or consider redirecting back to the team page
            return redirect(request.path) # Redirect back to the GET view of this transfer page

        try:
            # Validate the target profile
            target_profile = get_object_or_404(
                UserProfile.objects.select_related('user'), # Select related user
                id=target_profile_id,
                org=team.org, # Ensure target user is in the same org as the team
            )

            # Ensure the target profile is actually a member of the team
            if not team.member_profiles.filter(id=target_profile.id).exists():
                 messages.error(request, f"Selected user {target_profile.user.email} is not a member of team {team.name}.")
                 return redirect(request.path) # Redirect back to the form

            # Get the token to transfer
            token = Token.objects.select_for_update().get(service_account=sa)
            original_owner_email = token.user.email if token.user else "(No Owner)"

            if token.user == target_profile.user:
                messages.info(request, f"{target_profile.user.email} already owns the token for Service Account '{sa.name}'.")
            else:
                token.user = target_profile.user # Assign the target user
                token.save(update_fields=['user'])
                messages.success(request,
                                 f"Token ownership for '{sa.name}' transferred from {original_owner_email} to {target_profile.user.email}.")

        except UserProfile.DoesNotExist:
             messages.error(request, "Selected target user profile not found or not in this team's organization.")
             return redirect(request.path) # Redirect back to the form
        except Token.DoesNotExist:
            messages.error(request, f"No token found for Service Account '{sa.name}'. Cannot transfer.")
        except Exception as e:
            # Log error e
            # logger.error(f"Error transferring token for SA {sa.id} to profile {target_profile_id}: {e}", exc_info=True)
            messages.error(request, f"An unexpected error occurred during token transfer: {e}")
            # Consider redirecting back to form or team page depending on error
            return redirect(request.path) # Redirect back to form on generic error

        return redirect(redirect_url) # Redirect to team page on success

    # --- Helper for GET request ---
    def render_to_response(self, context):
        # A helper method like Django's TemplateResponseMixin would provide
        from django.template.response import TemplateResponse
        return TemplateResponse(
            request=self.request,
            template=self.template_name,
            context=context,
        )


