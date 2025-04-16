# token_administration/views/service_account.py

from django.views.generic import CreateView, DeleteView
from django.views import View
from django.utils.decorators import method_decorator
from django.urls import reverse
from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect, get_object_or_404
from django.http import Http404, HttpResponseRedirect
from django.core.exceptions import ImproperlyConfigured  # Import for base view

# Import base views/mixins and models/forms
# Assuming base.py is in the same directory level
from .base import BaseAqueductView, TeamAdminRequiredMixin, BaseServiceAccountView, BaseTeamView
from ..models import Team, ServiceAccount, Token
from ..forms import ServiceAccountForm


# --- Create View ---
# Uses BaseTeamView to get the team context for creation.
# Uses TeamAdminRequiredMixin to ensure the user has admin rights on that team.
class ServiceAccountCreateView(TeamAdminRequiredMixin, BaseTeamView, CreateView):
    """
    Handles creation of Service Accounts within a specific Team.
    Requires Team Admin privileges for that team.
    Also creates an initial Token owned by the creator.
    """
    model = ServiceAccount
    form_class = ServiceAccountForm
    template_name = 'token_administration/create/service_account.html'
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
            token_name = f"Initial token for {self.object.name}"  # Changed name slightly
            new_token = Token.objects.create(
                name=token_name,
                user=user,
                service_account=self.object
            )
            messages.success(self.request,
                             f"Service Account '{self.object.name}' created for team '{self.team.name}'.")
            messages.info(self.request, f"Associated Token '{new_token.name}' created. Key: {new_token.key}",
                          extra_tags='token-key-info')

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
    template_name = 'token_administration/common/confirm_delete.html'
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


# --- Transfer View ---
# Uses BaseServiceAccountView to fetch SA and handle permissions via TeamAdminRequiredMixin
class ServiceAccountTransferView(BaseServiceAccountView, View):
    """
    Transfers ownership of a Service Account's Token to the requesting user (POST only).
    Requires the requesting user to be an admin of the Service Account's team.
    Inherits SA/Team fetching and permission checks from BaseServiceAccountView.
    """
    http_method_names = ['post']  # Only allow POST requests
    pk_url_kwarg = 'service_account_id'  # Tell base view how to find the SA ID

    # get_team_object is inherited from BaseServiceAccountView for the mixin

    def get_success_url(self) -> str:
        """Redirect back to the team detail page."""
        # self.team_object is set by the mixin via BaseServiceAccountView's get_team_object
        return reverse('team', kwargs={'id': self.team_object.id})

    @method_decorator(transaction.atomic)
    def post(self, request, *args, **kwargs):
        """ Handles the token transfer logic. """
        # self.service_account_object is fetched and cached during dispatch via get_team_object
        sa = self.service_account_object
        requesting_user = self.request.user
        redirect_url = self.get_success_url()

        # Permission check (is_team_admin) is handled by TeamAdminRequiredMixin (via BaseServiceAccountView)

        try:
            token = Token.objects.select_for_update().get(service_account=sa)

            if token.user == requesting_user:
                messages.info(request, f"You already own the token for Service Account '{sa.name}'.")
            else:
                original_owner_email = token.user.email
                token.user = requesting_user
                token.save(update_fields=['user'])
                messages.success(request,
                                 f"Token ownership for '{sa.name}' transferred from {original_owner_email} to you.")

        except Token.DoesNotExist:
            messages.error(request, f"No token found for Service Account '{sa.name}'. Cannot transfer.")
        except Exception as e:
            # Log error e
            messages.error(request, f"An error occurred during token transfer: {e}")

        return redirect(redirect_url)
