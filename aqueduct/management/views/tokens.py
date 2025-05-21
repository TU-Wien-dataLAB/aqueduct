from django.conf import settings
from django.utils import timezone

from .base import BaseAqueductView
from django.views.generic import TemplateView, CreateView, DeleteView, UpdateView
from ..models import Token, ServiceAccount
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from ..forms import TokenCreateForm
from django.db import transaction
from django.views import View
from django.shortcuts import redirect, get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_POST

class UserTokensView(BaseAqueductView, TemplateView):
    template_name = 'management/tokens.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile  # Provided by BaseAqueductView
        user = profile.user

        tokens = Token.objects.filter(user=user, service_account__isnull=True)
        teams = profile.teams.all()
        service_accounts = ServiceAccount.objects.filter(team__in=teams)

        from django.conf import settings
        max_tokens = getattr(settings, 'MAX_USER_TOKENS', 3)
        can_add_token = tokens.count() < max_tokens

        context.update({
            'tokens': tokens,
            'service_accounts': service_accounts,
            'can_add_token': can_add_token,
            'max_tokens': max_tokens,
        })
        return context

class TokenCreateView(BaseAqueductView, CreateView):
    model = Token
    form_class = TokenCreateForm
    template_name = 'management/create/token.html'
    success_url = reverse_lazy('tokens')  # Redirect to tokens list

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        # Don't set user here, it's done in form_valid
        # form.instance.user = self.request.user
        return form

    @transaction.atomic
    def form_valid(self, form):
        # Get the instance without saving to DB yet
        self.object = form.save(commit=False)

        # Assign the user directly to the instance
        self.object.user = self.request.user

        # Call the instance method to generate and set key data
        secret_key = self.object._set_new_key() # Call method on the instance

        # Now save the fully prepared object to the database
        # This will trigger clean() with the user assigned
        self.object.save()

        # Use the returned secret_key in the success message
        messages.success(self.request, f"Token '{self.object.name}' created successfully.")
        messages.info(self.request, f"{secret_key}", extra_tags='token-regenerated-key')

        return redirect(self.get_success_url())

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Create New Token'

        now = timezone.now().replace(second=0, microsecond=0)
        context['now'] = now.strftime('%Y-%m-%dT%H:%M')
        context['tz'] = settings.TIME_ZONE
        return context


class TokenEditView(BaseAqueductView, UpdateView):
    model = Token
    form_class = TokenCreateForm
    template_name = 'management/edit/token.html'
    pk_url_kwarg = 'id'
    success_url = reverse_lazy('tokens')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Edit Token'
        context['cancel_url'] = self.get_success_url()
        return context

class TokenDeleteView(BaseAqueductView, DeleteView):
    model = Token
    pk_url_kwarg = 'id'
    template_name = 'management/common/confirm_delete.html'
    success_url = reverse_lazy('tokens')

    def get_queryset(self):
        # Only allow deleting tokens owned by the user and not associated with a service account
        return Token.objects.filter(user=self.request.user, service_account__isnull=True)

    def form_valid(self, form):
        token_name = self.object.name
        response = super().form_valid(form)
        messages.success(self.request, f"Token '{token_name}' deleted successfully.")
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['object_type_name'] = 'Token'
        context['object_name'] = str(self.object)
        context['cancel_url'] = self.get_success_url()
        return context

@method_decorator(require_POST, name='dispatch')
class TokenRegenerateView(BaseAqueductView, View):
    """
    Handles regeneration of a token key (both user and service account tokens).
    Requires POST request.
    Permissions are checked based on token type.
    """
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        token_id = kwargs.get('id')
        profile = self.profile
        user = request.user

        try:
            token = get_object_or_404(Token.objects.select_related('service_account__team', 'user'), pk=token_id)
            token_name = token.name
            is_service_account_token = token.service_account is not None

            # --- Permission Check --- #
            can_regenerate = False
            redirect_url = reverse_lazy('tokens') # Default redirect

            if is_service_account_token:
                # Check if user is admin of the SA's team
                team = token.service_account.team
                redirect_url = reverse('team', kwargs={'id': team.id})
                if profile.is_team_admin(team):
                    can_regenerate = True
                else:
                    messages.error(request, f"You do not have permission to regenerate the token for service account '{token.service_account.name}'.")
            else:
                # Check if user owns the token
                if token.user == user:
                    can_regenerate = True
                else:
                    # This case shouldn't happen with correct view/URL setup, but good to have
                    messages.error(request, "You do not have permission to regenerate this token.")

            # --- Regeneration --- #
            if can_regenerate:
                new_key = token.regenerate_key() # Assumes regenerate_key() saves the token
                messages.success(
                    request,
                    f"Token '{token_name}' has been regenerated. The new key is displayed below. Please update any clients using the old key."
                )
                messages.info(
                    request,
                    f"{new_key}", # Only include the key itself in the message
                    extra_tags='token-regenerated-key' # Use a specific tag
                )
            # Else: error message already added during permission check

        except Token.DoesNotExist:
            messages.error(request, "The specified token could not be found.")
            # Redirect to a default page if token not found
            redirect_url = reverse_lazy('tokens')
        except Exception as e:
            # logger.error(f"Error regenerating token {token_id}: {e}", exc_info=True)
            messages.error(request, f"An unexpected error occurred while regenerating the token: {e}")
            # Try to determine redirect URL based on what we know, default if necessary
            try:
                # Attempt to get redirect URL again if token was fetched
                if is_service_account_token and team:
                    redirect_url = reverse('team', kwargs={'id': team.id})
            except NameError: # If token fetch failed earlier
                redirect_url = reverse_lazy('tokens')

        return redirect(redirect_url)