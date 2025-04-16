from .base import BaseAqueductView
from django.views.generic import TemplateView, CreateView
from ..models import Token, ServiceAccount
from django.urls import reverse_lazy
from django.contrib import messages
from ..forms import TokenCreateForm
from django.db import transaction

class UserTokensView(BaseAqueductView, TemplateView):
    template_name = 'token_administration/tokens.html'

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
    template_name = 'token_administration/create/token.html'
    success_url = reverse_lazy('tokens')  # Redirect to tokens list

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        form.instance.user = self.request.user  # Set user before validation
        return form

    @transaction.atomic
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, f"Token '{form.instance.name}' created successfully.")
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Create New Token'
        return context