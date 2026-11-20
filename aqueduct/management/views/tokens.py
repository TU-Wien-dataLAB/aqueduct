from typing import ClassVar

from django import forms
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.cache import patch_cache_control
from django.utils.decorators import method_decorator
from django.utils.translation import gettext
from django.views import View
from django.views.decorators.http import require_POST
from django.views.generic import CreateView, DeleteView, DetailView, TemplateView, UpdateView

from gateway.config import model_names
from management.access import KeyAct
from management.availability import Reach, configured_mcp_servers
from management.deletion import Crumb, deletion
from management.forms import TokenCreateForm
from management.keys import (
    FIRST_KEY_NAME,
    Expiry,
    FirstKey,
    KeyKind,
    NewKey,
    first_key,
    key_list,
    key_report,
)
from management.limits import panels
from management.models import Token
from management.nav import gateway_url
from management.ownership import RELATED, team_of
from management.views.base import BaseAqueductView, show_new_key, take_new_key


class UserTokensView(BaseAqueductView, TemplateView):
    template_name = "management/tokens.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        kind = KeyKind.parse(self.request.GET.get("kind", ""))
        query = self.request.GET.get("q", "").strip()
        max_tokens = getattr(settings, "MAX_USER_TOKENS", 3)
        keys = key_list(self.viewer, kind, query, timezone.now())
        first: FirstKey | None = None
        if keys.personal_count == 0 and not (kind or query):
            form = TokenCreateForm(user=self.request.user, initial={"name": FIRST_KEY_NAME})
            first = first_key(self.viewer, form, gateway_url(self.request))
        context.update(
            keys=keys,
            kinds=tuple(KeyKind),
            kind=kind,
            query=query,
            max_tokens=max_tokens,
            can_add_token=keys.personal_count < max_tokens,
            first_key=first,
        )
        return context


class TokenDetailView(BaseAqueductView, DetailView):
    model = Token
    template_name = "management/token_detail.html"
    pk_url_kwarg = "id"
    context_object_name = "token"

    def get_queryset(self) -> QuerySet:
        return self.viewer.keys(KeyAct.OPEN).select_related(*RELATED)

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        viewer, token = self.viewer, self.object
        now = timezone.now()
        report = key_report(token, now)
        secret = take_new_key(self.request, token.id)
        can_manage = viewer.may_key(KeyAct.MANAGE, token)
        context.update(
            new_key=NewKey.of(token, secret, gateway_url(self.request))
            if secret and can_manage
            else None,
            report=report,
            charts=tuple((shown, report.chart(shown)) for shown in ("tokens", "requests")),
            expiry=Expiry.of(token.expires_at, now),
            panel=panels([token], now)[token.id],
            models=Reach.of(model_names(), token.model_exclusion_list()),
            mcp_servers=Reach.of(configured_mcp_servers(), token.mcp_server_exclusion_list()),
            team=team_of(token),
            own=viewer.may_key(KeyAct.OWN, token),
            can_manage=can_manage,
        )
        return context

    def render_to_response(self, context, **response_kwargs) -> HttpResponse:
        response = super().render_to_response(context, **response_kwargs)
        if context["new_key"]:
            # Back must not bring the secret up again from the browser's cache.
            patch_cache_control(response, no_store=True, private=True)
        return response


class TokenCreateView(BaseAqueductView, CreateView):
    model = Token
    form_class = TokenCreateForm
    template_name = "management/create/token.html"

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_form(self, form_class=None) -> forms.BaseForm:
        return super().get_form(form_class)

    @transaction.atomic
    def form_valid(self, form) -> HttpResponse:
        # Get the instance without saving to DB yet
        self.object = form.save(commit=False)

        # Assign the user directly to the instance
        self.object.user = self.request.user

        # Call the instance method to generate and set key data
        secret_key = self.object._set_new_key()  # Call method on the instance

        # Now save the fully prepared object to the database
        # This will trigger clean() with the user assigned
        self.object.save()

        show_new_key(self.request, self.object.id, secret_key)

        return redirect("token_detail", id=self.object.id)

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = gettext("New API key")
        return context


class TokenEditView(BaseAqueductView, UpdateView):
    model = Token
    form_class = TokenCreateForm
    template_name = "management/edit/token.html"
    pk_url_kwarg = "id"

    def get_queryset(self) -> QuerySet:
        return self.viewer.keys(KeyAct.OWN)

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_success_url(self) -> str:
        return reverse("token_detail", kwargs={"id": self.object.id})

    def form_valid(self, form) -> HttpResponse:
        response = super().form_valid(form)
        messages.success(
            self.request, gettext("API key %(name)s saved.") % {"name": self.object.name}
        )
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["cancel_url"] = self.get_success_url()
        return context


class TokenDeleteView(BaseAqueductView, DeleteView):
    model = Token
    pk_url_kwarg = "id"
    template_name = "management/common/confirm_delete.html"
    success_url = reverse_lazy("tokens")

    def get_queryset(self) -> QuerySet:
        return self.viewer.keys(KeyAct.OWN)

    def form_valid(self, form) -> HttpResponse:
        token_name = self.object.name
        response = super().form_valid(form)
        messages.success(self.request, gettext("API key %(name)s deleted.") % {"name": token_name})
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        detail_url = reverse("token_detail", kwargs={"id": self.object.id})
        return super().get_context_data(
            confirmation=deletion(self.object),
            activity=key_report(self.object, timezone.now()).totals,
            cancel_url=detail_url,
            trail=[
                Crumb(gettext("API keys"), reverse("tokens")),
                Crumb(self.object.name, detail_url, mono=True),
            ],
            **kwargs,
        )


@method_decorator(require_POST, name="dispatch")
class TokenRegenerateView(BaseAqueductView, View):
    http_method_names: ClassVar[list] = ["post"]

    def post(self, request, *args, **kwargs) -> HttpResponse:
        token = get_object_or_404(self.viewer.keys(KeyAct.MANAGE), pk=kwargs.get("id"))
        show_new_key(request, token.id, token.regenerate_key())
        return redirect("token_detail", id=token.id)
