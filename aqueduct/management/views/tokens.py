from collections.abc import Iterable
from datetime import timedelta
from typing import ClassVar

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.messages import get_messages
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Count, Max, Q, QuerySet
from django.db.models.functions import TruncDate
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.formats import date_format
from django.utils.http import urlencode
from django.views import View
from django.views.decorators.http import require_POST
from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    TemplateView,
    UpdateView,
)

from gateway.config import get_router_config
from gateway.rate_limiting import get_per_token_usage
from management.forms import TokenCreateForm
from management.models import LimitSet, ServiceAccount, Token, UserProfile
from management.views.base import BaseAqueductView
from management.views.usage import model_rows, show_requests, spark, usage_charts, usage_series

NEW_KEY_TAG = "token-regenerated-key"
"""The message tag that carries a new key's secret to the next page, which shows it once."""


def add_week(profile: UserProfile, tokens: Iterable[Token]) -> None:
    """Give each key `week`, a spark of its requests on each of the last 7 days."""
    days = [timezone.localdate() - timedelta(days=n) for n in range(6, -1, -1)]
    tokens = list(tokens)
    counts = {
        (row["token_id"], row["day"]): row["n"]
        for row in profile.visible_requests()
        .filter(token__in=tokens, timestamp__date__gte=days[0])
        .values("token_id", day=TruncDate("timestamp"))
        .annotate(n=Count("id"))
        .order_by()
    }
    for token in tokens:
        token.week = spark(  # type: ignore[attr-defined]
            [(date_format(day, "D j M"), counts.get((token.id, day), 0)) for day in days],
            show_requests,
        )


class UserTokensView(BaseAqueductView, TemplateView):
    template_name = "management/tokens.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile  # Provided by BaseAqueductView

        tokens = profile.own_tokens().annotate(last_used=Max("requests__timestamp"))
        teams = profile.member_teams()
        service_accounts = (
            ServiceAccount.objects.filter(team__in=teams)
            .select_related("team__org", "token")
            .annotate(last_used=Max("token__requests__timestamp"))
        )

        max_tokens = getattr(settings, "MAX_USER_TOKENS", 3)
        can_add_token = tokens.count() < max_tokens

        # Effective limits for user-owned tokens (UserProfile + Org fallback)
        user_limits: LimitSet = LimitSet.from_objects(profile, [profile.org])

        # Attach each token's current rate-limit usage (per minute/hour/day bucket)
        # to the token object so templates can render it. All tokens are fetched in a
        # single ``cache.get_many`` call to avoid one cache lookup per token.
        per_token_usage = get_per_token_usage([token.id for token in tokens])
        for token in tokens:
            token.usage = per_token_usage[token.id]  # type: ignore[attr-defined]

        # Effective limits per service account (Team + Org fallback).
        # Computed inline from already-select_related data to avoid the
        # per-SA DB query that sa.token.get_limit() would trigger.
        per_sa_token_usage = get_per_token_usage([sa.token.id for sa in service_accounts])
        for sa in service_accounts:
            sa.effective_limits = LimitSet.from_objects(sa.team, [sa.team.org])  # type: ignore[attr-defined]
            sa.usage = per_sa_token_usage[sa.token.id]  # type: ignore[attr-defined]
        add_week(profile, [*tokens, *(sa.token for sa in service_accounts)])

        context.update(
            {
                "tokens": tokens,
                "service_accounts": service_accounts,
                "can_add_token": can_add_token,
                "max_tokens": max_tokens,
                "user_limits": user_limits,
            }
        )
        if can_add_token and not tokens:
            try:
                models = get_router_config().get("model_list", [])
            except RuntimeError:  # Without a router config the example names no model.
                models = []
            context["first_key_form"] = TokenCreateForm(
                user=self.request.user, initial={"name": "my-first-key"}
            )
            modes = [(m.get("model_info") or {}).get("mode") or "chat" for m in models]
            context["first_model"] = models[modes.index("chat")] if "chat" in modes else None
        return context


class TokenCreateView(BaseAqueductView, CreateView):
    model = Token
    form_class = TokenCreateForm
    template_name = "management/create/token.html"
    success_url = reverse_lazy("tokens")  # Redirect to tokens list

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

        # Use the returned secret_key in the success message
        messages.success(self.request, f"API key '{self.object.name}' created successfully.")
        messages.info(self.request, f"{secret_key}", extra_tags=NEW_KEY_TAG)

        return redirect("token_detail", id=self.object.id)

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = "Create New API Key"
        context["tz"] = settings.TIME_ZONE
        return context


class AllKeysView(BaseAqueductView, ListView):
    """Every key of the org the page shows, for its admins, with owner and last use."""

    template_name = "management/all_keys.html"
    context_object_name = "keys"
    paginate_by = 50

    def get_queryset(self) -> QuerySet[Token]:
        if not self.profile.is_org_admin(self.org):
            raise PermissionDenied
        keys = self.profile.visible_tokens(self.org).select_related("service_account__team", "user")
        query = self.request.GET.get("q", "").strip()
        if query:
            keys = keys.filter(
                Q(name__icontains=query)
                | Q(user__email__icontains=query)
                | Q(service_account__team__name__icontains=query)
            )
        return keys.annotate(last_used=Max("requests__timestamp")).order_by("name", "id")

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(query=self.request.GET.get("q", "").strip(), **kwargs)
        add_week(self.profile, context["keys"])
        return context


class TokenDetailView(BaseAqueductView, DetailView):
    """One key: who owns it, its use in the last 7 days, and its limits and current use."""

    template_name = "management/token_detail.html"
    pk_url_kwarg = "id"
    context_object_name = "token"

    def get_queryset(self) -> QuerySet[Token]:
        return self.profile.visible_tokens().select_related("service_account__team", "user")

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        token: Token = self.object
        limits = token.get_limits()
        usage = get_per_token_usage([token.id])[token.id]
        # The last 7 days, drawn as the Usage page draws them.
        now = timezone.now()
        start = now - timedelta(days=7)
        week = self.profile.visible_requests().filter(token=token, timestamp__gte=start)
        log = f"{reverse('requests')}?key={token.pk}"
        context.update(
            charts=usage_charts(usage_series(week, "day", start, now), "day"),
            per_model=model_rows(
                week, lambda model: f"{log}&span=1w&{urlencode({'model': model})}"
            ),
            log_url=log,
            can_manage=self.profile.manageable_tokens().filter(pk=token.pk).exists(),
            team=token.service_account.team if token.service_account else None,
            own=token.user_id == self.request.user.id and token.service_account is None,
            limit_rows=[
                {
                    "window": name,
                    "requests": (usage[name]["req"], requests),
                    "input_tokens": (usage[name]["in"], input_tokens),
                    "output_tokens": (usage[name]["out"], output_tokens),
                }
                for name, _seconds, requests, input_tokens, output_tokens in limits.windows()
            ],
            concurrent_limit=limits.concurrent_requests_limit,
            last_used=token.requests.aggregate(last=Max("timestamp"))["last"],
            new_key=next(
                (
                    str(message.message)
                    for message in get_messages(self.request)
                    if message.extra_tags == NEW_KEY_TAG
                ),
                None,
            ),
        )
        return context


class TokenEditView(BaseAqueductView, UpdateView):
    model = Token
    form_class = TokenCreateForm
    template_name = "management/edit/token.html"
    pk_url_kwarg = "id"
    success_url = reverse_lazy("tokens")

    def get_queryset(self) -> QuerySet:
        return self.profile.own_tokens()

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = "Edit API Key"
        context["cancel_url"] = self.get_success_url()
        return context


class TokenDeleteView(BaseAqueductView, DeleteView):
    model = Token
    pk_url_kwarg = "id"
    template_name = "management/common/confirm_delete.html"
    success_url = reverse_lazy("tokens")

    def get_queryset(self) -> QuerySet:
        # Only allow deleting tokens owned by the user and not associated with a service account
        return self.profile.own_tokens()

    def form_valid(self, form) -> HttpResponse:
        token_name = self.object.name
        response = super().form_valid(form)
        messages.success(self.request, f"API key '{token_name}' deleted successfully.")
        return response

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["object_type_name"] = "API Key"
        context["object_name"] = str(self.object)
        context["cancel_url"] = self.get_success_url()
        return context


@method_decorator(require_POST, name="dispatch")
class TokenRegenerateView(BaseAqueductView, View):
    """
    Handles regeneration of a token key (both user and service account tokens).
    Requires POST request.
    Permissions are checked based on token type.
    """

    http_method_names: ClassVar[list] = ["post"]

    def post(self, request, *args, **kwargs) -> HttpResponse:
        token_id = kwargs.get("id")
        profile = self.profile
        user = request.user

        try:
            token = get_object_or_404(
                Token.objects.select_related("service_account__team", "user"), pk=token_id
            )
            token_name = token.name
            is_service_account_token = token.service_account is not None

            # --- Permission Check --- #
            can_regenerate = False
            redirect_url = reverse_lazy("tokens")  # Default redirect

            if is_service_account_token:
                # Check if user is admin of the SA's team
                team = token.service_account.team
                redirect_url = reverse("team", kwargs={"id": team.id})
                if profile.is_team_admin(team):
                    can_regenerate = True
                else:
                    messages.error(
                        request,
                        f"You do not have permission to regenerate the token for service account "
                        f"'{token.service_account.name}'.",
                    )
            # Check if user owns the token
            elif token.user == user:
                can_regenerate = True
            else:
                # This case shouldn't happen with correct view/URL setup, but good to have
                messages.error(request, "You do not have permission to regenerate this token.")

            # --- Regeneration --- #
            if can_regenerate:
                redirect_url = reverse("token_detail", kwargs={"id": token.id})
                new_key = token.regenerate_key()  # Assumes regenerate_key() saves the token
                messages.success(
                    request,
                    f"API key '{token_name}' has been regenerated. "
                    f"Please update any clients using the old key.",
                )
                messages.info(
                    request,
                    f"{new_key}",  # Only include the key itself in the message
                    extra_tags=NEW_KEY_TAG,
                )
            # Else: error message already added during permission check

        except Token.DoesNotExist:
            messages.error(request, "The specified token could not be found.")
            # Redirect to a default page if token not found
            redirect_url = reverse_lazy("tokens")
        except Exception as e:
            messages.error(
                request, f"An unexpected error occurred while regenerating the token: {e}"
            )
            # Try to determine redirect URL based on what we know, default if necessary
            try:
                # Attempt to get redirect URL again if token was fetched
                if is_service_account_token and team:
                    redirect_url = reverse("team", kwargs={"id": team.id})
            except NameError:  # If token fetch failed earlier
                redirect_url = reverse_lazy("tokens")

        return redirect(redirect_url)
