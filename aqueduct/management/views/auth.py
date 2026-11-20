from urllib.parse import urlsplit

from django.conf import settings
from django.contrib import admin
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import DisallowedRedirect
from django.http import HttpRequest, HttpResponseBase, HttpResponseRedirect
from django.shortcuts import redirect
from django.urls import NoReverseMatch, reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.cache import never_cache
from django.views.generic import TemplateView
from tos import views as tos_views


def is_safe(request: HttpRequest, url: str | None) -> bool:
    return bool(url) and url_has_allowed_host_and_scheme(
        url, allowed_hosts={request.get_host()}, require_https=request.is_secure()
    )


def safe_next(request: HttpRequest) -> str | None:
    next_url = request.GET.get("next")
    return next_url if is_safe(request, next_url) else None


@never_cache
def check_tos(request: HttpRequest) -> HttpResponseBase:
    """django-tos checks `next` only for `//`, so `http:evil.example` passes its check."""
    try:
        response = tos_views.check_tos(request)
    except DisallowedRedirect:
        return redirect(settings.LOGIN_REDIRECT_URL)
    next_url = request.POST.get(REDIRECT_FIELD_NAME, request.GET.get(REDIRECT_FIELD_NAME))
    if isinstance(response, HttpResponseRedirect) and not is_safe(request, next_url):
        return redirect(settings.LOGIN_REDIRECT_URL)
    return response


class SSOTemplateView(TemplateView):
    template_name = "management/login.html"

    def dispatch(self, request, *args, **kwargs) -> HttpResponseBase:
        if request.user.is_authenticated:
            next_url = safe_next(request)
            if next_url:
                admin_path = reverse("admin:index").rstrip("/")
                next_path = urlsplit(next_url).path
                targets_admin = next_path == admin_path or next_path.startswith(admin_path + "/")
                # Authenticated users without admin access would otherwise bounce
                # between the admin and this login view indefinitely.
                if not targets_admin or admin.site.has_permission(request):
                    return redirect(next_url)
            try:
                org_url = reverse("org")
                return redirect(org_url)
            except NoReverseMatch:
                return redirect(settings.LOGIN_REDIRECT_URL)
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["OIDC_PROVIDER"] = getattr(settings, "OIDC_PROVIDER", "SSO")
        return context
