from urllib.parse import urlsplit

from django.conf import settings
from django.contrib import admin
from django.http import HttpResponseBase
from django.shortcuts import redirect
from django.urls import NoReverseMatch, reverse
from django.views.generic import TemplateView


class SSOTemplateView(TemplateView):
    template_name = "management/login.html"

    def dispatch(self, request, *args, **kwargs) -> HttpResponseBase:
        if request.user.is_authenticated:
            next_url = request.GET.get("next")
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
