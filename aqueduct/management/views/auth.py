from django.conf import settings
from django.shortcuts import redirect
from django.urls import NoReverseMatch, reverse
from django.views.generic import TemplateView


class SSOTemplateView(TemplateView):
    template_name = "management/login.html"

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            next_url = request.GET.get("next")
            if next_url:
                return redirect(next_url)
            try:
                org_url = reverse("org")
                return redirect(org_url)
            except NoReverseMatch:
                return redirect(settings.LOGIN_REDIRECT_URL)
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["OIDC_PROVIDER"] = getattr(settings, "OIDC_PROVIDER", "SSO")
        return context
