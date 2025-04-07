from django.http import HttpResponse
from django.conf import settings
from django.views.generic import TemplateView


class SSOTemplateView(TemplateView):
    template_name = 'token_administration/sso.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add setting with fallback default
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
