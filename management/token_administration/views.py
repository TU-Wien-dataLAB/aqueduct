from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.conf import settings
from django.views.generic import TemplateView
from django.shortcuts import redirect


class SSOTemplateView(TemplateView):
    template_name = 'token_administration/login.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            # Redirect the logged-in user to the desired page
            return redirect('/')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add setting with fallback default
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context


@login_required
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
