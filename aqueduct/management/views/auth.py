from django.views.generic import TemplateView
from django.shortcuts import redirect
from django.conf import settings

class SSOTemplateView(TemplateView):
    """Displays the login prompt, redirecting authenticated users."""
    template_name = 'management/login.html' # Adjust path if needed

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            # Redirect logged-in users away from the login page
            # Redirect to 'org' view assuming it's the main landing page after login
            return redirect('org')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Pass OIDC provider name (or default) to template for button text etc.
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context