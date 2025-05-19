from django.urls import NoReverseMatch, reverse
from django.views.generic import TemplateView
from django.shortcuts import redirect
from django.conf import settings

class SSOTemplateView(TemplateView):
    template_name = 'management/login.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            next_url = request.GET.get('next')
            # If 'next' is provided and it's for the admin, redirect there
            # You might need a more robust way to check if next_url is an admin path
            if next_url and next_url.startswith(reverse('admin:index')): # Or resolve_url(admin:index)
                return redirect(next_url)
            # Otherwise, redirect to the default 'org' page
            try:
                org_url = reverse('org') # Make sure 'org' is a valid named URL
                return redirect(org_url)
            except NoReverseMatch:
                # Fallback if 'org' isn't defined, or handle error
                return redirect(settings.LOGIN_REDIRECT_URL) # Default redirect

        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context