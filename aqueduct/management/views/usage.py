from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin


class UsageDashboardView(LoginRequiredMixin, TemplateView):
    """
    Dummy usage dashboard page.
    """
    template_name = 'management/usage.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Usage'
        return context