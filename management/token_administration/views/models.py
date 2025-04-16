from django.views.generic import ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from ..models import Endpoint

class ModelListView(LoginRequiredMixin, ListView):
    """
    Displays a list of Endpoints and the Models associated with each.
    """
    model = Endpoint
    template_name = 'token_administration/model_list.html'
    context_object_name = 'endpoints'  # Use 'endpoints' in the template

    def get_queryset(self):
        """
        Optimize query by prefetching related models.
        """
        return Endpoint.objects.prefetch_related('models').order_by('name')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Models and Endpoints"
        return context 