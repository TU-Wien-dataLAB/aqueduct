from django.views.generic import ListView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from gateway.config import get_router_config


class ModelListView(LoginRequiredMixin, TemplateView):
    """
    Displays a list of models from the Pydantic RouterConfig.
    """
    template_name = 'management/model_list.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        config = get_router_config()
        context['title'] = "Models"
        context['model_list'] = config["model_list"]
        return context