from copy import deepcopy

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from gateway.router import get_router_config


class ModelListView(LoginRequiredMixin, TemplateView):
    """
    Displays a list of models from the Pydantic RouterConfig.
    """

    template_name = "management/model_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            model_list = deepcopy(get_router_config())["model_list"]
        except KeyError:
            model_list = []
        else:
            for model in model_list:
                litellm_params: dict = model.get("litellm_params", {})
                if not litellm_params.get("api_key", "").startswith("os.environ/"):
                    litellm_params["api_key"] = "*********"

        context["title"] = "Models"
        context["model_list"] = model_list
        return context
