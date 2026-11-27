from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

from gateway.config import get_mcp_config


class MCPServersView(LoginRequiredMixin, TemplateView):
    """
    Displays a list of MCP servers from the configuration.
    """

    template_name = "management/mcp_servers.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        try:
            context["mcp_servers"] = get_mcp_config()
            # Add the current site's domain for constructing full URLs
            context["site_host"] = f"{self.request.scheme}://{self.request.get_host()}"
        except RuntimeError as e:
            context["mcp_servers"] = {}
            context["error"] = str(e)

        return context
