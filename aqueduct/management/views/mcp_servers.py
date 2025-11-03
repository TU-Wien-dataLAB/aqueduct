from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

from gateway.config import get_mcp_config


class MCPServersView(LoginRequiredMixin, TemplateView):
    """
    Displays a list of MCP servers from the configuration.
    """

    template_name = "management/mcp_servers.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            mcp_config = get_mcp_config()
            # Add default icon_url fallback for servers that don't have it
            for server_name, server_config in mcp_config.items():
                if "icon_url" not in server_config:
                    server_config["icon_url"] = "/static/icons/mcp.svg"
            context["mcp_servers"] = mcp_config
            # Add the current site's domain for constructing full URLs
            context["site_host"] = f"{self.request.scheme}://{self.request.get_host()}"
        except RuntimeError as e:
            context["mcp_servers"] = {}
            context["error"] = str(e)

        return context
