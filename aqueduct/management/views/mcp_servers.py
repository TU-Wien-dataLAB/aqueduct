import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from django.urls import reverse
from django.views.generic import TemplateView

from gateway.config import MCPServerConfig, get_mcp_config
from management.access import TeamAct
from management.model_catalog import Availability, Scope, scopes_for, split_scopes
from management.models import UserProfile
from management.models.exclusions import Exclusion
from management.views.base import BaseAqueductView


@dataclass(frozen=True)
class MCPServerCard(Availability):
    name: str
    description: str
    icon_url: str | None
    url: str

    @property
    def initial(self) -> str:
        return self.name[:1].upper()


def server_cards(
    config: dict[str, MCPServerConfig], base_url: str, scopes: Sequence[Scope]
) -> tuple[MCPServerCard, ...]:
    def card(name: str, server: MCPServerConfig) -> MCPServerCard:
        off_for, on_for = split_scopes(name, scopes)
        path = reverse("gateway:mcp_server", kwargs={"name": name})
        return MCPServerCard(
            off_for=off_for,
            on_for=on_for,
            name=name,
            description=server.get("description") or "",
            icon_url=server.get("icon_url"),
            url=f"{base_url}{path}",
        )

    cards = (card(name, server) for name, server in config.items())
    return tuple(server for server in cards if server.usable)


def client_config(cards: Sequence[MCPServerCard]) -> str:
    servers = {
        card.name: {"url": card.url, "headers": {"Authorization": "Bearer sk-..."}}
        for card in cards
    }
    return json.dumps({"mcpServers": servers}, indent=2)


class MCPServersView(BaseAqueductView, TemplateView):
    template_name = "management/mcp_servers.html"

    def scopes(self) -> tuple[Scope, ...]:
        try:
            viewer = self.viewer
            return scopes_for(viewer.profile, viewer.teams(TeamAct.MINE), Exclusion.MCP_SERVERS)
        except UserProfile.DoesNotExist:
            return ()

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        try:
            config = get_mcp_config()
        except RuntimeError as e:
            context.update(servers=(), error=str(e))
            return context
        base_url = f"{self.request.scheme}://{self.request.get_host()}"
        servers = server_cards(config, base_url, self.scopes())
        context.update(servers=servers, client_config=client_config(servers) if servers else None)
        return context
