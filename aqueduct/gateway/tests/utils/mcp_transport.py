"""Controlled upstream transport for MCP relay tests."""

from contextlib import contextmanager
from unittest.mock import patch

import httpx


@contextmanager
def upstream(handler):
    """Use real HTTP clients with a controlled external transport and track ownership."""
    clients = []
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    with patch("gateway.views.mcp.httpx.AsyncClient", new=factory):
        yield clients
