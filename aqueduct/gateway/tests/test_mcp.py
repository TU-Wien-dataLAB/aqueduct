import json
from unittest.mock import MagicMock, patch

from gateway.tests.utils.base import GatewayMCPTestCase


class MCPSessionLifecycleTest(GatewayMCPTestCase):
    def test_mcp_session_lifecycle(self):
        server_name = "test-server"
        mcp_endpoint = f"/mcp-servers/{server_name}/mcp"

        mock_session = MagicMock()
        mock_session.session_id = "test-session-id"
        mock_session.terminated = False

        with (
            patch("gateway.views.mcp.session_manager.create_session") as mock_create,
            patch("gateway.views.mcp.session_manager.get_session") as mock_get,
            patch("gateway.views.mcp.session_manager.send_message") as mock_send,
            patch("gateway.views.mcp.session_manager.terminate_session") as mock_terminate,
        ):
            mock_create.return_value = "test-session-id"
            mock_get.return_value = mock_session

            response = self.client.get(
                mcp_endpoint,
                headers={
                    "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 OK for session creation, got {response.status_code}: {response.content}",
            )

            response_json = response.json()
            self.assertIn("session_id", response_json)
            self.assertIn("server_name", response_json)
            self.assertEqual(response_json["server_name"], server_name)
            session_id = response_json["session_id"]
            self.assertEqual(session_id, "test-session-id")

            mock_create.assert_called_once()

            message_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }

            response = self.client.post(
                mcp_endpoint,
                data=json.dumps(message_data),
                content_type="application/json",
                headers={
                    "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                    "X-MCP-Session-ID": session_id,
                },
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 OK for message send, got {response.status_code}: {response.content}",
            )

            response_json = response.json()
            self.assertEqual(response_json["status"], "message_sent")

            mock_get.assert_called()
            mock_send.assert_called_once()

            response = self.client.delete(
                mcp_endpoint,
                headers={
                    "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                    "X-MCP-Session-ID": session_id,
                },
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 OK for session termination, got {response.status_code}: {response.content}",
            )

            response_json = response.json()
            self.assertEqual(response_json["status"], "session_terminated")

            mock_terminate.assert_called_once_with(session_id)

    def test_mcp_session_create_invalid_server(self):
        mcp_endpoint = "/mcp-servers/nonexistent-server/mcp"

        response = self.client.get(
            mcp_endpoint,
            headers={
                "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
        )

        self.assertEqual(
            response.status_code,
            404,
            f"Expected 404 Not Found for invalid server, got {response.status_code}: {response.content}",
        )

    def test_mcp_session_post_without_session_id(self):
        server_name = "test-server"
        mcp_endpoint = f"/mcp-servers/{server_name}/mcp"

        message_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = self.client.post(
            mcp_endpoint,
            data=json.dumps(message_data),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
        )

        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400 Bad Request for missing session ID, got {response.status_code}: {response.content}",
        )

    def test_mcp_session_post_with_invalid_session(self):
        server_name = "test-server"
        mcp_endpoint = f"/mcp-servers/{server_name}/mcp"

        with patch("gateway.views.mcp.session_manager.get_session") as mock_get:
            mock_get.return_value = None

            message_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }

            response = self.client.post(
                mcp_endpoint,
                data=json.dumps(message_data),
                content_type="application/json",
                headers={
                    "Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
                    "X-MCP-Session-ID": "invalid-session-id",
                },
            )

            self.assertEqual(
                response.status_code,
                404,
                f"Expected 404 Not Found for invalid session, got {response.status_code}: {response.content}",
            )

    def test_mcp_session_delete_without_session_id(self):
        server_name = "test-server"
        mcp_endpoint = f"/mcp-servers/{server_name}/mcp"

        response = self.client.delete(
            mcp_endpoint, headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"}
        )

        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400 Bad Request for missing session ID, got {response.status_code}: {response.content}",
        )
