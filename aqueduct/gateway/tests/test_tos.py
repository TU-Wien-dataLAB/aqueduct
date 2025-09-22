import json
from gateway.tests.utils.base import TOSGatewayTestCase
from gateway.tests.utils import _build_chat_headers


class TOSTestCase(TOSGatewayTestCase):
    def test_tos_accepted(self):
        """
        Test that when a user has accepted the TOS, they can access the /models endpoint.
        """
        # Accept TOS for the user
        self.accept_tos()
        
        # Call the /models endpoint
        response = self.client.get(
            "/models",
            data='',
            content_type="application/json",
            headers=_build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        )
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")
        
        # Verify the response contains model data
        response_json = response.json()
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)

    def test_tos_rejected(self):
        """
        Test that when a user has not accepted the TOS, they get a 403 error when accessing the /models endpoint.
        """
        from tos.models import TermsOfService
        
        # Create an active Terms of Service but DON'T create a UserAgreement for the user
        tos = TermsOfService.objects.create(
            active=True,
            content="Test Terms of Service content"
        )
        
        # Call the /models endpoint - user should be blocked because they haven't accepted TOS
        response = self.client.get(
            "/models",
            data='',
            content_type="application/json",
            headers=_build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        )
        
        # Should return 403 Forbidden
        self.assertEqual(response.status_code, 403, f"Expected 403 Forbidden, got {response.status_code}: {response.content}")
        
        # Verify the response contains an error message about TOS
        response_json = response.json()
        self.assertIn("error", response_json)
        self.assertIn("terms of service", response_json["error"].lower())
