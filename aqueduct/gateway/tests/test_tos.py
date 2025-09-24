import json
from unittest.mock import patch

from django.apps import apps
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.cache import caches
from django.test import override_settings

from gateway.tests.utils.base import TOSGatewayTestCase
from gateway.tests.utils import _build_chat_headers


class TOSTestCase(TOSGatewayTestCase):
    def test_tos_accepted(self):
        """
        Test that when a user has accepted the TOS, they can access the /models endpoint.
        """
        # Accept TOS for non-admin user
        UPDATED_ACCESS_TOKEN, user_id = self.create_new_user()
        self.accept_tos(user_id=user_id)

        with patch('gateway.views.decorators.cache', caches['default']):
            # Call the /models endpoint
            response = self.client.get(
                "/models",
                data='',
                content_type="application/json",
                headers=_build_chat_headers(UPDATED_ACCESS_TOKEN)
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

        # non-admin user
        UPDATED_ACCESS_TOKEN, _ = self.create_new_user()

        # Create an active Terms of Service but DON'T create a UserAgreement for the user
        TermsOfService.objects.create(
            active=True,
            content="Test Terms of Service content"
        )

        with patch('gateway.views.decorators.cache', caches['default']):
            # Call the /models endpoint - user should be blocked because they haven't accepted TOS
            response = self.client.get(
                "/models",
                data='',
                content_type="application/json",
                headers=_build_chat_headers(UPDATED_ACCESS_TOKEN)
            )

        # Should return 403 Forbidden
        self.assertEqual(response.status_code, 403,
                         f"Expected 403 Forbidden, got {response.status_code}: {response.content}")

        # Verify the response contains an error message about TOS
        response_json = response.json()
        self.assertIn("error", response_json)
        self.assertIn("terms of service", response_json["error"].lower())

    def test_tos_user_skip(self):
        """ Tests that admin users are skipped in the decorator check. """
        cache = caches['default']
        # set cache for user with id 1
        cache.set('django:tos:skip_tos_check:{}'.format(1), True)

        from tos.models import TermsOfService

        TermsOfService.objects.create(
            active=True,
            content="Test Terms of Service content"
        )

        with patch('gateway.views.decorators.cache', caches['default']):
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

    @override_settings(TOS_GATEWAY_VALIDATION=False)
    def test_tos_gateway_validation_disabled(self):
        """
        Test that when a user has not accepted the TOS, but gateway validation is disabled, the request is handled.
        """
        from tos.models import TermsOfService

        # non-admin user
        UPDATED_ACCESS_TOKEN, _ = self.create_new_user()

        # Create an active Terms of Service but DON'T create a UserAgreement for the user
        TermsOfService.objects.create(
            active=True,
            content="Test Terms of Service content"
        )

        # Call the /models endpoint - user should be blocked because they haven't accepted TOS
        response = self.client.get(
            "/models",
            data='',
            content_type="application/json",
            headers=_build_chat_headers(UPDATED_ACCESS_TOKEN)
        )

        # Should return 403 Forbidden
        self.assertEqual(response.status_code, 200,
                         f"Expected 200 Forbidden, got {response.status_code}: {response.content}")

        # Verify the response contains model data
        response_json = response.json()
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
