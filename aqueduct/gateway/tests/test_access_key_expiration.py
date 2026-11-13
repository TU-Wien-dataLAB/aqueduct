import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Token


class AccessKeyExpirationAuthenticationTests(GatewayIntegrationTestCase):
    @override_settings(
        AUTHENTICATION_BACKENDS=[
            "django.contrib.auth.backends.ModelBackend",
            "gateway.authentication.TokenAuthenticationBackend",
        ]
    )
    def test_expired_bearer_key_is_rejected_even_with_authenticated_session(self):
        token = Token.objects.get(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN))
        token.expires_at = timezone.now() - timedelta(seconds=1)
        token.save(update_fields=["expires_at"])
        self.client.force_login(get_user_model().objects.get(pk=token.user_id))
        response = self.client.post(
            reverse("gateway:embeddings"),
            data=json.dumps({"model": "text-embedding-ada-002", "input": "hello"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 401)
