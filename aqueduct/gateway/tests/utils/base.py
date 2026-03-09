import json
import os
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from litellm import Router
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import Deployment
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextCompletionResponse,
)
from tos.models import TermsOfService, UserAgreement

from gateway.tests.utils import _build_chat_headers, _build_chat_payload
from gateway.tests.utils.test_runner import get_shared_mock_server
from management.models import Org, Token, UserProfile
from mock_api.mock_server import MockAPIServer

INTEGRATION_TEST_BACKEND: Literal["vllm", "openai"] = os.environ.get(
    "INTEGRATION_TEST_BACKEND", "openai"
)
if INTEGRATION_TEST_BACKEND not in ["vllm", "openai"]:
    raise ValueError("Integration test backend must be one of 'vllm' or 'openai'.")

ROOT = Path(__file__).parent.parent.parent.parent.parent

ROUTER_CONFIG_PATH = str(ROOT / "example_router_config.yaml")
with open(ROUTER_CONFIG_PATH) as f:
    ROUTER_CONFIG = f.read()

User = get_user_model()


def get_mock_router(model: str = "test-model"):
    router = MagicMock(spec=Router)
    router.acompletion = AsyncMock(return_value=ModelResponse())
    router.atext_completion = AsyncMock(return_value=TextCompletionResponse())
    router.aembedding = AsyncMock(return_value=EmbeddingResponse())
    router.image_generation = MagicMock(return_value=ImageResponse())
    router.aspeech = AsyncMock(return_value=HttpxBinaryResponseContent(response=MagicMock()))
    router.get_deployment = MagicMock(
        return_value=Deployment("test-model", {"model": f"openai/{model}"})
    )
    return router


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    API_MAX_RETRIES=1,  # for some reason, in a few tests the 1st request to the mock API fails (503)
)
class GatewayIntegrationTestCase(TestCase):
    """
    Integration tests for gateway endpoints, using the mock OpenAI server.
    """

    model = "gpt-4.1-nano"

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"

    fixtures = ["gateway_data.json"]
    mock_server: MockAPIServer = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if INTEGRATION_TEST_BACKEND == "openai" and not settings.TESTS_USE_MOCK_API:
            # When running tests against the real OpenAI API, the API key has to be set
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable has to be set for OpenAI integration."
                )

        if settings.TESTS_USE_MOCK_API:
            # Mock all requests to the external OpenAI API
            cls.mock_server = get_shared_mock_server()
            # OpenAI's AsyncClient first tries to get the base url and API key from the router
            # config, and only falls back to env variables if they are not set there.
            # The patching is not strictly necessary if the router config defines these values,
            # but it's here as a safety measure.
            cls._patcher = patch.dict(
                "os.environ",
                {"OPENAI_BASE_URL": cls.mock_server.base_url, "OPENAI_API_KEY": "fake_openai_key"},
            )
            cls._patcher.start()

            # We want to override the files API url for all tests.
            settings.AQUEDUCT_FILES_API_URL = cls.mock_server.base_url
            settings.AQUEDUCT_FILES_API_KEY = "test_key"

        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_patcher") and cls._patcher:
            cls._patcher.stop()
        super().tearDownClass()

    @staticmethod
    def create_new_user() -> tuple[str, int]:
        # Create a new user and a new token for that user
        new_user = User.objects.create_user(username="OtherUser", email="other@example.com")

        new_user.groups.add(Group.objects.get(name="user"))
        org = Org.objects.get(name="E060")
        _ = UserProfile.objects.create(user=new_user, org=org)

        # Create a new token for the different user
        new_token = Token(name="TestToken", user=new_user)
        token_value = new_token._set_new_key()
        new_token.save()
        return token_value, new_user.id


class GatewayFilesTestCase(GatewayIntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Prepare auth headers for file API
        headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)
        # Remove Content-Type header to allow multipart file upload
        headers.pop("Content-Type", None)
        cls.headers = headers
        cls.url_files = reverse("gateway:files")

    def setUp(self):
        super().setUp()
        # Clear the cached client so it picks up the test settings (override_settings)
        from gateway.config import get_files_api_client

        get_files_api_client.cache_clear()

    def tearDown(self):
        # Clear the cached client so it doesn't affect other tests
        from gateway.config import get_files_api_client

        get_files_api_client.cache_clear()
        super().tearDown()


@override_settings(
    MAX_USER_BATCHES=3,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB=1000000,  # Limit set high to avoid conflicts
)
class GatewayBatchesTestCase(GatewayIntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers.pop("Content-Type", None)
        cls.url_files = reverse("gateway:files")
        cls.url_chat = reverse("gateway:v1_chat_completions")

    def _make_jsonl_content(self) -> bytes:
        """Create valid JSONL content for batch upload."""
        payload = _build_chat_payload(
            self.model,
            messages=[{"role": "system", "content": "Hi"}, {"role": "user", "content": "Test"}],
        )
        wrapped = {"custom_id": "1", "method": "POST", "url": self.url_chat, "body": payload}
        return json.dumps(wrapped).encode() + b"\n"

    def _create_jsonl_file(self, name: str | None = "testfile", headers: dict | None = None) -> str:
        """Prepare a simple JSONL file in batch API format.

        `name` is the name of the SimpleUploadedFile (without the file extension).
        `headers` contain the secret of the token under which the file should be created.
        If not provided, the headers with the default token's secret will be used.
        """
        if not headers:
            headers = self.headers

        content = self._make_jsonl_content()
        f = SimpleUploadedFile(f"{name}.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=headers)
        return resp.json()["id"]


class GatewayTTSSTTestCase(GatewayIntegrationTestCase):
    fixtures = ["gateway_data.json"]
    tts_model = "gpt-4o-mini-tts"
    stt_model = "whisper-1"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)

        # Remove Content-Type header to allow multipart file upload
        multipart_headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)
        multipart_headers.pop("Content-Type", None)
        cls.multipart_headers = multipart_headers

        cls.url_tts = reverse("gateway:speech")
        cls.url_stt = reverse("gateway:transcriptions")


@override_settings(TOS_ENABLED=True)
class TOSGatewayTestCase(GatewayIntegrationTestCase):
    fixtures = ["gateway_data.json"]

    def accept_tos(self, user_id: int = 1):
        # Create an active Terms of Service
        tos = TermsOfService.objects.create(active=True, content="Test Terms of Service content")

        # Get the user with pk=user_id and create a UserAgreement
        user = User.objects.get(pk=user_id)
        UserAgreement.objects.create(terms_of_service=tos, user=user)
