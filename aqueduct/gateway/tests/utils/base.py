import os
import shutil
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TransactionTestCase, override_settings
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

from gateway.tests.utils import _build_chat_headers
from management.models import Org, Token, UserProfile

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

TEST_FILES_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files_root")
os.makedirs(TEST_FILES_ROOT, exist_ok=True)


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
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    API_MAX_RETRIES=5,  # for some reason the OpenAI API fails with 503 sometimes...
)
class GatewayIntegrationTestCase(TransactionTestCase):
    """
    Integration tests using the embedded RemoteOpenAIServer (with httpx).
    """

    model = "gpt-4.1-nano"

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"

    fixtures = ["gateway_data.json"]

    @classmethod
    def _write_router_config(cls):
        os.makedirs(os.path.dirname(ROUTER_CONFIG_PATH), exist_ok=True)
        with open(ROUTER_CONFIG_PATH, "w") as f:
            f.write(ROUTER_CONFIG)

    @classmethod
    def setUpClass(cls):
        cls._write_router_config()
        super().setUpClass()
        if INTEGRATION_TEST_BACKEND == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable has to be set for OpenAI integration."
                )
        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)

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


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    API_MAX_RETRIES=5,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    AQUEDUCT_FILES_API_URL="https://api.openai.com",
    AQUEDUCT_FILES_API_KEY=os.environ.get("OPENAI_API_KEY"),
)
class GatewayFilesTestCase(GatewayIntegrationTestCase):
    # Load default fixture (includes test Token) and set test access token
    fixtures = ["gateway_data.json"]

    @classmethod
    def setUpClass(cls):
        cls._write_router_config()
        super().setUpClass()
        if INTEGRATION_TEST_BACKEND == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable has to be set for OpenAI integration."
                )
        # Prepare auth headers for file API
        headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)
        # Remove Content-Type header to allow multipart file upload
        headers.pop("Content-Type", None)
        cls.headers = headers
        cls.url_files = reverse("gateway:files")

    def tearDown(self):
        super().tearDown()


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_FILES_API_URL="https://api.openai.com",
    AQUEDUCT_FILES_API_KEY=os.environ.get("OPENAI_API_KEY"),
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    MAX_USER_BATCHES=3,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class GatewayBatchesTestCase(GatewayIntegrationTestCase):
    def tearDown(self):
        super().tearDown()
        # Clean up the file storage directory after each test
        if os.path.exists(TEST_FILES_ROOT):
            shutil.rmtree(TEST_FILES_ROOT)
            os.makedirs(TEST_FILES_ROOT, exist_ok=True)


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
)
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


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    TOS_ENABLED=True,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class TOSGatewayTestCase(GatewayIntegrationTestCase):
    fixtures = ["gateway_data.json"]

    def accept_tos(self, user_id: int = 1):
        # Create an active Terms of Service
        tos = TermsOfService.objects.create(active=True, content="Test Terms of Service content")

        # Get the user with pk=user_id and create a UserAgreement
        user = User.objects.get(pk=user_id)
        UserAgreement.objects.create(terms_of_service=tos, user=user)
