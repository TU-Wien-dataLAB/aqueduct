import os
import sys
from typing import Optional, Literal

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

INTEGRATION_TEST_BACKEND: Literal["vllm", "openai"] = os.environ.get("INTEGRATION_TEST_BACKEND", "vllm")
if INTEGRATION_TEST_BACKEND not in ["vllm", "openai"]:
    raise ValueError("Integration test backend must be one of 'vllm' or 'openai‘.")

START_VLLM_SERVER = INTEGRATION_TEST_BACKEND == "vllm" and os.environ.get("START_VLLM_SERVER", "true") == "true"

# --- vLLM Internal Imports ---
RemoteOpenAIServer = None
_VLLM_AVAILABLE = False
if START_VLLM_SERVER:
    try:
        from gateway.tests.utils.vllm import get_openai_server, stop_openai_server

        _VLLM_AVAILABLE = True
    except ImportError as e:
        get_openai_server, stop_openai_server = None, None
        _VLLM_IMPORT_ERROR = e

# Import Org, Team, UserProfile for direct DB manipulation
from management.models import Request

User = get_user_model()


# --- Django Test Class ---
@override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
class GatewayIntegrationTestCase(TransactionTestCase):
    """
    Integration tests using the embedded RemoteOpenAIServer (with httpx).
    """
    vllm_server: Optional["RemoteOpenAIServer"] = None
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "gpt-4o-mini"

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"
    AQUEDUCT_ENDPOINT = "vllm" if INTEGRATION_TEST_BACKEND == "vllm" else "openai"

    VLLM_SEED = 42
    fixtures = ["gateway_data.json", "vllm_endpoint.json", "openai_endpoint.json"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if INTEGRATION_TEST_BACKEND == "vllm" and START_VLLM_SERVER:
            if not _VLLM_AVAILABLE:
                raise RuntimeError(
                    f"vLLM integration tests require vllm to be installed. "
                    f"Import error: {_VLLM_IMPORT_ERROR}"
                )
            print(f"\nStarting vLLM server for {cls.__name__}...")
            try:
                MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
                vllm_args = ["--host", "0.0.0.0", "--port", "8009"]
                cls.vllm_server = get_openai_server(
                    model_name=MODEL_NAME,
                    vllm_serve_args=vllm_args,
                    seed=cls.VLLM_SEED,
                    auto_port=False,
                    max_wait_seconds=300
                )
                print(f"vLLM server started on: {cls.vllm_server.url_root}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"ERROR starting vLLM server: {e}", file=sys.stderr)
                cls.vllm_server = None
                raise AssertionError(f"Failed to set up vLLM server: {e}") from e
        elif INTEGRATION_TEST_BACKEND == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY environment variable has to be set for OpenAI integration.")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self):
        # Clear Request table before test
        Request.objects.all().delete()
