import os
import shutil
import sys
from textwrap import dedent
from typing import Optional, Literal

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

INTEGRATION_TEST_BACKEND: Literal["vllm", "openai"] = os.environ.get("INTEGRATION_TEST_BACKEND", "vllm")
if INTEGRATION_TEST_BACKEND not in ["vllm", "openai"]:
    raise ValueError("Integration test backend must be one of 'vllm' or 'openai‘.")

ROUTER_CONFIG_PATH = f"/tmp/aqueduct/{INTEGRATION_TEST_BACKEND}-router-config.yaml"
ROUTER_CONFIG_VLLM = dedent("""
model_list:
- model_name: Qwen-0.5B
  litellm_params:
    model: openai/Qwen/Qwen2.5-0.5B-Instruct
    api_key: "dummy-vllm-key"
    api_base: http://localhost:8009/v1
  tpm: 10000
  rpm: 100
""")

ROUTER_CONFIG_OPENAI = dedent("""
model_list:
- model_name: gpt-4.1-nano
  litellm_params:
    model: openai/gpt-4.1-nano
    api_key: "os.environ/OPENAI_API_KEY"
- model_name: text-embedding-ada-002
  litellm_params:
    model: openai/text-embedding-ada-002
    api_key: "os.environ/OPENAI_API_KEY"
  model_info:
    mode: embedding
""")
ROUTER_CONFIG = ROUTER_CONFIG_VLLM if INTEGRATION_TEST_BACKEND == "vllm" else ROUTER_CONFIG_OPENAI

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
@override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'],
                   LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class GatewayIntegrationTestCase(TransactionTestCase):
    """
    Integration tests using the embedded RemoteOpenAIServer (with httpx).
    """
    vllm_server: Optional["RemoteOpenAIServer"] = None
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "gpt-4.1-nano"

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"

    VLLM_SEED = 42
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


TEST_FILES_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files_root")
os.makedirs(TEST_FILES_ROOT, exist_ok=True)


@override_settings(
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'],
)
class GatewayFilesTestCase(TransactionTestCase):
    # Load default fixture (includes test Token) and set test access token
    fixtures = ["gateway_data.json"]
    AQUEDUCT_ACCESS_TOKEN = GatewayIntegrationTestCase.AQUEDUCT_ACCESS_TOKEN

    def tearDown(self):
        super().tearDown()
        # Clean up the file storage directory after each test
        if os.path.exists(TEST_FILES_ROOT):
            shutil.rmtree(TEST_FILES_ROOT)
            os.makedirs(TEST_FILES_ROOT, exist_ok=True)


@override_settings(
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'],
    AQUEDUCT_BATCH_PROCESSING_MAX_CONCURRENCY=2,
    AQUEDUCT_BATCH_PROCESSING_MIN_CONCURRENCY=2,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    MAX_USER_BATCHES = 3,
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }
)
class GatewayBatchesTestCase(GatewayIntegrationTestCase):
    def tearDown(self):
        super().tearDown()
        # Clean up the file storage directory after each test
        if os.path.exists(TEST_FILES_ROOT):
            shutil.rmtree(TEST_FILES_ROOT)
            os.makedirs(TEST_FILES_ROOT, exist_ok=True)

    @staticmethod
    def run_batch_processing_loop():
        from gateway.views.batches import run_batch_processing
        async_to_sync(run_batch_processing)()
