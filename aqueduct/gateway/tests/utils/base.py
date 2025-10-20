import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import LiveServerTestCase, TransactionTestCase, override_settings

from management.models import Org, Token, UserProfile

INTEGRATION_TEST_BACKEND: Literal["vllm", "openai"] = os.environ.get(
    "INTEGRATION_TEST_BACKEND", "openai"
)
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

ROOT = Path(__file__).parent.parent.parent.parent.parent

with open(ROOT / "example_router_config.yaml") as f:
    ROUTER_CONFIG_OPENAI = f.read()

ROUTER_CONFIG = ROUTER_CONFIG_VLLM if INTEGRATION_TEST_BACKEND == "vllm" else ROUTER_CONFIG_OPENAI

START_VLLM_SERVER = (
    INTEGRATION_TEST_BACKEND == "vllm" and os.environ.get("START_VLLM_SERVER", "true") == "true"
)

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


User = get_user_model()

TEST_FILES_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files_root")
os.makedirs(TEST_FILES_ROOT, exist_ok=True)


# --- Django Test Class ---
@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
)
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
                    max_wait_seconds=300,
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
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable has to be set for OpenAI integration."
                )

    @staticmethod
    def create_new_user() -> tuple[str, int]:
        # Create a new user and a new token for that user
        new_user = User.objects.create_user(username="OtherUser", email="other@example.com")
        new_user.groups.add(Group.objects.get(name="user"))
        org = Org.objects.get(name="E060")
        profile = UserProfile.objects.create(user=new_user, org=org)
        new_user.profile = profile

        # Create a new token for the different user
        new_token = Token(name="TestToken", user=new_user)
        token_value = new_token._set_new_key()
        new_token.save()
        return token_value, new_user.id


@override_settings(
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
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
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_BATCH_PROCESSING_CONCURRENCY=lambda: 2,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
    MAX_USER_BATCHES=3,
    AQUEDUCT_BATCH_PROCESSING_RUNTIME_MINUTES=3 / 60,
    AQUEDUCT_BATCH_PROCESSING_RELOAD_INTERVAL_SECONDS=2,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
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


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    AQUEDUCT_FILES_API_ROOT=TEST_FILES_ROOT,
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
)
class GatewayTTSSTTestCase(GatewayIntegrationTestCase):
    fixtures = ["gateway_data.json"]
    tts_model = "gpt-4o-mini-tts"
    stt_model = "whisper-1"


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
        from django.contrib.auth import get_user_model
        from tos.models import TermsOfService, UserAgreement

        User = get_user_model()

        # Create an active Terms of Service
        tos = TermsOfService.objects.create(active=True, content="Test Terms of Service content")

        # Get the user with pk=user_id and create a UserAgreement
        user = User.objects.get(pk=user_id)
        UserAgreement.objects.create(terms_of_service=tos, user=user)


MCP_CONFIG_PATH = "/tmp/aqueduct/test-mcp-config.json"
MCP_TEST_CONFIG = {
    "mcpServers": {"test-server": {"type": "streamable-http", "url": "http://localhost:3001"}}
}


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    MCP_CONFIG_FILE_PATH=MCP_CONFIG_PATH,
)
class GatewayMCPTestCase(TransactionTestCase):
    fixtures = ["gateway_data.json"]
    AQUEDUCT_ACCESS_TOKEN = GatewayIntegrationTestCase.AQUEDUCT_ACCESS_TOKEN

    @classmethod
    def _write_mcp_config(cls):
        import json

        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(MCP_TEST_CONFIG, f)

    @classmethod
    def setUpClass(cls):
        cls._write_mcp_config()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(MCP_CONFIG_PATH):
            os.remove(MCP_CONFIG_PATH)


@override_settings(
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    MCP_CONFIG_FILE_PATH=MCP_CONFIG_PATH,
)
class MCPLiveServerTestCase(LiveServerTestCase):
    """
    Live server test case for MCP endpoints using Django's LiveServerTestCase.
    Provides a running server for testing HTTP endpoints with httpx.
    """

    fixtures = ["gateway_data.json"]
    AQUEDUCT_ACCESS_TOKEN = GatewayIntegrationTestCase.AQUEDUCT_ACCESS_TOKEN
    mcp_server_process = None

    @classmethod
    def _write_mcp_config(cls):
        import json

        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(MCP_TEST_CONFIG, f)

    @classmethod
    def _start_mcp_server(cls):
        """Start the MCP everything server on localhost:3001."""
        print("\nStarting MCP everything server...")
        try:
            cls.mcp_server_process = subprocess.Popen(
                ["npx", "@modelcontextprotocol/server-everything", "streamableHttp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait a moment for server to start
            time.sleep(3)

            # Check if process is still running
            if cls.mcp_server_process.poll() is None:
                print("MCP everything server started on localhost:3001")
            else:
                stdout, stderr = cls.mcp_server_process.communicate()
                raise RuntimeError(
                    f"MCP server failed to start. "
                    f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "npx command not found. Please ensure Node.js and npm are installed."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {e}") from e

    @classmethod
    def _stop_mcp_server(cls):
        """Stop the MCP everything server."""
        if cls.mcp_server_process and cls.mcp_server_process.poll() is None:
            print("Stopping MCP everything server...")
            cls.mcp_server_process.terminate()
            try:
                cls.mcp_server_process.wait(timeout=10)
                print("MCP server stopped successfully")
            except subprocess.TimeoutExpired:
                print("MCP server did not stop gracefully, forcing kill")
                cls.mcp_server_process.kill()
                cls.mcp_server_process.wait()
            cls.mcp_server_process = None

    @classmethod
    def setUpClass(cls):
        cls._write_mcp_config()
        cls._start_mcp_server()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._stop_mcp_server()
        if os.path.exists(MCP_CONFIG_PATH):
            os.remove(MCP_CONFIG_PATH)
