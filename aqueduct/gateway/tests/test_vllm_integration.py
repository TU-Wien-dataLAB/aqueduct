# tests/test_vllm_integration.py
import json
import os
import sys
from typing import Optional, Literal

# Third-party imports
import httpx  # Using httpx instead of requests
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
# Django specific imports
from django.test import LiveServerTestCase, Client, override_settings
from openai import OpenAI

INTEGRATION_TEST_BACKEND: Literal["vllm", "openai"] = os.environ.get("INTEGRATION_TEST_BACKEND", "vllm")
if INTEGRATION_TEST_BACKEND not in ["vllm", "openai"]:
    raise ValueError("Integration test backend must be one of 'vllm' or 'openai‘.")

START_VLLM_SERVER = INTEGRATION_TEST_BACKEND == "vllm" and os.environ.get("START_VLLM_SERVER", "true") == "true"

# --- vLLM Internal Imports ---
RemoteOpenAIServer = None
_VLLM_AVAILABLE = False
if START_VLLM_SERVER:
    try:
        from gateway.tests.utils import RemoteOpenAIServer

        _VLLM_AVAILABLE = True
    except ImportError as e:
        _VLLM_IMPORT_ERROR = e

# Import Org, Team, UserProfile for direct DB manipulation
from management.models import Org, Team, Request, UserProfile, ServiceAccount

User = get_user_model()


# --- Django Test Class ---

# Define this as a standalone async function or a staticmethod
async def parse_streamed_data(streaming_content):
    lines = []
    try:
        async for chunk in streaming_content:  # This is where the error likely occurs
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            for line in chunk.splitlines():
                if line.strip() and line.startswith("data:"):
                    lines.append(line)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print("RuntimeError: Event loop is closed encountered inside parse_streamed_data's async for loop.")
            # Potentially add more context here, like what lines have been collected so far
            print(f"Lines collected before error: {lines}")
        raise  # Re-raise the exception to fail the test
    except Exception as e:
        print(f"Unexpected error during streaming content parsing: {e}")
        print(f"Lines collected before error: {lines}")
        raise
    return lines


@override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
class VLLMIntegrationTests(LiveServerTestCase):
    """
    Integration tests using the embedded RemoteOpenAIServer (with httpx).
    """
    vllm_server: Optional["RemoteOpenAIServer"] = None
    client: Optional[Client] = None
    open_ai_client: Optional[OpenAI] = None
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "gpt-4o-mini"

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"

    VLLM_SEED = 42
    fixtures = ["gateway_data.json", "vllm_endpoint.json", "openai_endpoint.json"]

    @classmethod
    def get_client(cls, **kwargs) -> OpenAI:
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600.0  # Use float for httpx compatibility underlying openai client
        base_url = cls.live_server_url.rstrip("/") + f"/{INTEGRATION_TEST_BACKEND}"
        return OpenAI(
            base_url=base_url,
            api_key=cls.AQUEDUCT_ACCESS_TOKEN,
            max_retries=0,
            **kwargs,
        )

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
                MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Or any other small model
                vllm_args = ["--host", "0.0.0.0", "--port", "8009"]
                cls.vllm_server = RemoteOpenAIServer(
                    model=MODEL_NAME,
                    vllm_serve_args=vllm_args,
                    seed=cls.VLLM_SEED,
                    auto_port=False,
                    max_wait_seconds=300
                )
                print(f"vLLM server started on: {cls.vllm_server.url_root}")

            except (ImportError, RuntimeError, Exception) as e:
                import traceback
                traceback.print_exc()  # Print full traceback for setup errors
                print(f"ERROR starting vLLM server: {e}", file=sys.stderr)
                cls.vllm_server = None
                raise AssertionError(f"Failed to set up vLLM server: {e}") from e
        elif INTEGRATION_TEST_BACKEND == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY environment variable has to be set for OpenAI integration.")

        cls.open_ai_client = cls.get_client()
        print("OpenAI client configured.")

    @classmethod
    def tearDownClass(cls):
        print(f"\nStopping vLLM server for {cls.__name__}...")
        if cls.vllm_server:
            try:
                cls.vllm_server.__exit__(None, None, None)
                print("vLLM server stopped.")
            except Exception as e:
                print(f"Error during vLLM server __exit__: {e}", file=sys.stderr)
                if hasattr(cls.vllm_server, 'proc') and cls.vllm_server.proc:
                    try:
                        if cls.vllm_server.proc.poll() is None: cls.vllm_server.proc.kill()
                        print("vLLM server process killed as fallback.")
                    except Exception as kill_e:
                        print(f"Error force killing vLLM server process: {kill_e}", file=sys.stderr)
        else:
            print("No vLLM server instance to stop (likely failed during setup).")
        super().tearDownClass()

    def test_vllm_server_is_running(self):
        """Checks if the server setup appears successful using httpx."""
        if not START_VLLM_SERVER:
            self.skipTest(reason="No self managed vLLM server.")

        self.assertIsNotNone(self.vllm_server, "vLLM server instance should exist (check setUpClass)")
        self.assertTrue(hasattr(self.vllm_server, 'proc'), "vLLM server should have a process attribute")
        self.assertIsNone(self.vllm_server.proc.poll(), "vLLM server process should be running")
        self.assertIsNotNone(self.client, "OpenAI client instance should exist (check setUpClass)")
        # Basic health check using httpx
        health_url = self.vllm_server.url_for("health")
        with httpx.Client(timeout=10.0) as client:
            response = client.get(health_url)
        self.assertEqual(response.status_code, 200,
                         f"Server health check failed ({response.status_code}) at {health_url}")

    def test_chat_completion(self):
        """
        Sends a simple chat completion request to the vLLM server.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Respond concisely."}
        ]

        # Clear Request table before test
        Request.objects.all().delete()

        response = self.open_ai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=50,
            temperature=0.0,
        )

        print(f"\nChat completion response: {response}")

        self.assertIsNotNone(response)
        self.assertTrue(response.choices)
        self.assertIsInstance(response.choices, list)
        self.assertGreater(len(response.choices), 0)

        first_choice = response.choices[0]
        self.assertTrue(hasattr(first_choice, 'message'))
        self.assertIsNotNone(first_choice.message)
        self.assertTrue(hasattr(first_choice.message, 'content'))
        self.assertIsNotNone(first_choice.message.content)

        response_text = first_choice.message.content.strip()
        print(response_text)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

    async def test_chat_completion_streaming(self):
        """
        Sends a streaming chat completion request to the vLLM server.
        Checks that multiple chunks are received and the content is non-empty.
        Also checks that the database contains one request and the endpoint matches.
        """
        # TODO: fix async error (RuntimeError: Event loop is closed) for streaming chat completions
        #  https://channels.readthedocs.io/en/latest/topics/testing.html#channelsliveservertestcase
        self.skipTest("Streaming causes error as event loop does not seem to be running when iterating over async for.")

        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name three countries in Europe."}
        ]

        # Clear Request table before test
        await Request.objects.all().adelete()

        # The OpenAI client should support streaming via .create(..., stream=True)
        chunks = []
        try:
            stream = self.open_ai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=50,
                temperature=0.0,
                stream=True,
            )
        except TypeError:
            self.skipTest("OpenAI client does not support streaming in this environment.")

        # Collect chunks
        for chunk in stream:
            chunks.append(chunk)
            # Optionally print each chunk for debugging
            print(f"Stream chunk: {chunk}")

        self.assertGreater(len(chunks), 1, "Should receive more than one chunk in streaming mode.")

        # Concatenate the content from all chunks (if present)
        content_pieces = []
        for chunk in chunks:
            # OpenAI python client: chunk.choices[0].delta.content (for streaming)
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                # vLLM and OpenAI: streaming delta is in .delta.content
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    piece = choice.delta.content
                    if piece:
                        content_pieces.append(piece)
                # Some clients may use .text or .message
                elif hasattr(choice, "text"):
                    piece = choice.text
                    if piece:
                        content_pieces.append(piece)
        full_content = "".join(content_pieces).strip()
        print(f"Full streamed content: {full_content}")
        self.assertTrue(full_content, "Streamed content should not be empty.")

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after streaming chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion (streaming).")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 (streaming)")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 (streaming)")

    def test_list_models(self):
        """
        Sends a request to list available models from the vLLM server.
        After the request, checks that the database contains one request and the endpoint matches.
        """
        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Clear Request table before test
        Request.objects.all().delete()

        # The OpenAI client should have a .models.list() method
        response = self.open_ai_client.models.list()
        print(f"\nList models response: {response}")

        self.assertIsNotNone(response)
        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertTrue(hasattr(response, 'data'))
        self.assertIsInstance(response.data, list)
        self.assertGreater(len(response.data), 0)

        # Check that at least one model matches the expected model name
        model_ids = [m.id for m in response.data if hasattr(m, 'id')]
        print(f"Available model IDs: {model_ids}")
        self.assertIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")

    def test_list_models_with_invalid_token(self):
        """
        Sends a request to list available models from the vLLM server with an invalid API key.
        Expects an authentication error (401 or 403).
        """
        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Save the original API key to restore later
        original_api_key = self.open_ai_client.api_key
        try:
            # Set an invalid API key
            self.open_ai_client.api_key = "invalid-token-123"

            # Clear Request table before test
            Request.objects.all().delete()

            # The OpenAI client should have a .models.list() method
            with self.assertRaises(Exception) as cm:
                self.open_ai_client.models.list()
            # Optionally, check the exception type or message
            # For httpx, it may be openai.AuthenticationError or openai.APIStatusError, or httpx.HTTPStatusError
            # You can check for 401/403 in the exception message
            msg = str(cm.exception)
            self.assertTrue(
                "401" in msg or "403" in msg or "Unauthorized" in msg or "forbidden" in msg.lower(),
                f"Expected authentication error, got: {msg}"
            )

            # There should be no request recorded in the database (or possibly one, depending on implementation)
            requests = list(Request.objects.all())
            self.assertEqual(len(requests), 0, "There should be no request recorded for invalid token.")
        finally:
            # Restore the original API key
            self.open_ai_client.api_key = original_api_key

    # --- Helper for setting limits on Org, Team, or UserProfile ---

    def setup_limits(self, kind: str, field: str, value: int):
        """
        Set a rate limit for the given kind ('org', 'team', 'user') and field.
        Returns the object whose limit was set.

        For 'team', also ensure the service account is associated with the test token.
        """
        if kind == "org":
            org = Org.objects.get(name="E060")
            setattr(org, field, value)
            org.save(update_fields=[field])
            return org
        elif kind == "team":
            team = Team.objects.get(name="Whale")
            setattr(team, field, value)
            team.save(update_fields=[field])
            # Ensure a service account exists for the team and associate it with the token
            # Only associate the service account with the token if the token does not already have a service account

            service_account = ServiceAccount.objects.create(team=team, name="Whale Service")

            from management.models import Token
            token = Token.objects.filter(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN)).first()
            if not token:
                raise RuntimeError("Could not find Token associated with AQUEDUCT_ACCESS_TOKEN.")

            # Only set the service_account if it is not already set
            if getattr(token, "service_account_id", None) is None:
                token.service_account = service_account
                token.save(update_fields=["service_account"])
            elif token.service_account_id != service_account.id:
                # If the token is already associated with a different service account, raise an error
                raise RuntimeError(
                    f"Token is already associated with a different service account (id={token.service_account_id})."
                )
            # Otherwise, already associated with the correct service account, do nothing
            return team
        elif kind == "user":
            user = User.objects.get(username="Me")
            profile = user.profile if hasattr(user, "profile") else UserProfile.objects.get(user=user)
            setattr(profile, field, value)
            profile.save(update_fields=[field])
            return profile
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def _rate_limit_test_template(self, kind: str, field: str, value: int, messages, max_tokens, limit_desc):
        """
        Generic template for rate limit tests for org, team, or user.
        """
        # Set the limit
        obj = self.setup_limits(kind, field, value)

        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Clear Request table before test
        Request.objects.all().delete()

        # First request should succeed
        response1 = self.open_ai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        self.assertIsNotNone(response1)
        self.assertTrue(response1.choices)
        self.assertGreater(len(response1.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1,
                         f"There should be exactly one request after first chat completion ({limit_desc}).")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        from openai._exceptions import RateLimitError

        with self.assertRaises(RateLimitError) as cm:
            self.open_ai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        err = cm.exception
        self.assertTrue("429" in str(err) or "rate limit" in str(err).lower())

    def test_org_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Org 'E060' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ]
        self._rate_limit_test_template(
            kind="org",
            field="requests_per_minute",
            value=1,
            messages=messages,
            max_tokens=5,
            limit_desc="org requests_per_minute"
        )

    def test_team_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Team 'Whale' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello from the team."}
        ]
        self._rate_limit_test_template(
            kind="team",
            field="requests_per_minute",
            value=1,
            messages=messages,
            max_tokens=5,
            limit_desc="team requests_per_minute"
        )

    def test_user_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of UserProfile for user 'Me' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello from the user."}
        ]
        self._rate_limit_test_template(
            kind="user",
            field="requests_per_minute",
            value=1,
            messages=messages,
            max_tokens=5,
            limit_desc="user requests_per_minute"
        )

    def test_org_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to the world."}
        ]
        self._rate_limit_test_template(
            kind="org",
            field="input_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=1,
            limit_desc="org input_tokens_per_minute"
        )

    def test_team_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to the world from the team."}
        ]
        self._rate_limit_test_template(
            kind="team",
            field="input_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=1,
            limit_desc="team input_tokens_per_minute"
        )

    def test_user_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to the world from the user."}
        ]
        self._rate_limit_test_template(
            kind="user",
            field="input_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=1,
            limit_desc="user input_tokens_per_minute"
        )

    def test_org_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say something long and verbose about the weather."}
        ]
        self._rate_limit_test_template(
            kind="org",
            field="output_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=10,
            limit_desc="org output_tokens_per_minute"
        )

    def test_team_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say something long and verbose about the weather as a team."}
        ]
        self._rate_limit_test_template(
            kind="team",
            field="output_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=10,
            limit_desc="team output_tokens_per_minute"
        )

    def test_user_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say something long and verbose about the weather as a user."}
        ]
        self._rate_limit_test_template(
            kind="user",
            field="output_tokens_per_minute",
            value=5,
            messages=messages,
            max_tokens=10,
            limit_desc="user output_tokens_per_minute"
        )
