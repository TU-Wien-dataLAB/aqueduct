# tests/test_vllm_integration.py
import json
import os
import sys
from typing import Optional, Literal

# Third-party imports
import httpx  # Using httpx instead of requests
from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth import get_user_model
# Django specific imports
from django.test import TransactionTestCase, Client, override_settings
from openai.types.chat import ChatCompletion, ChatCompletionChunk  # Use OpenAI types for response parsing
import functools


def reset_gateway_httpx_async_client(test_func):
    """
    This is needed since the httpxAsyncClient uses connection pooling and cannot run under different event loops.
    When Django runs tests for async views, async_to_sync is called, which spawns a new event loop, breaking connection pooling.
    For each test requiring a relay request, the async client, therefore, has to be reset.
    See: https://github.com/encode/httpx/discussions/2959
    """

    @functools.wraps(test_func)
    def wrapper(self, *args, **kwargs):
        import gateway.views
        gateway.views.async_client = httpx.AsyncClient(timeout=60, follow_redirects=True)
        return test_func(self, *args, **kwargs)

    return wrapper


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


@override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
class VLLMIntegrationTests(TransactionTestCase):
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

    @reset_gateway_httpx_async_client
    def test_chat_completion(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """

        # Use Django's test client to POST to the chat completion endpoint
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Respond concisely."}
        ]

        # Clear Request table before test
        Request.objects.all().delete()

        # Prepare headers and payload
        headers = {
            "HTTP_AUTHORIZATION": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
            "CONTENT_TYPE": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.0,
        }

        # POST to the OpenAI-compatible endpoint
        response = self.client.post(
            f"/{self.AQUEDUCT_ENDPOINT}/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
            **headers
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        # Parse the response as JSON and convert to OpenAI ChatCompletion type for compatibility
        response_json = response.json()
        # Optionally, you can use openai.types.chat.ChatCompletion.from_dict if available
        chat_completion = ChatCompletion.model_validate(response_json)

        print(f"\nChat completion response: {chat_completion}")

        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertIsInstance(chat_completion.choices, list)
        self.assertGreater(len(chat_completion.choices), 0)

        first_choice = chat_completion.choices[0]
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

    @reset_gateway_httpx_async_client
    @async_to_sync
    async def test_chat_completion_streaming(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # reset_gateway_httpx_async_client(lambda: None)

        # Use Django's test client to POST to the chat completion endpoint with stream=True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name three countries in Europe."}
        ]

        # Clear Request table before test
        await Request.objects.all().adelete()
        # For some reason authentication does not work in async test case...
        await sync_to_async(lambda: self.async_client.force_login(User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        # Prepare headers and payload
        headers = {
            "HTTP_AUTHORIZATION": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
            "CONTENT_TYPE": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": True,
        }

        # POST to the OpenAI-compatible endpoint (streaming) using self.async_client
        response = await self.async_client.post(
            f"/{self.AQUEDUCT_ENDPOINT}/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
            **headers
        )

        # Should be a StreamingHttpResponse with status 200
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Collect all streamed chunks (each line is a data: ... event)
        streamed_lines = []

        async for chunk in response.streaming_content:
            # chunk may be bytes, decode if needed
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            for line in chunk.splitlines():
                line = line.strip()
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        continue
                    streamed_lines.append(data)

        self.assertGreater(len(streamed_lines), 0, "Should receive at least one streamed data chunk.")

        # Parse each chunk as JSON and collect content pieces
        content_pieces = []
        for data in streamed_lines:
            try:
                # Use OpenAI's ChatCompletionChunk for validation and parsing
                chunk = ChatCompletionChunk.model_validate(json.loads(data))
            except Exception as e:
                self.fail(f"Failed to parse streamed chunk as ChatCompletionChunk: {data} ({e})")
            # OpenAI streaming: chunk.choices[0].delta.content
            choices = chunk.choices
            if choices:
                delta = choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    content_pieces.append(piece)
        full_content = "".join(content_pieces).strip()
        print(f"Full streamed content: {full_content}")
        self.assertTrue(full_content, "Streamed content should not be empty.")  

        # Check that the database contains one request and endpoint matches
        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(len(requests), 1, "There should be exactly one request after streaming chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion (streaming).")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 (streaming)")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 (streaming)")

    @reset_gateway_httpx_async_client
    def test_list_models(self):
        """
        Sends a request to list available models from the vLLM server using the Django test client.
        After the request, checks that the database contains one request and the endpoint matches.
        """
        # Clear Request table before test
        Request.objects.all().delete()

        # Prepare headers
        headers = {
            "HTTP_AUTHORIZATION": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
            "CONTENT_TYPE": "application/json",
        }

        # No payload needed for model listing
        response = self.client.get(
            f"/{self.AQUEDUCT_ENDPOINT}/models",
            data='',
            content_type="application/json",
            **headers
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        response_json = response.json()
        print(f"\nList models response: {response_json}")

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        print(f"Available model IDs: {model_ids}")
        self.assertIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")

    @reset_gateway_httpx_async_client
    def test_list_models_with_invalid_token(self):
        """
        Sends a request to list available models from the vLLM server with an invalid API key.
        Expects an authentication error (401 or 403).
        """
        # Clear Request table before test
        Request.objects.all().delete()

        # Prepare headers with an invalid token
        headers = {
            "HTTP_AUTHORIZATION": "Bearer invalid-token-123",
            "CONTENT_TYPE": "application/json",
        }

        # No payload needed for model listing
        response = self.client.get(
            f"/{self.AQUEDUCT_ENDPOINT}/models",
            data='',
            content_type="application/json",
            **headers
        )

        # Should be 401 Unauthorized or 403 Forbidden
        self.assertIn(
            response.status_code, [401, 403],
            f"Expected 401 or 403 for invalid token, got {response.status_code}: {response.content}"
        )

        # There should be no request recorded in the database (or possibly one, depending on implementation)
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 0, "There should be no request recorded for invalid token.")

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
        Uses the Django test client to POST to the chat completion endpoint.
        """
        # Set the limit
        obj = self.setup_limits(kind, field, value)

        # Clear Request table before test
        Request.objects.all().delete()

        # Prepare headers and payload
        headers = {
            "HTTP_AUTHORIZATION": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}",
            "CONTENT_TYPE": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        # First request should succeed
        response1 = self.client.post(
            f"/{self.AQUEDUCT_ENDPOINT}/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
            **headers
        )
        self.assertEqual(
            response1.status_code, 200,
            f"Expected 200 OK, got {response1.status_code}: {response1.content}"
        )
        response_json = response1.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertGreater(len(chat_completion.choices), 0)

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
        response2 = self.client.post(
            f"/{self.AQUEDUCT_ENDPOINT}/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
            **headers
        )
        self.assertEqual(
            response2.status_code, 429,
            f"Expected 429 Too Many Requests, got {response2.status_code}: {response2.content}"
        )
        # Optionally, check error message
        try:
            error_json = response2.json()
            self.assertTrue(
                "rate limit" in str(error_json).lower() or "429" in str(error_json),
                f"Expected rate limit error message, got: {error_json}"
            )
        except Exception:
            # If not JSON, just check content
            self.assertTrue(
                "rate limit" in response2.content.decode().lower() or "429" in response2.content.decode(),
                f"Expected rate limit error message, got: {response2.content}"
            )

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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

    @reset_gateway_httpx_async_client
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
