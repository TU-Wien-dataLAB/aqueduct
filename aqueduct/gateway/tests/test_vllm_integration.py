# tests/test_vllm_integration.py

import sys
from typing import Optional

# Third-party imports
import httpx  # Using httpx instead of requests
from openai import OpenAI

# Django specific imports
from django.test import TestCase, LiveServerTestCase, Client, override_settings

# --- vLLM Internal Imports ---
# These are still required by the embedded RemoteOpenAIServer logic.
# Ensure vLLM is installed correctly for these to work.

from gateway.tests.utils import RemoteOpenAIServer

# Import Org for direct DB manipulation
from management.models import Org
from management.models import Request

# --- Django Test Class ---

class VLLMIntegrationTests(LiveServerTestCase):
    """
    Integration tests using the embedded RemoteOpenAIServer (with httpx).
    """
    vllm_server: Optional[RemoteOpenAIServer] = None
    client: Optional[Client] = None
    open_ai_client: Optional[OpenAI] = None
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Or any other small model

    # Hash: 750a701272d7624a3e6f10d5e0d9efdf0e2c7e575803219081358db36bfd243a
    # Preview: k-...3abc
    AQUEDUCT_ACCESS_TOKEN = "sk-123abc"

    VLLM_SEED = 42
    fixtures = ["gateway_data.json", "vllm_endpoint.json"]

    @classmethod
    def get_client(cls, **kwargs) -> OpenAI:
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600.0  # Use float for httpx compatibility underlying openai client
        base_url = cls.live_server_url.rstrip("/") + "/vllm"
        return OpenAI(
            base_url=base_url,
            api_key=cls.AQUEDUCT_ACCESS_TOKEN,
            max_retries=0,
            **kwargs,
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        print(f"\nStarting vLLM server for {cls.__name__}...")
        try:
            vllm_args = ["--host", "0.0.0.0", "--port", "8009"]
            cls.vllm_server = RemoteOpenAIServer(
                model=cls.MODEL_NAME,
                vllm_serve_args=vllm_args,
                seed=cls.VLLM_SEED,
                auto_port=False,
                max_wait_seconds=300
            )
            print(f"vLLM server started on: {cls.vllm_server.url_root}")
            cls.open_ai_client = cls.get_client()
            print("OpenAI client configured.")

        except (ImportError, RuntimeError, Exception) as e:
            import traceback
            traceback.print_exc()  # Print full traceback for setup errors
            print(f"ERROR starting vLLM server: {e}", file=sys.stderr)
            cls.vllm_server = None
            raise AssertionError(f"Failed to set up vLLM server: {e}") from e

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

    @override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
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
            model="Qwen-0.5B",
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

    @override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
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
        self.assertIn("Qwen-0.5B", model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")

    @override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
    def test_org_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Org 'E060' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        After the first request, checks that the database contains one request and the endpoint matches,
        and input/output tokens are > 0.
        """
        # Set Org requests_per_minute to 1
        org = Org.objects.get(name="E060")
        org.requests_per_minute = 1
        org.save(update_fields=["requests_per_minute"])

        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Clear Request table before test
        Request.objects.all().delete()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ]

        # First request should succeed
        response1 = self.open_ai_client.chat.completions.create(
            model="Qwen-0.5B",
            messages=messages,
            max_tokens=5,
            temperature=0.0,
        )
        self.assertIsNotNone(response1)
        self.assertTrue(response1.choices)
        self.assertGreater(len(response1.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after first chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        from openai import OpenAIError
        from openai._exceptions import RateLimitError

        with self.assertRaises(RateLimitError) as cm:
            self.open_ai_client.chat.completions.create(
                model="Qwen-0.5B",
                messages=messages,
                max_tokens=5,
                temperature=0.0,
            )
        # Optionally, check the error message or status
        err = cm.exception
        self.assertTrue("429" in str(err) or "rate limit" in str(err).lower())

    @override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
    def test_org_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        After the first request, checks that the database contains one request and the endpoint matches,
        and input/output tokens are > 0.
        """
        org = Org.objects.get(name="E060")
        org.input_tokens_per_minute = 5
        org.save(update_fields=["input_tokens_per_minute"])

        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Clear Request table before test
        Request.objects.all().delete()

        # Use a prompt that is at least 5 tokens (should be easy with a sentence)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to the world."}
        ]

        # First request should succeed
        response1 = self.open_ai_client.chat.completions.create(
            model="Qwen-0.5B",
            messages=messages,
            max_tokens=1,
            temperature=0.0,
        )
        self.assertIsNotNone(response1)
        self.assertTrue(response1.choices)
        self.assertGreater(len(response1.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after first chat completion.")
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
                model="Qwen-0.5B",
                messages=messages,
                max_tokens=1,
                temperature=0.0,
            )
        err = cm.exception
        self.assertTrue("429" in str(err) or "rate limit" in str(err).lower())

    @override_settings(AUTHENTICATION_BACKENDS=['gateway.authentication.TokenAuthenticationBackend'])
    def test_org_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        After the first request, checks that the database contains one request and the endpoint matches,
        and input/output tokens are > 0.
        """
        org = Org.objects.get(name="E060")
        org.output_tokens_per_minute = 5
        org.save(update_fields=["output_tokens_per_minute"])

        if not self.open_ai_client:
            self.skipTest("Skipping test: OpenAI client not available (server setup likely failed).")

        # Clear Request table before test
        Request.objects.all().delete()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say something long and verbose about the weather."}
        ]

        # First request should succeed
        response1 = self.open_ai_client.chat.completions.create(
            model="Qwen-0.5B",
            messages=messages,
            max_tokens=10,  # Should use up the output token budget
            temperature=0.0,
        )
        self.assertIsNotNone(response1)
        self.assertTrue(response1.choices)
        self.assertGreater(len(response1.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after first chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request path should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        from openai._exceptions import RateLimitError

        with self.assertRaises(RateLimitError) as cm:
            self.open_ai_client.chat.completions.create(
                model="Qwen-0.5B",
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
        err = cm.exception
        self.assertTrue("429" in str(err) or "rate limit" in str(err).lower())
