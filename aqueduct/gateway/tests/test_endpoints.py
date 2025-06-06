import json

from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth import get_user_model
from django.test import override_settings
from openai.types.chat import ChatCompletion

from gateway.router import get_router_config
from gateway.tests.utils import reset_gateway_httpx_async_client, _build_chat_headers, _build_chat_payload, \
    _read_streaming_response_lines, _parse_streamed_content_pieces
from gateway.tests.utils.base import GatewayIntegrationTestCase, INTEGRATION_TEST_BACKEND, ROUTER_CONFIG
from management.models import Request, UserProfile, ServiceAccount, Team, Org

User = get_user_model()


class EmbeddingTest(GatewayIntegrationTestCase):
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "text-embedding-ada-002"

    @reset_gateway_httpx_async_client
    @override_settings(RELAY_REQUEST_TIMEOUT=5)
    def test_embeddings(self):
        """
        Sends a simple embeddings request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... Requires GatewayIntegrationTestCase to manage multiple servers!")

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        assert self.model in ROUTER_CONFIG
        payload = {
            "model": self.model,
            "input": ["The quick brown fox jumps over the lazy dog."]
        }
        endpoint = f"/embeddings"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        response_json = response.json()
        print(f"\nEmbeddings response: {response_json}")

        # OpenAI-style embeddings response should have 'data' and 'embedding' fields
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)
        embedding_obj = response_json["data"][0]
        self.assertIn("embedding", embedding_obj)
        self.assertIsInstance(embedding_obj["embedding"], list)
        self.assertGreater(len(embedding_obj["embedding"]), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after embeddings.")
        req = requests[0]
        self.assertIn("embeddings", req.path, "Request endpoint should be for embeddings.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertEqual(req.output_tokens, 0, "output_tokens should be 0")


class ChatCompletionsIntegrationTest(GatewayIntegrationTestCase):
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a short poem!"}
    ]

    def _build_chat_completion_request(self, messages, stream=False, **payload_kwargs) -> dict:
        """
        Helper to build headers, payload, and endpoint for chat completion requests.
        """
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        payload = _build_chat_payload(self.model, messages, stream=stream, **payload_kwargs)
        endpoint = f"/chat/completions"
        return dict(path=endpoint, data=json.dumps(payload), headers=headers)

    def _send_chat_completion(self, messages, **payload_kwargs):
        """
        Helper to send a chat completion request (non-streaming) using Django test client.
        """
        request = self._build_chat_completion_request(messages, stream=False, **payload_kwargs)
        return self.client.post(**request, content_type="application/json")

    async def _send_chat_completion_streaming(self, messages, **payload_kwargs):
        """
        Helper to send a streaming chat completion request using Django async test client.
        """
        request = self._build_chat_completion_request(messages, stream=True, **payload_kwargs)
        response = await self.async_client.post(**request, content_type="application/json")
        return response

    @reset_gateway_httpx_async_client
    def test_chat_completion(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        # Parse the response as JSON and convert to OpenAI ChatCompletion type for compatibility
        response_json = response.json()
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
    @override_settings(RELAY_REQUEST_TIMEOUT=0.1)
    def test_chat_completion_timeout(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(
            response.status_code, 504,
            f"Expected 504 Gateway Timeout, got {response.status_code}: {response.content}"
        )

    @reset_gateway_httpx_async_client
    @override_settings(STREAM_REQUEST_TIMEOUT=5)
    @async_to_sync
    async def test_chat_completion_streaming(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # For some reason authentication does not work in async test case...
        await sync_to_async(lambda: self.async_client.force_login(
            User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        response = await self._send_chat_completion_streaming(self.MESSAGES)

        # Should be a StreamingHttpResponse with status 200
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Collect all streamed lines (each line is a data: ... event)
        streamed_lines = await _read_streaming_response_lines(response)

        self.assertGreater(len(streamed_lines), 0, "Should receive at least one streamed data chunk.")

        # Parse each chunk as JSON and collect content pieces
        content_pieces = _parse_streamed_content_pieces(streamed_lines)
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
    @override_settings(RELAY_REQUEST_TIMEOUT=0.001)
    @async_to_sync
    async def test_chat_completion_streaming_relay_request_timeout(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # For some reason authentication does not work in async test case...
        await sync_to_async(lambda: self.async_client.force_login(
            User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        response = await self._send_chat_completion_streaming(self.MESSAGES)

        self.assertEqual(
            response.status_code, 504,
            f"Expected 504 Gateway Timeout, got {response.status_code}"
        )


class ListModelsIntegrationTest(GatewayIntegrationTestCase):

    def _send_model_list_request(self):
        return self.client.get(
            f"/models",
            data='',
            content_type="application/json",
            headers=_build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        )

    @reset_gateway_httpx_async_client
    def test_list_models(self):
        """
        Sends a request to list available models from the vLLM server using the Django test client.
        After the request, checks that the database contains one request and the endpoint matches.
        """
        response = self._send_model_list_request()

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
        # Prepare headers with an invalid token
        headers = {
            "HTTP_AUTHORIZATION": "Bearer invalid-token-123",
            "CONTENT_TYPE": "application/json",
        }

        # No payload needed for model listing
        response = self.client.get(
            f"/models",
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

    @reset_gateway_httpx_async_client
    def test_list_excluded_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()
        assert len(org.excluded_models) == 1

        router_config = get_router_config()
        model_list: list[dict] = router_config["model_list"]

        response = self._send_model_list_request()

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
        self.assertEqual(len(model_ids), len(model_list) - 1)
        self.assertNotIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")


class TokenLimitTest(ChatCompletionsIntegrationTest):
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

        response1 = self._send_chat_completion(messages, max_tokens=max_tokens, temperature=0.0)
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
        response2 = self._send_chat_completion(messages, max_tokens=max_tokens, temperature=0.0)
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
        self._rate_limit_test_template(
            kind="org",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="org requests_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_team_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Team 'Whale' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="team requests_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_user_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of UserProfile for user 'Me' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="user requests_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_org_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="org input_tokens_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_team_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="team input_tokens_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_user_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="user input_tokens_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_org_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="org output_tokens_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_team_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="team output_tokens_per_minute"
        )

    @reset_gateway_httpx_async_client
    def test_user_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="user output_tokens_per_minute"
        )
