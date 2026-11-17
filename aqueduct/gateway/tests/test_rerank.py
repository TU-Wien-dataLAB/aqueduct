import json

from django.urls import reverse

from gateway.tests.utils.base import (
    INTEGRATION_TEST_BACKEND,
    ROUTER_CONFIG,
    GatewayIntegrationTestCase,
)
from management.models import Request
from mock_api.mock_configs import MockConfig


class RerankTest(GatewayIntegrationTestCase):
    model = "bge-reranker-v2-m3"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.url = reverse("gateway:rerank")

    def test_rerank(self):
        """
        Sends a simple rerank request to the mock server using the Django test
        client. After the request, checks that the response contains the
        expected results and that the database contains one logged request with
        the token usage recorded.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... "
                "Requires GatewayIntegrationTestCase to manage multiple servers!"
            )

        assert self.model in ROUTER_CONFIG
        payload = {
            "model": self.model,
            "query": "What is Aqueduct?",
            "documents": [
                "Aqueduct is an AI gateway developed by TU Wien.",
                "Vienna is the capital of Austria.",
                "Aqueduct supports multiple AI model providers.",
            ],
            "top_n": 3,
            "return_documents": True,
        }

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()

        # Rerank responses should have 'results' with index and score
        self.assertIn("results", response_json)
        self.assertIsInstance(response_json["results"], list)
        self.assertGreater(len(response_json["results"]), 0)
        result = response_json["results"][0]
        self.assertIn("index", result)
        self.assertIn("relevance_score", result)
        self.assertIn("document", result)

        # The request body is forwarded to the upstream rerank endpoint
        forwarded = self.mock_server.get_last_request("rerank")
        self.assertIsNotNone(forwarded, "The rerank request should reach the upstream mock.")
        self.assertEqual(forwarded["model"], self.model)
        self.assertEqual(forwarded["query"], payload["query"])
        self.assertEqual(forwarded["documents"], payload["documents"])
        self.assertEqual(forwarded["top_n"], 3)
        self.assertNotIn("timeout", forwarded)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after rerank.")
        req = requests[0]
        self.assertIn("rerank", req.path, "Request endpoint should be for rerank.")
        self.assertEqual(req.model, self.model)
        # Rerank usage is reported as input tokens
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertEqual(req.output_tokens, 0, "output_tokens should be 0")
        self.assertEqual(req.cached_input_tokens, 0)
        self.assertEqual(req.user_id, "")

    def test_rerank_with_alias(self):
        """
        Test that rerank requests using an alias are routed correctly.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("Requires the mock API server")

        payload = {"model": "reranker", "query": "What is Aqueduct?", "documents": ["One", "Two"]}
        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for rerank. "
            f"Expected 200, got {response.status_code}: {response.content}",
        )

        request_log = Request.objects.first()
        self.assertIsNotNone(request_log)
        self.assertEqual(
            request_log.model,
            self.model,
            "Request model should be the resolved model, not the alias",
        )

    def test_rerank_cohere_format_passthrough(self):
        """
        Cohere-style requests and responses pass through unchanged. The
        Cohere-specific rank_fields request field reaches the upstream and
        the token counts are logged from meta.tokens.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... "
                "Requires GatewayIntegrationTestCase to manage multiple servers!"
            )

        payload = {
            "model": self.model,
            "query": "What is Aqueduct?",
            "documents": ["One", "Two", "Three"],
            "rank_fields": ["title", "text"],
        }
        with self.mock_server.patch_external_api(
            "rerank",
            MockConfig(
                response_data={
                    "id": "rerank-cohere-1",
                    "results": [{"index": 0, "relevance_score": 0.9, "document": {"text": "One"}}],
                    "meta": {
                        "api_version": {"version": "1"},
                        "tokens": {"input_tokens": 17, "output_tokens": 0},
                    },
                }
            ),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )
        # The response is relayed back unchanged (Cohere-style fields intact).
        response_json = response.json()
        self.assertEqual(response_json["id"], "rerank-cohere-1")
        self.assertIn("meta", response_json)

        # The Cohere-specific rank_fields request field reached the upstream.
        forwarded = self.mock_server.get_last_request("rerank")
        self.assertIsNotNone(forwarded)
        self.assertEqual(forwarded["rank_fields"], ["title", "text"])

        # Token usage is read from meta.tokens of the Cohere-style response.
        req = Request.objects.first()
        self.assertIsNotNone(req)
        self.assertEqual(req.input_tokens, 17)
        self.assertEqual(req.output_tokens, 0)
