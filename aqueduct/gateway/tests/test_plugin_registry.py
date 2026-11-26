from django.test import TestCase

from gateway.decorators.plugins import generate_plugin_stub, get_plugin_endpoint_registry
from management.plugins import compile_plugin_class


class PluginEndpointRegistryTestCase(TestCase):
    def test_registry_is_populated_and_has_expected_endpoints(self):
        reg = get_plugin_endpoint_registry()
        self.assertIn("chat/completions", reg)
        self.assertIn("embeddings", reg)
        self.assertIn("audio/speech", reg)
        self.assertIn("audio/transcriptions", reg)
        self.assertIn("images/generations", reg)

    def test_registry_response_kinds(self):
        reg = get_plugin_endpoint_registry()
        self.assertEqual(reg["chat/completions"].response_kind, "json_or_streaming")
        self.assertEqual(reg["completions"].response_kind, "json_or_streaming")
        self.assertEqual(reg["embeddings"].response_kind, "json")
        self.assertEqual(reg["decisions"].response_kind, "json")
        self.assertEqual(reg["audio/speech"].response_kind, "streaming")
        self.assertEqual(reg["images/generations"].response_kind, "json")

    def test_registry_type_names(self):
        reg = get_plugin_endpoint_registry()
        self.assertEqual(reg["chat/completions"].type_name, "CompletionCreateParams")
        self.assertEqual(reg["embeddings"].type_name, "EmbeddingCreateParams")
        self.assertEqual(reg["audio/speech"].type_name, "SpeechCreateParams")
        # OpenAI params are NonStreaming | Streaming unions; the base name wins.
        self.assertEqual(reg["responses"].type_name, "ResponseCreateParams")


class GeneratePluginStubTestCase(TestCase):
    def _ep(self, key, type_name, kind):
        return {"key": key, "type_name": type_name, "response_kind": kind}

    def test_stub_json_only_endpoint(self):
        stub = generate_plugin_stub([self._ep("embeddings", "EmbeddingCreateParams", "json")])
        self.assertIn("endpoints = ['embeddings']", stub)
        self.assertIn("def before_request", stub)
        self.assertIn("def after_response", stub)
        self.assertNotIn("def after_streaming_response", stub)
        self.assertIn("EmbeddingCreateParams", stub)

    def test_stub_streaming_only_endpoint(self):
        stub = generate_plugin_stub([self._ep("audio/speech", "SpeechCreateParams", "streaming")])
        self.assertNotIn("def after_response", stub)
        self.assertIn("def after_streaming_response", stub)

    def test_stub_json_or_streaming_endpoint(self):
        stub = generate_plugin_stub(
            [self._ep("chat/completions", "CompletionCreateParams", "json_or_streaming")]
        )
        self.assertIn("def after_response", stub)
        self.assertIn("def after_streaming_response", stub)

    def test_stub_multiple_endpoints(self):
        stub = generate_plugin_stub(
            [
                self._ep("chat/completions", "CompletionCreateParams", "json_or_streaming"),
                self._ep("embeddings", "EmbeddingCreateParams", "json"),
            ]
        )
        self.assertIn("'chat/completions'", stub)
        self.assertIn("'embeddings'", stub)
        self.assertIn("def after_response", stub)
        self.assertIn("def after_streaming_response", stub)

    def test_stub_no_endpoints_runs_everywhere(self):
        stub = generate_plugin_stub([])
        self.assertIn("endpoints = []", stub)
        self.assertIn("runs on every endpoint", stub)

    def test_stub_compiles_to_plugin(self):
        stub = generate_plugin_stub(
            [self._ep("chat/completions", "CompletionCreateParams", "json_or_streaming")]
        )
        cls = compile_plugin_class(stub)
        self.assertEqual(cls.endpoints, ["chat/completions"])

    def test_stub_keeps_fixed_signature_and_comment_on_single_endpoint(self):
        stub = generate_plugin_stub(
            [self._ep("chat/completions", "CompletionCreateParams", "json_or_streaming")]
        )
        # Hook signatures are fixed; only a comment carries the request-body type.
        self.assertIn("def before_request(self, *args, **kwargs) -> dict | None:", stub)
        self.assertNotIn("from __future__ import annotations", stub)
        self.assertNotIn("body: CompletionCreateParams", stub)
        self.assertIn("# Request body is validated against: CompletionCreateParams", stub)

    def test_stub_keeps_fixed_signature_and_comment_on_multiple_endpoints(self):
        stub = generate_plugin_stub(
            [
                self._ep("chat/completions", "CompletionCreateParams", "json_or_streaming"),
                self._ep("embeddings", "EmbeddingCreateParams", "json"),
            ]
        )
        self.assertIn("def before_request(self, *args, **kwargs) -> dict | None:", stub)
        # Both request-body types are documented for the author via comment only.
        comment = (
            "# Request body is validated against: CompletionCreateParams, EmbeddingCreateParams"
        )
        self.assertIn(comment, stub)
