import json
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from openai.types.chat import ChatCompletion

from gateway.config import (
    get_all_model_request_limit_multipliers,
    get_model_request_limit_multiplier,
    get_model_supports_response_format,
    get_router_config,
)
from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Org, Request


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH="router.yaml")
class ModelAliasConfigValidationTest(SimpleTestCase):
    """
    Tests for model alias configuration validation.
    These tests validate that the router config correctly validates aliases when loaded.
    """

    def test_config_load_with_valid_aliases(self):
        """
        Test that config loads successfully when aliases are unique and valid.
        """
        # Mock config with unique aliases
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main", "coding"]},
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"mode": "embedding", "aliases": ["embedding"]},
                },
            ]
        }

        with (
            patch("pathlib.Path.open"),
            patch("yaml.safe_load", return_value=mock_config) as mock_load,
        ):
            get_router_config.cache_clear()

            loaded_config = get_router_config()

            # Config should load without errors
            self.assertIsInstance(loaded_config, dict)
            self.assertIn("model_list", loaded_config)

            # Verify aliases are in the config
            first_model = loaded_config["model_list"][0]
            self.assertIn("model_info", first_model)
            self.assertIn("aliases", first_model["model_info"])
            self.assertEqual(first_model["model_info"]["aliases"], ["main", "coding"])
            mock_load.assert_called_once()

    def test_config_load_with_duplicate_aliases_raises_error(self):
        """
        Test that config validation raises error when aliases are not unique.
        """
        # Mock config with duplicate "main" alias
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},  # Duplicate!
                },
            ]
        }

        with (
            patch("pathlib.Path.open"),
            patch("yaml.safe_load", return_value=mock_config) as mock_load,
        ):
            get_router_config.cache_clear()

            # Should raise RuntimeError due to duplicate aliases
            with self.assertRaises(RuntimeError) as context:
                get_router_config()

            # Verify the error message mentions the duplicate alias
            self.assertIn("Duplicate alias", str(context.exception))
            self.assertIn("main", str(context.exception))
            mock_load.assert_called_once()

    def test_config_load_with_multiple_aliases_per_model(self):
        """
        Test that a single model can have multiple aliases.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main", "coding", "default", "primary"]},
                }
            ]
        }

        with (
            patch("pathlib.Path.open"),
            patch("yaml.safe_load", return_value=mock_config) as mock_load,
        ):
            get_router_config.cache_clear()

            loaded_config = get_router_config()

            # Config should load without errors
            self.assertIsInstance(loaded_config, dict)
            model = loaded_config["model_list"][0]
            self.assertEqual(len(model["model_info"]["aliases"]), 4)
            self.assertIn("main", model["model_info"]["aliases"])
            self.assertIn("primary", model["model_info"]["aliases"])
            mock_load.assert_called_once()

    def test_alias_case_sensitivity(self):
        """
        Test that aliases are case-sensitive and "Main" and "main" are distinct.
        """
        # Config with case-sensitive aliases on different models
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["Main"]},  # Capital M
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},  # Lowercase m
                },
            ]
        }

        with (
            patch("pathlib.Path.open"),
            patch("yaml.safe_load", return_value=mock_config) as mock_load,
        ):
            get_router_config.cache_clear()

            loaded_config = get_router_config()

            # Should load successfully since "Main" and "main" are different
            self.assertIsInstance(loaded_config, dict)

            # Collect all aliases
            all_aliases = []
            for model in loaded_config["model_list"]:
                if "model_info" in model and "aliases" in model["model_info"]:
                    all_aliases.extend(model["model_info"]["aliases"])

            # Both should be present and distinct
            self.assertIn("Main", all_aliases)
            self.assertIn("main", all_aliases)
            self.assertEqual(len([a for a in all_aliases if a.lower() == "main"]), 2)
            mock_load.assert_called_once()

    def test_config_load_with_request_limit_multiplier(self):
        """
        Test that config loads successfully with request_limit_multiplier.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"], "request_limit_multiplier": 2},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["default"]},
                },
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            loaded_config = get_router_config()

            self.assertIsInstance(loaded_config, dict)
            first_model = loaded_config["model_list"][0]
            self.assertEqual(first_model["model_info"]["request_limit_multiplier"], 2)

    def test_get_model_request_limit_multiplier_returns_configured_value(self):
        """
        Test that get_model_request_limit_multiplier returns the configured value.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"], "request_limit_multiplier": 1.5},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            multiplier = get_model_request_limit_multiplier("gpt-4.1-nano")
            self.assertEqual(multiplier, 1.5)

    def test_get_model_request_limit_multiplier_returns_default_for_unconfigured(self):
        """
        Test that get_model_request_limit_multiplier returns 1.0 for models without multiplier.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["default"]},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            multiplier = get_model_request_limit_multiplier("gpt-4o")
            self.assertEqual(multiplier, 1.0)

    def test_get_model_request_limit_multiplier_resolves_alias(self):
        """
        Test that get_model_request_limit_multiplier resolves aliases correctly.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"], "request_limit_multiplier": 2},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            # Should resolve alias and return the multiplier
            multiplier = get_model_request_limit_multiplier("main")
            self.assertEqual(multiplier, 2.0)

    def test_get_model_request_limit_multiplier_returns_1_for_unknown_model(self):
        """
        Test that get_model_request_limit_multiplier returns 1.0 for unknown models.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["default"]},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            multiplier = get_model_request_limit_multiplier("unknown-model")
            self.assertEqual(multiplier, 1.0)

    def test_get_model_supports_response_format_returns_false_when_configured(self):
        """
        Test that get_model_supports_response_format returns False when explicitly configured.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-image-1",
                    "litellm_params": {
                        "model": "openai/gpt-image-1",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"supports_response_format": False},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            self.assertFalse(get_model_supports_response_format("gpt-image-1"))

    def test_get_model_supports_response_format_defaults_to_true(self):
        """
        Test that get_model_supports_response_format defaults to True when not configured.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "dall-e-2",
                    "litellm_params": {
                        "model": "openai/dall-e-2",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["image"]},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            self.assertTrue(get_model_supports_response_format("dall-e-2"))

    def test_get_model_supports_response_format_resolves_alias(self):
        """
        Test that get_model_supports_response_format resolves aliases correctly.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-image-1",
                    "litellm_params": {
                        "model": "openai/gpt-image-1",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["img"], "supports_response_format": False},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            # Should resolve the alias and return the configured value
            self.assertFalse(get_model_supports_response_format("img"))

    def test_get_model_supports_response_format_true_for_unknown_model(self):
        """
        Test that get_model_supports_response_format returns True for unknown models.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["default"]},
                }
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()

            self.assertTrue(get_model_supports_response_format("unknown-model"))

    def test_get_all_model_request_limit_multipliers(self):
        """
        Test that get_all_model_request_limit_multipliers returns multipliers for all models.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"], "request_limit_multiplier": 2.0},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["default"]},
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"mode": "embedding", "request_limit_multiplier": 0.5},
                },
            ]
        }

        with patch("pathlib.Path.open"), patch("yaml.safe_load", return_value=mock_config):
            get_router_config.cache_clear()
            get_all_model_request_limit_multipliers.cache_clear()

            multipliers = get_all_model_request_limit_multipliers()

            self.assertEqual(multipliers["gpt-4.1-nano"], 2.0)
            self.assertEqual(multipliers["gpt-4o"], 1.0)
            self.assertEqual(multipliers["text-embedding-ada-002"], 0.5)
            self.assertEqual(len(multipliers), 3)


class ModelAliasRoutingTest(GatewayIntegrationTestCase):
    """
    Tests for model alias resolution during API requests.
    These tests check that aliases are correctly resolved to model names.
    """

    tts_model = "gpt-4o-mini-tts"
    stt_model = "whisper-1"
    image_gen_model = "dall-e-2"

    def setUp(self):
        super().setUp()
        get_router_config.cache_clear()

    def test_chat_completion_with_alias(self):
        """
        Test that chat completion requests using an alias are routed correctly.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ]

        # Try to use alias instead of model name
        payload = {"model": "main", "messages": messages, "max_tokens": 10}

        response = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working. Expected 200, "
            f"got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)

    def test_speech_endpoint_with_alias(self):
        """Test that TTS endpoint correctly resolves model aliases."""
        payload = {
            "model": "tts",  # Using alias instead of actual model name
            "input": "Testing alias resolution for text-to-speech.",
            "voice": "alloy",
            "response_format": "mp3",
        }

        response = self.client.post(
            "/audio/speech",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for TTS. Expected 200, got {response.status_code}",
        )

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after speech generation."
        )

    def test_transcriptions_endpoint_with_alias(self):
        """Test that STT endpoint correctly resolves model aliases."""
        # Create a simple audio file for testing
        test_audio_content = b"mock audio data"
        test_audio_file = SimpleUploadedFile(
            "test.mp3", test_audio_content, content_type="audio/mp3"
        )
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        # Remove Content-Type for multipart form data
        headers.pop("Content-Type", None)

        # Using alias instead of actual model name
        response = self.client.post(
            "/audio/transcriptions", {"file": test_audio_file, "model": "stt"}, headers=headers
        )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for STT. Expected 200, "
            f"got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        self.assertIn("text", response_json, "Response should contain 'text' field")

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after transcription."
        )

    def test_chat_completion_with_nonexistent_alias(self):
        """
        Test that requests with non-existent aliases return appropriate errors.
        """
        messages = [{"role": "user", "content": "Hello!"}]
        payload = {"model": "nonexistent-alias-12345", "messages": messages, "max_tokens": 10}

        response = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_embeddings_with_alias(self):
        """
        Test that embedding requests using an alias are routed correctly.
        """
        payload = {"model": "embedding", "input": ["The quick brown fox."]}

        response = self.client.post(
            "/embeddings",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for embeddings. "
            f"Expected 200, got {response.status_code}",
        )

        response_json = response.json()
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

    def test_image_generation_with_alias(self):
        """Test that image generation endpoint correctly resolves model aliases."""
        payload = {
            "model": "image",  # Using alias instead of actual model name
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
        }

        response = self.client.post(
            "/images/generations",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Should return 200 with alias resolution
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        self.assertIn("data", response_json, "Response should contain 'data' field")
        self.assertIsInstance(response_json["data"], list, "Data should be a list")
        self.assertGreater(len(response_json["data"]), 0, "Data should not be empty")

        # Check first image object structure
        img_data = response_json["data"][0]
        self.assertIn("b64_json", img_data, "Image data should contain 'b64_json' field")
        self.assertIsInstance(img_data["b64_json"], str, "Image data should be a string")

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after image generation."
        )

    def test_multiple_aliases_same_model(self):
        """
        Test that multiple aliases for the same model all work correctly.
        """
        messages = [{"role": "user", "content": "Test"}]
        aliases_to_test = ["main", "coding"]

        first_status = None
        for alias in aliases_to_test:
            payload = {"model": alias, "messages": messages, "max_tokens": 5}

            response = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

            # All should behave the same (either all work or all fail)
            if first_status is None:
                # Remember result for first alias
                first_status = response.status_code
            else:
                # Others should match
                self.assertEqual(
                    response.status_code,
                    first_status,
                    f"Alias '{alias}' returned {response.status_code}, "
                    f"but first alias returned {first_status}",
                )

    def test_excluded_model_alias_rejected(self):
        """
        Test that aliases for excluded models are rejected.
        """
        # Exclude the main model for the organization
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()

        messages = [{"role": "user", "content": "Test"}]

        # Test with actual model name
        payload = {"model": self.model, "messages": messages, "max_tokens": 5}

        response_actual = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Test with alias
        payload = {"model": "main", "messages": messages, "max_tokens": 5}

        response_alias = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Both should be rejected (404)
        self.assertEqual(
            response_actual.status_code,
            404,
            f"Expected 404 for excluded model, got {response_actual.status_code}",
        )

        # Alias should also be rejected since it resolves to excluded model
        self.assertEqual(
            response_alias.status_code,
            404,
            f"Alias to excluded model should return 404, got {response_alias.status_code}",
        )

    def test_alias_case_sensitivity_in_requests(self):
        """
        Test that alias case sensitivity is preserved in actual requests.
        """
        messages = [{"role": "user", "content": "Test"}]

        # Test with correct case
        payload = {"model": "main", "messages": messages, "max_tokens": 5}

        response_lowercase = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # Should return 200 with alias resolution
        self.assertEqual(
            response_lowercase.status_code,
            200,
            f"Alias resolution not working for lowercase. Expected 200, "
            f"got {response_lowercase.status_code}",
        )

        # Test with wrong case
        payload = {"model": "Main", "messages": messages, "max_tokens": 5}
        response_uppercase = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # "Main" should fail since only "main" is defined (case-sensitive)
        self.assertIn(
            response_uppercase.status_code,
            [400, 404],
            f"Wrong case alias 'Main' should fail, got {response_uppercase.status_code}",
        )

    def test_request_stores_resolved_model_name(self):
        """
        Test that when a request is made with an alias, the Request object
        stored in the database has the actual model name, not the alias.
        """

        endpoints = [
            {
                "name": "chat_completions",
                "path": "/chat/completions",
                "alias": "main",
                "resolved_model": "gpt-4.1-nano",
                "payload": {
                    "model": "main",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 5,
                },
            },
            {
                "name": "completions",
                "path": "/completions",
                "alias": "main",
                "resolved_model": "gpt-4.1-nano",
                "payload": {"model": "main", "prompt": "Test prompt", "max_tokens": 5},
            },
            {
                "name": "embeddings",
                "path": "/embeddings",
                "alias": "embedding",
                "resolved_model": "text-embedding-ada-002",
                "payload": {"model": "embedding", "input": ["Test input"]},
            },
            {
                "name": "speech",
                "path": "/audio/speech",
                "alias": "tts",
                "resolved_model": "gpt-4o-mini-tts",
                "payload": {"model": "tts", "input": "Test", "voice": "alloy"},
            },
            {
                "name": "image_generation",
                "path": "/images/generations",
                "alias": "image",
                "resolved_model": "dall-e-2",
                "payload": {"model": "image", "prompt": "Test image", "size": "256x256"},
            },
            {
                "name": "transcriptions",
                "path": "/audio/transcriptions",
                "alias": "stt",
                "resolved_model": "whisper-1",
                "payload": None,
                "multipart": True,
            },
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint["name"]):
                Request.objects.all().delete()

                if endpoint.get("multipart"):
                    test_file = SimpleUploadedFile(
                        "test.mp3", b"mock audio", content_type="audio/mp3"
                    )
                    headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
                    headers.pop("Content-Type", None)
                    response = self.client.post(
                        endpoint["path"],
                        {"file": test_file, "model": endpoint["alias"]},
                        headers=headers,
                    )
                else:
                    response = self.client.post(
                        endpoint["path"],
                        data=json.dumps(endpoint["payload"]),
                        headers=self.headers,
                        content_type="application/json",
                    )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"{endpoint['name']}: Expected 200, got {response.status_code}",
                )

                request_log = Request.objects.first()
                self.assertIsNotNone(request_log, f"{endpoint['name']}: Request should be logged")
                self.assertEqual(
                    request_log.model,
                    endpoint["resolved_model"],
                    f"{endpoint['name']}: Request model should be '{endpoint['resolved_model']}' "
                    f"(resolved from alias '{endpoint['alias']}'), got '{request_log.model}'",
                )

    def test_invalid_alias_returns_error_and_logs_request(self):
        """
        Test that requests with invalid aliases return appropriate error
        and the request is logged with the unresolved model name.
        """
        messages = [{"role": "user", "content": "Test"}]
        payload = {"model": "nonexistent-alias-xyz", "messages": messages, "max_tokens": 5}

        response = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        request_log = Request.objects.first()
        self.assertIsNotNone(request_log, "Request should be logged even for invalid alias")
        self.assertEqual(
            request_log.model,
            "nonexistent-alias-xyz",
            "Invalid alias should be stored as-is since it cannot be resolved",
        )
