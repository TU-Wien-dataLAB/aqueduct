"""Configuration policy through the public loader, without Redis or background processes."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from gateway.config import get_router, get_router_config


@override_settings(AQUEDUCT_MODEL_RPM_LIMITS={}, API_MAX_RETRIES=0)
class RouterConfigTests(SimpleTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "router.json"
        settings = override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=self.path)
        settings.enable()
        self.addCleanup(settings.disable)
        self.addCleanup(get_router_config.cache_clear)
        self.addCleanup(get_router.cache_clear)
        self.config = {
            "model_list": [
                {
                    "model_name": "public",
                    "litellm_params": {"model": "openai/gpt-4.1-nano"},
                    "model_info": {"aliases": ["alias"]},
                },
                {"model_name": "public/*", "litellm_params": {"model": "openai/*"}},
            ]
        }

    def load(self, config):
        self.path.write_text(json.dumps(config))
        get_router.cache_clear()
        get_router_config.cache_clear()
        return get_router_config()

    def test_public_limits_require_canonical_names_and_positive_integer_rpm(self):
        for limits in [
            None,
            [],
            "public",
            {"public": True},
            {"public": 0},
            {"public": -1},
            {"public": 1.5},
            {"public": "1"},
            {"alias": 1},
            {"missing": 1},
            {"public/*": 1},
        ]:
            with (
                self.subTest(limits=limits),
                override_settings(AQUEDUCT_MODEL_RPM_LIMITS=limits),
                self.assertRaisesRegex(RuntimeError, "AQUEDUCT_MODEL_RPM_LIMITS"),
            ):
                self.load(self.config)

    def test_sdk_limits_are_rejected_instead_of_becoming_process_local(self):
        cases = [{"default_max_parallel_requests": 1}]
        for field in ("rpm", "tpm", "max_parallel_requests"):
            cases.extend(
                [
                    {field: 1},
                    {"default_litellm_params": {field: 1}},
                    {"model_list": [{field: 1}]},
                    {"model_list": [{"litellm_params": {field: 1}}]},
                    {"model_list": [{"model_info": {field: 1}}]},
                ]
            )
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(RuntimeError, "AQUEDUCT_MODEL_RPM_LIMITS"),
            ):
                self.load(config)

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={"public": 2})
    def test_public_limits_do_not_inject_sdk_configuration(self):
        with patch("gateway.config.Router") as router:
            self.assertEqual(self.load(self.config), self.config)
            get_router()
            router.assert_called_once_with(**self.config, num_retries=0)
