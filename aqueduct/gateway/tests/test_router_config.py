"""Public-model RPM configuration, alongside #223's concurrency metadata."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from gateway.config import (
    get_model_max_parallel_requests,
    get_model_rpm,
    get_router,
    get_router_config,
)


@override_settings(API_MAX_RETRIES=0)
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
                    "model_info": {"aliases": ["alias"], "id": "backend"},
                }
            ]
        }

    def load(self, config):
        self.path.write_text(json.dumps(config))
        get_router.cache_clear()
        get_router_config.cache_clear()
        return get_router_config()

    def test_rpm_requires_a_positive_integer(self):
        for rpm in [True, 0, -1, 1.5, "1"]:
            with self.subTest(rpm=rpm), self.assertRaisesRegex(RuntimeError, "model_info.rpm"):
                self.config["model_list"][0]["model_info"]["rpm"] = rpm
                self.load(self.config)

    def test_sdk_rpm_is_rejected_instead_of_becoming_process_local(self):
        for config in [
            {"rpm": 1},
            {"default_litellm_params": {"rpm": 1}},
            {"model_list": [{"rpm": 1}]},
            {"model_list": [{"litellm_params": {"rpm": 1}}]},
        ]:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(RuntimeError, "model_info.rpm"),
            ):
                self.load(config)

    def test_rpm_and_concurrency_metadata_coexist_without_injecting_sdk_limits(self):
        self.config["model_list"][0]["model_info"].update(rpm=2, max_parallel_requests=3)
        with patch("gateway.config.Router") as router:
            self.assertEqual(self.load(self.config), self.config)
            for name in ["public", "alias", "backend"]:
                self.assertEqual(get_model_rpm(name), 2)
            self.assertEqual(get_model_max_parallel_requests("alias"), 3)
            self.assertIsNone(get_model_rpm("unknown"))
            get_router()
            router.assert_called_once_with(**self.config, num_retries=0)

    def test_unconfigured_model_has_no_rpm_limit(self):
        self.load(self.config)
        self.assertIsNone(get_model_rpm("public"))

    def test_deployments_cannot_disagree_on_public_model_rpm(self):
        self.config["model_list"][0]["model_info"]["rpm"] = 2
        self.config["model_list"].append({"model_name": "public", "model_info": {"rpm": 3}})
        with self.assertRaisesRegex(RuntimeError, "Conflicting model_info.rpm"):
            self.load(self.config)
