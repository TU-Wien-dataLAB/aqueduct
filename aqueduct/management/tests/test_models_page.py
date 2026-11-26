import re
from unittest.mock import patch

import yaml
from django.urls import reverse

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.tests.helpers import RoleWorld

EXAMPLE_NAMES = [
    raw["model_name"] for raw in yaml.safe_load(ROUTER_CONFIG_PATH.read_text())["model_list"]
]

RICH = {
    "model_name": "rich",
    "litellm_params": {"model": "hosted_vllm/meta-llama/Llama-3.3-70B", "api_key": "sk-leak"},
    "model_info": {
        "aliases": ["main"],
        "description": "Fast general chat",
        "max_tokens": 131072,
        "max_output_tokens": 16384,
        "output_vector_size": 1024,
        "supported_modalities": ["text", "image"],
        "supported_output_modalities": ["text"],
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_reasoning": True,
        "request_limit_multiplier": 0.5,
        "max_parallel_requests": 4,
        "deprecation_date": "2026-11-30",
    },
}


def rows(response) -> list[str]:
    return re.findall(r'#model-detail"[^>]*>([^<]+)</a>', response.content.decode())


def selected(response) -> str:
    return re.search(r'<h2 id="model-detail-name"[^>]*>([^<]+)</h2>', response.content.decode())[1]


class ModelsPageTest(RoleWorld):
    def page(self, user=None, **params):
        response = self.as_user(user or self.member).get(reverse("models"), params)
        self.assertEqual(response.status_code, 200)
        return response

    def page_with(self, entries, user=None, **params):
        config = {"model_list": entries}
        with patch("management.views.models.get_router_config", return_value=config):
            return self.page(user, **params)

    def test_lists_every_model_and_selects_the_first(self):
        response = self.page()
        self.assertEqual(rows(response), EXAMPLE_NAMES)
        self.assertEqual(selected(response), "gpt-4.1-nano")
        self.assertContains(response, "curl http://testserver/v1/chat/completions")

    def test_type_tab_lists_only_that_type(self):
        response = self.page(type="images")
        self.assertEqual(rows(response), ["dall-e-2", "gpt-image-1"])
        self.assertEqual(selected(response), "dall-e-2")
        self.assertContains(response, 'aria-current="page">Images</a>')
        self.assertEqual(rows(self.page(type="other")), ["bge-reranker-v2-m3"])

    def test_unknown_type_lists_every_model(self):
        self.assertEqual(rows(self.page(type="video")), EXAMPLE_NAMES)

    def test_search_matches_names_and_aliases(self):
        response = self.page(q="STT")
        self.assertEqual(rows(response), ["whisper-1"])
        self.assertEqual(selected(response), "whisper-1")
        self.assertEqual(rows(self.page(q="gpt-4", type="audio")), ["gpt-4o-mini-tts"])

    def test_search_without_a_match_says_so(self):
        response = self.page(q="no-such-model")
        self.assertEqual(rows(response), [])
        self.assertContains(response, "No model matches.")

    def test_model_accepts_an_alias_and_falls_back_to_the_first_listed(self):
        self.assertEqual(selected(self.page(model="tts")), "gpt-4o-mini-tts")
        self.assertEqual(selected(self.page(type="audio", model="nope")), "gpt-4o-mini-tts")

    def test_deep_link_to_a_filtered_out_model_still_shows_it(self):
        response = self.page(type="images", model="whisper-1")
        self.assertEqual(rows(response), ["dall-e-2", "gpt-image-1"])
        self.assertEqual(selected(response), "whisper-1")
        self.assertContains(response, "curl http://testserver/v1/audio/transcriptions")

    def test_each_mode_has_its_example(self):
        examples = {
            "text-embedding-ada-002": "/v1/embeddings",
            "bge-reranker-v2-m3": "/v1/rerank",
            "gpt-4o-mini-tts": "/v1/audio/speech",
            "dall-e-2": "/v1/images/generations",
            "gpt-image-1": "/v1/images/generations",
        }
        for model, path in examples.items():
            with self.subTest(model=model):
                response = self.page(model=model)
                self.assertContains(response, f"curl http://testserver{path}")
                self.assertContains(response, f'"model": "{model}"')
        unknown = self.page_with([{"model_name": "odd", "model_info": {"mode": "video"}}])
        self.assertContains(unknown, ">Other</p>")
        self.assertNotContains(unknown, "curl ")

    def test_rich_model_shows_its_specs_and_capabilities(self):
        response = self.page_with([RICH], model="main")
        self.assertEqual(selected(response), "rich")
        for shown in (
            "Fast general chat",
            "Context length",
            "131,072",
            "16,384",
            "Dimensions",
            "1,024",
            "Text, image",
            "Parallel requests",
            "Vision",
            "Tools",
            "Reasoning",
            "0.5x request limit",
            "Retirement date: 2026-11-30",
            'class="badge badge-outline badge-sm flex-none uppercase">Deprecated',
        ):
            self.assertContains(response, shown)
        self.assertNotContains(response, "JSON schema")
        self.assertNotContains(response, "sk-leak")

    def test_displays_configured_model_metadata(self):
        info = {"contextWindow": 131072, "maxTokens": 8192, "thinkingLevels": ["low", "high"]}
        response = self.page_with([{"model_name": "example", "model_info": info}])
        self.assertContains(response, "Context length")
        self.assertContains(response, "131,072")
        self.assertContains(response, "Max output tokens")
        self.assertContains(response, "8,192")
        self.assertContains(response, "Thinking levels")
        self.assertContains(
            response, '<span class="badge badge-sm bg-base-100 font-mono">high</span>'
        )

    def test_existing_litellm_context_limit_and_missing_metadata(self):
        entries = [
            {"model_name": "legacy", "model_info": {"max_tokens": 32768}},
            {"model_name": "no-metadata"},
        ]
        response = self.page_with(entries, model="legacy")
        self.assertContains(response, "32,768", count=2)
        self.assertNotContains(response, "Max output tokens")
        bare = self.page_with(entries, model="no-metadata")
        for absent in ("Context length", "Parallel requests", "Aliases</span>", "Model card"):
            self.assertNotContains(bare, absent)

    def test_router_default_counts_as_parallel_requests(self):
        config = {"model_list": [{"model_name": "m"}], "default_max_parallel_requests": 7}
        with patch("management.views.models.get_router_config", return_value=config):
            response = self.page()
        self.assertContains(response, "Parallel requests")

    def test_model_card_link_requires_http_url(self):
        entries = [
            {
                "model_name": "safe",
                "model_info": {"model_card_url": "https://huggingface.co/example/model"},
            },
            {"model_name": "unsafe", "model_info": {"model_card_url": "javascript:alert(1)"}},
        ]
        response = self.page_with(entries, model="safe")
        self.assertContains(response, 'href="https://huggingface.co/example/model"')
        unsafe = self.page_with(entries, model="unsafe")
        self.assertEqual(selected(unsafe), "unsafe")
        self.assertNotContains(unsafe, "Model card")
        self.assertNotContains(unsafe, 'href="javascript:')

    def test_a_repeated_model_name_is_listed_once(self):
        entries = [{"model_name": "twice"}, {"model_name": "twice"}]
        self.assertEqual(rows(self.page_with(entries)), ["twice"])

    def test_served_model_shows_only_to_org_admins_and_admins(self):
        for user, shown in ((self.member, False), (self.org_admin, True), (self.admin, True)):
            with self.subTest(user=user.username):
                response = self.page_with([RICH], user)
                self.assertEqual("Served model" in response.content.decode(), shown)

    def test_no_models_configured(self):
        self.assertContains(self.page_with([]), "No models available")
