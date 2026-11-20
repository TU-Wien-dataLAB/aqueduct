import html
import json
import re
import shlex
from typing import ClassVar
from urllib.parse import urlsplit

import yaml
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import resolve, reverse

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.model_catalog import (
    MODES,
    OTHER_MODE,
    Catalog,
    Category,
    Scope,
    hugging_face_repo,
    mode_for,
    parse_card,
    quantization_in,
)
from management.models import Org, Team, Token, UserGroup, UserProfile
from management.tests.helpers import router_config

User = get_user_model()

EXAMPLE_ENTRIES = yaml.safe_load(ROUTER_CONFIG_PATH.read_text())["model_list"]

RICH_ENTRY = {
    "model_name": "rich",
    "litellm_params": {"model": "hosted_vllm/meta-llama/Llama-3.3-70B"},
    "model_info": {
        "aliases": ["main"],
        "max_tokens": 131072,
        "max_output_tokens": 16384,
        "output_vector_size": 1024,
        "supported_modalities": ["text", "image"],
        "supported_output_modalities": ["text"],
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_response_schema": True,
        "supports_reasoning": True,
        "supports_web_search": True,
        "supports_pdf_input": True,
        "supports_audio_input": True,
        "supports_audio_output": True,
        "supports_prompt_caching": True,
        "supports_native_streaming": True,
        "request_limit_multiplier": 0.5,
        "max_parallel_requests": 4,
        "deprecation_date": "2026-11-30",
    },
}

GATEWAY = "http://testserver/v1"


def entry(name="m", **model_info):
    return {
        "model_name": name,
        "litellm_params": {"model": f"openai/{name}"},
        "model_info": model_info,
    }


def personal(*excluded):
    return Scope(None, frozenset(excluded))


def team(name, *excluded):
    return Scope(name, frozenset(excluded))


def curl_args(snippet: str) -> list[str]:
    return shlex.split(snippet.replace("\\\n", " "))


class ModelCardTest(SimpleTestCase):
    def test_example_config_shows_no_metadata_it_does_not_have(self):
        for raw in EXAMPLE_ENTRIES:
            with self.subTest(model=raw["model_name"]):
                card = parse_card(raw)
                self.assertEqual([spec.value for spec in card.specs], [None] * len(card.specs))
                self.assertEqual(
                    (card.description, card.model_card_url, card.quantization), (None,) * 3
                )
                self.assertIsNone(card.context)
                self.assertIsNone(card.deprecation_date)
                self.assertEqual(card.served_model, f"openai/{raw['model_name']}")
                expected = ("2x request limit",) if card.name == "text-embedding-ada-002" else ()
                self.assertEqual(card.capabilities, expected)

    def test_rich_entry_shows_every_cell(self):
        card = parse_card(RICH_ENTRY)
        self.assertEqual(
            [(spec.label, spec.value) for spec in card.specs],
            [
                ("Context length", "131,072"),
                ("Max output tokens", "16,384"),
                ("Dimensions", "1,024"),
                ("Input", "Text, image"),
                ("Output", "Text"),
                ("Parallel requests", "4"),
            ],
        )
        self.assertEqual(card.context, "131.1K")
        self.assertEqual(
            card.capabilities,
            (
                "Vision",
                "Tools",
                "JSON schema",
                "Reasoning",
                "Web search",
                "PDF input",
                "Audio input",
                "Audio output",
                "Prompt caching",
                "Streaming",
                "0.5x request limit",
            ),
        )
        self.assertEqual(card.served_model, "hosted_vllm/meta-llama/Llama-3.3-70B")
        self.assertEqual(str(card.deprecation_date), "2026-11-30")

    def test_wrongly_typed_values_drop_out(self):
        card = parse_card(
            entry(
                max_tokens="128k",
                max_output_tokens=True,
                output_vector_size=-1,
                supported_modalities="text",
                supports_vision="yes",
                request_limit_multiplier="2",
                deprecation_date="soon",
                description=["not", "text"],
                model_card_url="javascript:alert(1)",
                quantization=4,
            )
        )
        self.assertEqual((card.specs, card.capabilities, card.context), ((), (), None))
        self.assertEqual((card.description, card.model_card_url, card.quantization), (None,) * 3)
        self.assertIsNone(card.deprecation_date)

    def test_a_model_without_model_info_renders_as_unknown(self):
        card = parse_card({"model_name": "bare", "litellm_params": {"model": "openai/bare"}})
        self.assertEqual(card.specs, ())
        self.assertEqual(card.aliases, ())

    def test_parallel_requests_follow_the_gateway_then_the_deployment(self):
        raw = entry()
        raw["litellm_params"]["max_parallel_requests"] = 2
        self.assertEqual(
            [(spec.label, spec.value) for spec in parse_card(raw).specs],
            [("Parallel requests", "2")],
        )
        self.assertEqual([spec.value for spec in parse_card(raw, router_parallel=5).specs], ["5"])
        raw["model_info"]["max_parallel_requests"] = 6
        self.assertEqual([spec.value for spec in parse_card(raw, router_parallel=5).specs], ["6"])

    def test_explicit_context_wins_over_max_tokens(self):
        card = parse_card(entry(max_input_tokens=1_047_576, max_tokens=8192))
        self.assertEqual(card.context, "1M")

    def test_embeddings_label_their_context_as_max_input_tokens(self):
        card = parse_card(entry(mode="embedding", max_tokens=8191))
        self.assertEqual(
            [(spec.label, spec.value) for spec in card.specs], [("Max input tokens", "8,191")]
        )
        self.assertEqual(parse_card(entry(mode="audio_speech")).specs, ())


class ModelSourceTest(SimpleTestCase):
    def test_hugging_face_repo_comes_from_a_vllm_route_or_the_model_name(self):
        cases = {
            ("openai/Qwen/Qwen3.8-27B", "local-qwen"): "Qwen/Qwen3.8-27B",
            ("hosted_vllm/openai/gpt-oss-120b", "gpt-oss"): "openai/gpt-oss-120b",
            ("hosted_vllm/local-glm", "zai-org/GLM-5.3"): "zai-org/GLM-5.3",
            ("openai/local-qwen", "Qwen/Qwen3.8-27B"): "Qwen/Qwen3.8-27B",
            (None, "zai-org/GLM-5.3"): "zai-org/GLM-5.3",
        }
        for (served, name), repo in cases.items():
            with self.subTest(served=served, name=name):
                self.assertEqual(hugging_face_repo(served, name), repo)

    def test_routes_to_providers_with_their_own_catalog_link_nothing(self):
        cases = (
            ("openrouter/meta-llama/Llama-3.3-70B", "meta-llama/Llama-3.3-70B"),
            ("bedrock/meta.llama3-70b-instruct-v1:0", "meta-llama/Llama-3-70B"),
            ("together_ai/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
            ("openai/gpt-4.1-nano", "gpt-4.1-nano"),
            ("openai/gpt-4.1", "openai/gpt-4.1"),
            ("openai/a/b/c", "has space/x"),
        )
        for served, name in cases:
            with self.subTest(served=served):
                self.assertIsNone(hugging_face_repo(served, name))
                self.assertIsNone(
                    parse_card(entry(name) | {"litellm_params": {"model": served}}).model_card_url
                )

    def test_quantization_comes_from_the_repo_name(self):
        cases = {
            "Qwen/Qwen3.8-27B-FP8": "FP8",
            "QuantTrio/Qwen3.6-35B-A3B-AWQ": "AWQ",
            "Qwen/Qwen3-32B-GPTQ-Int4": "GPTQ INT4",
            "unsloth/Llama-3.3-70B-bnb-4bit": "BNB 4bit",
            "zai-org/GLM-5.3": "",
            "FP8-labs/plain-model": "",
        }
        for name, quantization in cases.items():
            with self.subTest(name=name):
                self.assertEqual(quantization_in(name), quantization)

    def test_the_quantization_badge_shows_only_when_the_name_does_not_say_it(self):
        named = parse_card(entry("Qwen/Qwen3.8-27B-FP8"))
        alias = parse_card(
            entry("fast")
            | {"litellm_params": {"model": "hosted_vllm/QuantTrio/Qwen3.6-35B-A3B-AWQ"}}
        )
        self.assertEqual((named.quantization, named.quantization_badge), ("FP8", None))
        self.assertEqual((alias.quantization, alias.quantization_badge), ("AWQ", "AWQ"))

    def test_card_links_the_hugging_face_repo_behind_the_route(self):
        card = parse_card(
            entry("local-awq", aliases=["fast"])
            | {"litellm_params": {"model": "hosted_vllm/QuantTrio/Qwen3.6-35B-A3B-AWQ"}}
        )
        self.assertEqual(
            card.model_card_url, "https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ"
        )
        self.assertEqual(card.quantization, "AWQ")
        self.assertEqual(card.specs[0].value, "AWQ")

    def test_configured_fields_win_over_derived_ones(self):
        card = parse_card(
            entry(
                "Qwen/Qwen3.8-27B-FP8",
                description="  General chat.  ",
                model_card_url="https://example.org/card",
                quantization="FP8 (dynamic)",
            )
        )
        self.assertEqual(card.description, "General chat.")
        self.assertEqual(card.model_card_url, "https://example.org/card")
        self.assertEqual(card.quantization, "FP8 (dynamic)")


class ModeTest(SimpleTestCase):
    def test_every_spelling_maps_to_its_mode(self):
        for mode in MODES:
            for spelling in mode.spellings:
                with self.subTest(spelling=spelling):
                    self.assertIs(mode_for(spelling), mode)
                    self.assertIs(mode_for(f" {spelling.upper()} "), mode)

    def test_singular_and_plural_agree(self):
        for singular in ("embedding", "audio_transcription", "image_generation"):
            with self.subTest(singular=singular):
                self.assertIs(mode_for(singular), mode_for(f"{singular}s"))
                self.assertIsNot(mode_for(singular), OTHER_MODE)

    def test_missing_mode_is_chat(self):
        for raw in (None, ""):
            self.assertEqual(mode_for(raw).label, "Chat")

    def test_unknown_mode_is_other_with_no_snippet(self):
        card = parse_card(entry(mode="moderation"))
        self.assertEqual(card.mode.category, Category.OTHER)
        self.assertIsNone(card.curl(GATEWAY))

    def test_snippet_targets_the_gateway_route_for_its_mode(self):
        for mode in MODES:
            with self.subTest(mode=mode.label):
                card = parse_card(entry("model-x", mode=mode.spellings[0]))
                url = curl_args(card.curl(GATEWAY))[1]
                self.assertTrue(url.startswith(GATEWAY))
                match = resolve(urlsplit(url).path)
                self.assertEqual(match.namespace, "gateway")
                self.assertEqual(match.url_name, f"v1_{mode.endpoint.removeprefix('gateway:')}")

    def test_snippet_sends_the_model_name(self):
        for mode in MODES:
            with self.subTest(mode=mode.label):
                name = 'it\'s "quoted"'
                args = curl_args(parse_card(entry(name, mode=mode.spellings[0])).curl(GATEWAY))
                if mode.form:
                    self.assertIn(f"model={name}", args)
                else:
                    self.assertEqual(json.loads(args[args.index("-d") + 1])["model"], name)


class AvailabilityTest(SimpleTestCase):
    def test_usable_while_one_scope_allows_it(self):
        cases = [
            ((), True),
            ((personal(),), True),
            ((personal("m"),), False),
            ((personal("m"), team("Whale")), True),
            ((personal("m"), team("Whale", "m")), False),
        ]
        for scopes, usable in cases:
            with self.subTest(scopes=scopes):
                self.assertEqual(parse_card(entry("m"), scopes).usable, usable)

    def test_names_the_keys_only_when_some_cannot_use_it(self):
        cases = [
            ((personal(), team("Whale")), ""),
            ((personal(), team("Orca", "m"), team("Whale", "m")), "your personal keys"),
            (
                (personal("m"), team("Orca"), team("Whale")),
                "Orca service accounts and Whale service accounts",
            ),
        ]
        for scopes, keys in cases:
            with self.subTest(keys=keys):
                self.assertEqual(parse_card(entry("m"), scopes).available_with, keys)

    def test_an_excluded_alias_hides_nothing(self):
        card = parse_card(entry("m", aliases=["a"]), (personal("a"),))
        self.assertTrue(card.usable)


class CatalogTest(SimpleTestCase):
    catalog = Catalog(tuple(parse_card(raw) for raw in EXAMPLE_ENTRIES))

    def test_lists_only_categories_present(self):
        self.assertEqual(
            self.catalog.categories,
            (Category.CHAT, Category.EMBEDDINGS, Category.AUDIO, Category.IMAGES, Category.OTHER),
        )
        self.assertEqual(Catalog((parse_card(entry()),)).categories, (Category.CHAT,))

    def test_context_column_only_when_some_model_has_one(self):
        self.assertFalse(self.catalog.has_context)
        self.assertTrue(Catalog((parse_card(entry(max_tokens=8192)),)).has_context)

    def test_alias_column_only_when_some_model_has_one(self):
        self.assertTrue(self.catalog.has_aliases)
        self.assertFalse(Catalog((parse_card(entry("org/model-AWQ")),)).has_aliases)

    def test_search_matches_names_and_aliases_within_a_category(self):
        names = [card.name for card in self.catalog.search(None, "STT")]
        self.assertEqual(names, ["whisper-1"])
        images = [card.name for card in self.catalog.search(Category.IMAGES, "")]
        self.assertEqual(images, ["dall-e-2", "gpt-image-1"])
        self.assertEqual(self.catalog.search(Category.CHAT, "image"), ())

    def test_find_by_name_or_exact_alias(self):
        self.assertEqual(self.catalog.find("tts").name, "gpt-4o-mini-tts")
        self.assertEqual(self.catalog.find("whisper-1").name, "whisper-1")
        self.assertIsNone(self.catalog.find("TTS"))

    def test_parse_unknown_category_is_all(self):
        self.assertEqual(Category.parse("images"), Category.IMAGES)
        for raw in ("", "Images", "video"):
            self.assertIsNone(Category.parse(raw))


def rows(response) -> list[str]:
    return re.findall(r'#model-detail"[^>]*>([^<]+)</a>', response.content.decode())


def selected(response) -> str:
    return selected_panel(response.content.decode())


def selected_panel(content: str) -> str:
    return re.search(r'<h2 id="model-detail-name-panel"[^>]*>([^<]+)</h2>', content).group(1)


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class ModelsPageTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self):
        self.org = Org.objects.get(name="E060")
        self.whale = Team.objects.get(name="Whale")

    def page(self, user_pk=3, **params):
        self.client.force_login(User.objects.get(pk=user_pk))
        response = self.client.get(reverse("models"), params)
        self.assertEqual(response.status_code, 200)
        return response

    def make_unavailable(self, holder, *models, merge=True):
        holder.excluded_models = list(models)
        holder.merge_exclusion_lists = merge
        holder.save()

    def test_user_sees_every_model_when_nothing_is_unavailable(self):
        response = self.page()
        self.assertEqual(rows(response), [raw["model_name"] for raw in EXAMPLE_ENTRIES])
        self.assertEqual(selected(response), "gpt-4.1-nano")
        self.assertNotContains(response, "Available with")
        self.assertContains(response, "curl http://testserver/v1/chat/completions")

    def test_a_model_one_of_the_keys_may_use_names_those_keys(self):
        self.make_unavailable(self.whale, "whisper-1")
        response = self.page(model="whisper-1")
        self.assertIn("whisper-1", rows(response))
        self.assertContains(response, "Available with your personal keys.")

    def test_a_model_none_of_the_keys_may_use_is_not_listed(self):
        self.make_unavailable(self.whale, "whisper-1")
        self.make_unavailable(UserProfile.objects.get(user_id=3), "whisper-1")
        response = self.page(model="whisper-1")
        self.assertNotIn("whisper-1", rows(response))
        self.assertNotEqual(selected(response), "whisper-1")
        self.assertNotContains(response, "whisper-1")
        self.assertEqual(self.page(q="stt").context["cards"], ())

    def test_another_teams_exclusion_hides_nothing(self):
        self.make_unavailable(self.whale, "whisper-1")
        outsider = User.objects.create_user("outsider", "outsider@example.com")
        UserProfile.objects.create(user=outsider, org=self.org)
        response = self.page(user_pk=outsider.pk, model="whisper-1")
        self.assertEqual(selected(response), "whisper-1")
        self.assertNotContains(response, "Available with")

    def test_org_exclusion_reaches_only_scopes_that_merge(self):
        self.make_unavailable(self.org, "dall-e-2")
        self.assertNotIn("dall-e-2", rows(self.page()))
        self.make_unavailable(UserProfile.objects.get(user_id=3), merge=False)
        self.assertContains(self.page(model="dall-e-2"), "Available with your personal keys.")
        self.make_unavailable(self.whale, merge=False)
        self.assertNotContains(self.page(model="dall-e-2"), "Available with")

    def test_nothing_usable_says_so(self):
        self.make_unavailable(self.org, *(raw["model_name"] for raw in EXAMPLE_ENTRIES))
        self.assertContains(self.page(), "No models are available to you.")

    def test_admins_see_only_what_their_keys_may_use(self):
        self.make_unavailable(self.whale, "whisper-1")
        self.make_unavailable(UserProfile.objects.get(user_id=1), "whisper-1")
        response = self.page(user_pk=1)
        self.assertNotIn("whisper-1", rows(response))
        self.assertNotContains(response, "in admin</a>")

    def test_a_switched_platform_admin_still_counts_their_teams_keys(self):
        self.make_unavailable(UserProfile.objects.get(user_id=1), "whisper-1")
        elsewhere = Org.objects.create(name="elsewhere")
        self.client.force_login(User.objects.get(pk=1))
        self.client.post(reverse("org_switch"), {"org_id": elsewhere.pk})
        response = self.client.get(reverse("models"), {"model": "whisper-1"})
        self.assertEqual(selected(response), "whisper-1")
        self.assertContains(response, "Available with Whale service accounts.")

    def test_type_tab_lists_only_that_type(self):
        response = self.page(type="images")
        self.assertEqual(rows(response), ["dall-e-2", "gpt-image-1"])
        self.assertEqual(selected(response), "dall-e-2")
        self.assertContains(response, 'aria-current="page">Images</a>')

    def test_unknown_type_lists_every_model(self):
        self.assertEqual(len(rows(self.page(type="video"))), len(EXAMPLE_ENTRIES))

    def test_search_matches_an_alias(self):
        response = self.page(q="stt")
        self.assertEqual(rows(response), ["whisper-1"])
        self.assertEqual(selected(response), "whisper-1")

    def test_search_without_a_match_says_so(self):
        response = self.page(q="no-such-model")
        self.assertEqual(rows(response), [])
        self.assertContains(response, '<td colspan="2" class="muted">No model matches.</td>')

    def test_model_accepts_an_alias(self):
        self.assertEqual(selected(self.page(model="tts")), "gpt-4o-mini-tts")

    def test_unknown_model_selects_the_first_listed(self):
        self.assertEqual(selected(self.page(type="audio", model="nope")), "gpt-4o-mini-tts")

    def test_deep_link_to_a_filtered_out_model_still_shows_it(self):
        response = self.page(type="images", model="whisper-1")
        self.assertEqual(rows(response), ["dall-e-2", "gpt-image-1"])
        self.assertEqual(selected(response), "whisper-1")
        self.assertContains(response, "curl http://testserver/v1/audio/transcriptions")

    def page_with(self, entries, **params):
        with router_config(entries):
            return self.page(**params)

    def test_a_literal_api_key_never_reaches_the_page(self):
        raw = {**EXAMPLE_ENTRIES[0], "litellm_params": {"model": "openai/x", "api_key": "sk-leak"}}
        response = self.page_with([raw])
        self.assertContains(response, raw["model_name"])
        self.assertNotContains(response, "sk-leak")

    def test_described_model_shows_its_card_quantization_and_limits(self):
        described = entry(
            "Qwen/Qwen3.8-27B-FP8",
            aliases=["qwen", "Qwen/Qwen3.8-27B"],
            description="Fast general chat",
            max_tokens=262144,
            max_parallel_requests=8,
        )
        response = self.page_with([described, entry("bare")], model="qwen")
        self.assertEqual(selected(response), "Qwen/Qwen3.8-27B-FP8")
        self.assertContains(response, 'title="Fast general chat">Chat · Fast general chat</p>')
        self.assertContains(
            response,
            'href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8" target="_blank" '
            'rel="noopener noreferrer"',
        )
        self.assertContains(response, ">262,144</span>")
        self.assertContains(response, ">262.1K</td>")
        self.assertContains(response, ">FP8</span>")
        self.assertNotContains(response, "Max output tokens")
        self.assertContains(
            response,
            '<span class="muted text-xs">Parallel requests</span>\n'
            '                <span class="break-words font-semibold">8</span>',
        )
        self.assertContains(response, 'title="qwen, Qwen/Qwen3.8-27B"')
        self.assertNotContains(response, "badge-outline")

    def test_displays_configured_model_metadata(self):
        info = {
            "contextWindow": 131072,
            "maxTokens": 8192,
            "thinkingLevels": ["low", "high", "max"],
        }
        response = self.page_with([{"model_name": "example", "model_info": info}])
        self.assertContains(response, "Context length")
        self.assertContains(response, "131,072")
        self.assertContains(response, "Max output tokens")
        self.assertContains(response, "8,192")
        self.assertContains(response, "Thinking levels")
        for level in ("low", "high", "max"):
            self.assertContains(
                response, f'<span class="badge badge-sm bg-base-100 font-mono">{level}</span>'
            )

    def test_existing_litellm_context_limit_and_missing_metadata(self):
        entries = [
            {"model_name": "legacy", "model_info": {"max_tokens": 32768}},
            {"model_name": "no-metadata"},
        ]
        response = self.page_with(entries, model="legacy")
        self.assertContains(response, ">32,768</span>")
        self.assertNotContains(response, "Max output tokens")
        self.assertNotContains(response, "Thinking levels")
        self.assertNotContains(self.page_with(entries, model="no-metadata"), "Context length")

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

    def test_documented_metadata_keys_win_over_litellm_ones(self):
        card = parse_card(
            entry(contextWindow=200_000, max_input_tokens=1000, maxTokens=4096, max_output_tokens=1)
        )
        self.assertEqual([spec.value for spec in card.specs], ["200,000", "4,096"])
        self.assertEqual(parse_card(entry(thinkingLevels="high")).thinking_levels, ())

    def test_model_without_metadata_shows_no_empty_tiles(self):
        response = self.page_with(
            [{"model_name": "bare", "litellm_params": {"model": "openai/bare"}}]
        )
        for absent in (
            "Not configured",
            "Context length",
            "Quantization",
            "Aliases",
            "Model card",
            "Parallel requests",
            "Thinking levels",
        ):
            self.assertNotContains(response, absent)

    def test_served_model_shows_only_to_org_admins_and_admins(self):
        org_admin = UserProfile.objects.get(user_id=2)
        org_admin.group = UserGroup.ORG_ADMIN
        for user_pk, shown in ((3, False), (2, True), (1, True)):
            with self.subTest(user_pk=user_pk):
                response = self.page(user_pk=user_pk, model="whisper-1")
                self.assertEqual("Served model" in response.content.decode(), shown)
                self.assertEqual("openai/whisper-1" in response.content.decode(), shown)

    def test_the_example_is_the_only_copy_button_on_the_page(self):
        response = self.page(model="tts")
        copied = set(re.findall(r'data-copy="([^"]*)"', response.content.decode()))
        snippet = html.escape(response.context["snippet"])
        self.assertEqual(copied, {response.context["shell"].gateway_url, snippet})
        self.assertIn('-H "Authorization: Bearer $OPENAI_API_KEY"', response.context["snippet"])

    def test_detail_opens_under_the_selected_row(self):
        content = self.page(model="whisper-1").content.decode()
        row = content.index('aria-current="true">whisper-1</a>')
        inline = content.index('id="model-detail"')
        following = content.index('#model-detail"', row + 1)
        self.assertLess(row, inline)
        self.assertLess(inline, following)
        self.assertIn(" max-xl:hidden", content)

    def test_detail_links_to_the_usage_and_requests_of_the_selected_model(self):
        response = self.page(model="whisper-1")
        self.assertContains(response, f'href="{reverse("usage")}?model=whisper-1"', count=2)
        self.assertContains(response, f'href="{reverse("requests")}?model=whisper-1"', count=2)

    def test_a_deep_linked_model_outside_the_list_shows_in_the_panel(self):
        content = self.page(type="images", model="whisper-1").content.decode()
        self.assertNotIn('id="model-detail"', content)
        self.assertNotIn(" max-xl:hidden", content)
        self.assertEqual(selected_panel(content), "whisper-1")

    def test_listed_models_are_those_one_of_the_keys_may_call(self):
        self.make_unavailable(self.org, "dall-e-2")
        self.make_unavailable(
            UserProfile.objects.get(user_id=1), "gpt-4.1-nano", "tts", "whisper-1"
        )
        self.make_unavailable(self.whale, "whisper-1", "gpt-4.1-nano")
        listed = {card.name for card in self.page(user_pk=1).context["catalog"].cards}
        keys = (Token.objects.get(pk=1), Token.objects.get(pk=2))
        for raw in EXAMPLE_ENTRIES:
            name = raw["model_name"]
            with self.subTest(model=name):
                callable_by = [key.name for key in keys if not key.model_excluded(name)]
                self.assertEqual(name in listed, bool(callable_by))
        self.assertNotIn("whisper-1", listed)
        self.assertIn("gpt-4o-mini-tts", listed)
