from django.test import SimpleTestCase

from gateway.config import model_entries, model_info_with_context, model_names
from management.tests.helpers import router_config


def entry(name: str, **model_info) -> dict:
    return {
        "model_name": name,
        "litellm_params": {"model": f"openai/{name}"},
        "model_info": model_info,
    }


class ModelEntriesTest(SimpleTestCase):
    def test_a_repeated_model_name_is_listed_once_and_described_by_its_first_entry(self):
        with router_config(
            [entry("chat", mode="chat"), entry("embed"), entry("chat", mode="other")]
        ):
            self.assertEqual(list(model_entries()), ["chat", "embed"])
            self.assertEqual(model_entries()["chat"]["model_info"], {"mode": "chat"})
            self.assertEqual(model_names(), ("chat", "embed"))

    def test_a_config_without_models_has_none(self):
        with router_config([]):
            self.assertEqual(model_entries(), {})
            self.assertEqual(model_names(), ())

    def test_readers_follow_a_changed_router_config(self):
        with router_config([entry("a")]):
            self.assertEqual(model_names(), ("a",))
        with router_config([entry("b")]):
            self.assertEqual(model_names(), ("b",))


class ModelInfoWithContextTest(SimpleTestCase):
    def test_max_tokens_stands_in_for_a_missing_max_input_tokens(self):
        self.assertEqual(
            model_info_with_context(entry("m", max_tokens=8192)),
            {"max_tokens": 8192, "max_input_tokens": 8192},
        )

    def test_an_explicit_max_input_tokens_wins(self):
        model = entry("m", max_tokens=8192, max_input_tokens=1000)
        self.assertEqual(model_info_with_context(model)["max_input_tokens"], 1000)

    def test_a_non_integer_max_tokens_is_not_copied(self):
        for value in ("8192", 8192.0, True, None):
            with self.subTest(value=value):
                self.assertNotIn(
                    "max_input_tokens", model_info_with_context(entry("m", max_tokens=value))
                )

    def test_the_configured_entry_is_left_untouched(self):
        model = entry("m", max_tokens=8192)
        model_info_with_context(model)
        self.assertEqual(model["model_info"], {"max_tokens": 8192})

    def test_an_entry_without_model_info_gives_an_empty_dict(self):
        self.assertEqual(model_info_with_context({"model_name": "m"}), {})
