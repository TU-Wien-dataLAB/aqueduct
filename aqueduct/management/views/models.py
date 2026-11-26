from copy import deepcopy
from typing import Any

from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.utils.translation import gettext_lazy as _
from django.views.generic import TemplateView

from gateway.config import get_router_config

CATEGORIES = {
    "chat": (_("Chat"), {"chat", "completion", "responses"}),
    "embeddings": (_("Embeddings"), {"embedding", "embeddings"}),
    "audio": (_("Audio"), {"audio_speech", "audio_transcription", "audio_transcriptions"}),
    "images": (_("Images"), {"image_generation", "image_generations"}),
    "other": (_("Other"), set()),
}
"""Each model type's tab label and the `model_info.mode` values it holds."""


class ModelListView(LoginRequiredMixin, TemplateView):
    """The router's models by type, searchable by name and alias, with one model's details."""

    template_name = "management/model_list.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        config = deepcopy(get_router_config())
        models: dict[str, dict[str, Any]] = {}
        # LiteLLM balances load over entries that repeat a name; the first one describes it.
        for model in config.get("model_list", []):
            models.setdefault(model["model_name"], model)
        for model in models.values():
            litellm_params: dict[str, Any] = model.get("litellm_params") or {}
            if not litellm_params.get("api_key", "").startswith("os.environ/"):
                litellm_params["api_key"] = "*********"
            info = model["model_info"] = dict(model.get("model_info") or {})
            if "max_input_tokens" not in info:
                info["max_input_tokens"] = info.get("max_tokens")
            # LiteLLM treats a model without a mode as chat.
            model["mode"] = str(info.get("mode") or "chat").strip().lower()
            model["category"] = next(
                (key for key, (_label, modes) in CATEGORIES.items() if model["mode"] in modes),
                "other",
            )
            model["names"] = [model["model_name"], *map(str, info.get("aliases") or [])]
            card_url = info.get("model_card_url")
            if card_url:
                try:
                    URLValidator(schemes=["http", "https"])(card_url)
                except ValidationError:
                    pass
                else:
                    model["model_card_url"] = card_url

        params = self.request.GET
        category = params.get("type", "") if params.get("type") in CATEGORIES else ""
        query = params.get("q", "").strip()
        listed = [
            model
            for model in models.values()
            if category in ("", model["category"])
            and any(query.casefold() in name.casefold() for name in model["names"])
        ]
        wanted = params.get("model", "")
        context.update(
            categories=[
                (key, label)
                for key, (label, _modes) in CATEGORIES.items()
                if any(model["category"] == key for model in models.values())
            ],
            category=category,
            query=query,
            model_list=listed,
            selected=next(
                (model for model in models.values() if wanted in model["names"]),
                next(iter(listed or models.values()), None),
            ),
            default_parallel=config.get("default_max_parallel_requests"),
            site_host=f"{self.request.scheme}://{self.request.get_host()}",
        )
        return context
