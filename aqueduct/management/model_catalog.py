import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum, auto
from typing import Any

from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import QuerySet
from django.urls import reverse
from django.utils.formats import number_format
from django.utils.functional import Promise
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
from litellm.types.utils import LlmProviders

from gateway.config import get_router_config, model_entries, model_info_with_context, model_names
from management.formatting import and_join, compact, grouped
from management.models import Team, UserProfile
from management.models.exclusions import Exclusion, excluded


class Category(StrEnum):
    CHAT = auto()
    EMBEDDINGS = auto()
    AUDIO = auto()
    IMAGES = auto()
    OTHER = auto()

    @property
    def label(self) -> Promise:
        return _CATEGORY_LABELS[self]

    @classmethod
    def parse(cls, raw: str) -> "Category | None":
        try:
            return cls(raw)
        except ValueError:
            return None


CONTEXT_LENGTH = _("Context length")
_CATEGORY_LABELS = {
    Category.CHAT: _("Chat"),
    Category.EMBEDDINGS: _("Embeddings"),
    Category.AUDIO: _("Audio"),
    Category.IMAGES: _("Images"),
    Category.OTHER: _("Other"),
}


@dataclass(frozen=True)
class Mode:
    category: Category
    label: str | Promise
    spellings: tuple[str, ...]
    endpoint: str | None = None
    fields: tuple[tuple[str, object], ...] = ()
    form: bool = False
    output: str | None = None
    context_label: str | Promise = CONTEXT_LENGTH


MODES: tuple[Mode, ...] = (
    Mode(
        Category.CHAT,
        _("Chat"),
        ("chat",),
        "gateway:chat_completions",
        (("messages", [{"role": "user", "content": "Hello"}]),),
    ),
    Mode(
        Category.CHAT,
        _("Completion"),
        ("completion",),
        "gateway:completions",
        (("prompt", "Hello"),),
    ),
    Mode(Category.CHAT, _("Responses"), ("responses",), "gateway:responses", (("input", "Hello"),)),
    Mode(
        Category.EMBEDDINGS,
        _("Embeddings"),
        ("embedding", "embeddings"),
        "gateway:embeddings",
        (("input", "Hello"),),
        context_label=_("Max input tokens"),
    ),
    Mode(
        Category.AUDIO,
        _("Text to speech"),
        ("audio_speech",),
        "gateway:speech",
        (("input", "Hello"), ("voice", "alloy")),
        output="speech.mp3",
    ),
    Mode(
        Category.AUDIO,
        _("Speech to text"),
        ("audio_transcription", "audio_transcriptions"),
        "gateway:transcriptions",
        (("file", "@speech.mp3"),),
        form=True,
    ),
    Mode(
        Category.IMAGES,
        _("Images"),
        ("image_generation", "image_generations"),
        "gateway:image_generation",
        (("prompt", "A lighthouse at dawn"),),
    ),
    Mode(
        Category.OTHER,
        _("Reranking"),
        ("rerank",),
        "gateway:rerank",
        (("query", "Capital of France?"), ("documents", ["Paris", "Berlin"])),
    ),
)
OTHER_MODE = Mode(Category.OTHER, _("Other"), ())
_MODE_BY_SPELLING = {spelling: mode for mode in MODES for spelling in mode.spellings}


def mode_for(raw: object) -> Mode:
    # LiteLLM treats a model without a mode as chat.
    if raw is None or raw == "":
        return _MODE_BY_SPELLING["chat"]
    return _MODE_BY_SPELLING.get(str(raw).strip().lower(), OTHER_MODE)


def _count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _tokens(value: object) -> str:
    count = _count(value)
    return "" if count is None else compact(count)


def _number(value: object) -> str:
    count = _count(value)
    return "" if count is None else grouped(count)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


_MODALITIES = {"text": _("text"), "image": _("image"), "audio": _("audio"), "video": _("video")}


def _modalities(value: object) -> str:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        return ""
    joined = ", ".join(str(_MODALITIES.get(v.lower(), v.lower())) for v in value)
    return joined[:1].upper() + joined[1:]


def _date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


_SPECS: tuple[tuple[str | Promise, str, Callable[[object], str]], ...] = (
    (_("Quantization"), "quantization", _text),
    ("", "max_input_tokens", _number),
    (_("Max output tokens"), "max_output_tokens", _number),
    (_("Dimensions"), "output_vector_size", _number),
    (_("Input"), "supported_modalities", _modalities),
    (_("Output"), "supported_output_modalities", _modalities),
    (_("Parallel requests"), "max_parallel_requests", _number),
)
_CAPABILITIES: tuple[tuple[str, str | Promise], ...] = (
    ("supports_vision", _("Vision")),
    ("supports_function_calling", _("Tools")),
    ("supports_response_schema", _("JSON schema")),
    ("supports_reasoning", _("Reasoning")),
    ("supports_web_search", _("Web search")),
    ("supports_pdf_input", _("PDF input")),
    ("supports_audio_input", _("Audio input")),
    ("supports_audio_output", _("Audio output")),
    ("supports_prompt_caching", _("Prompt caching")),
    ("supports_native_streaming", _("Streaming")),
)


_PROVIDERS = frozenset(provider.value for provider in LlmProviders)
_REPO_PROVIDERS = frozenset({"openai", "hosted_vllm", "vllm", "huggingface", "openai_like"})
_REPO = re.compile(r"(?P<org>[A-Za-z0-9][\w.-]*)/[A-Za-z0-9][\w.-]*")
_QUANTIZATIONS = {
    spelling.casefold(): spelling
    for spelling in (
        "AWQ", "GPTQ", "GGUF", "EXL2", "EXL3", "BNB", "FP8", "FP4", "NVFP4", "MXFP4",
        "INT4", "INT8", "W4A16", "W8A8", "W8A16", "W4A8", "4bit", "8bit",
    )
}  # fmt: skip


def hugging_face_repo(served: str | None, name: str) -> str | None:
    provider, _, rest = (served or "").partition("/")
    if provider in _PROVIDERS:
        if provider not in _REPO_PROVIDERS:
            return None
        if _REPO.fullmatch(rest):
            return rest
    elif served and _REPO.fullmatch(served):
        return served
    match = _REPO.fullmatch(name)
    return name if match and match["org"] not in _PROVIDERS else None


def quantization_in(name: str) -> str:
    words = re.split(r"[-_.]", name.rsplit("/", 1)[-1])
    found = (_QUANTIZATIONS.get(word.casefold()) for word in words)
    return " ".join(dict.fromkeys(spelling for spelling in found if spelling))


_is_web_url = URLValidator(schemes=["http", "https"])


def _web_url(value: object) -> str | None:
    try:
        _is_web_url(value)
    except ValidationError:
        return None
    return str(value)


def _levels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(level, str) for level in value):
        return ()
    return tuple(value)


@dataclass(frozen=True)
class Spec:
    label: str | Promise
    value: str


@dataclass(frozen=True)
class Scope:
    team: str | None
    excluded: frozenset[str]

    @property
    def name(self) -> str:
        if self.team:
            return gettext("%(team)s service accounts") % {"team": self.team}
        return gettext("your personal keys")


def split_scopes(name: str, scopes: Sequence[Scope]) -> tuple[tuple[Scope, ...], tuple[Scope, ...]]:
    return (
        tuple(scope for scope in scopes if name in scope.excluded),
        tuple(scope for scope in scopes if name not in scope.excluded),
    )


@dataclass(frozen=True)
class Availability:
    off_for: tuple[Scope, ...]
    on_for: tuple[Scope, ...]

    @property
    def usable(self) -> bool:
        return bool(self.on_for) or not self.off_for

    @property
    def available_with(self) -> str:
        return and_join([scope.name for scope in self.on_for]) if self.off_for else ""


@dataclass(frozen=True)
class ModelCard(Availability):
    name: str
    aliases: tuple[str, ...]
    mode: Mode
    description: str | None
    model_card_url: str | None
    quantization: str | None
    served_model: str | None
    context: str | None
    specs: tuple[Spec, ...]
    capabilities: tuple[str | Promise, ...]
    thinking_levels: tuple[str, ...]
    deprecation_date: date | None

    @property
    def quantization_badge(self) -> str | None:
        shown = self.name.casefold()
        if self.quantization and not all(
            word.casefold() in shown for word in self.quantization.split()
        ):
            return self.quantization
        return None

    def matches(self, text: str) -> bool:
        needle = text.casefold()
        return any(needle in key.casefold() for key in (self.name, *self.aliases))

    def curl(self, base_url: str, model: str | None = None) -> str | None:
        mode = self.mode
        if mode.endpoint is None:
            return None
        fields = {"model": model or self.name, **dict(mode.fields)}
        lines = [
            f"curl {base_url}{reverse(mode.endpoint)}",
            '-H "Authorization: Bearer $OPENAI_API_KEY"',
        ]
        if mode.form:
            lines += [f"-F {shlex.quote(f'{key}={value}')}" for key, value in fields.items()]
        else:
            members = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in fields.items())
            body = "{\n" + members + "\n  }"
            lines += ['-H "Content-Type: application/json"', f"-d {shlex.quote(body)}"]
        if mode.output:
            lines.append(f"--output {mode.output}")
        return " \\\n  ".join(lines)


@dataclass(frozen=True)
class Catalog:
    cards: tuple[ModelCard, ...]

    @property
    def categories(self) -> tuple[Category, ...]:
        present = {card.mode.category for card in self.cards}
        return tuple(category for category in Category if category in present)

    @property
    def has_context(self) -> bool:
        return any(card.context for card in self.cards)

    @property
    def has_aliases(self) -> bool:
        return any(card.aliases for card in self.cards)

    @property
    def columns(self) -> int:
        return 1 + self.has_aliases + self.has_context

    def search(self, category: Category | None, text: str) -> tuple[ModelCard, ...]:
        return tuple(
            card
            for card in self.cards
            if category in (None, card.mode.category) and card.matches(text)
        )

    def find(self, key: str) -> ModelCard | None:
        return next((card for card in self.cards if key in (card.name, *card.aliases)), None)


def parse_card(
    entry: Mapping[str, Any], scopes: Sequence[Scope] = (), router_parallel: int | None = None
) -> ModelCard:
    info = model_info_with_context(entry)
    # The keys upstream documents for this page win over LiteLLM's.
    for key, documented in (
        ("max_input_tokens", "contextWindow"),
        ("max_output_tokens", "maxTokens"),
    ):
        if _count(info.get(documented)) is not None:
            info[key] = info[documented]
    name: str = entry["model_name"]
    mode = mode_for(info.get("mode"))
    litellm_params = entry.get("litellm_params") or {}
    # The gateway enforces the `model_info` value, then the router's default; LiteLLM reads the
    # deployment's own.
    if info.get("max_parallel_requests") is None:
        deployment = litellm_params.get("max_parallel_requests")
        info["max_parallel_requests"] = deployment if router_parallel is None else router_parallel
    raw_served = litellm_params.get("model")
    served = raw_served if isinstance(raw_served, str) and raw_served else None
    repo = hugging_face_repo(served, name)
    quantization = _text(info.get("quantization")) or quantization_in(repo or served or name)
    info["quantization"] = quantization
    specs = tuple(
        Spec(label or mode.context_label, value)
        for label, key, format_value in _SPECS
        if (value := format_value(info.get(key)))
    )
    capabilities = tuple(label for key, label in _CAPABILITIES if info.get(key) is True)
    multiplier = info.get("request_limit_multiplier")
    if isinstance(multiplier, int | float) and not isinstance(multiplier, bool) and multiplier != 1:
        capabilities += (
            gettext("%(multiplier)sx request limit")
            % {"multiplier": number_format(Decimal(f"{multiplier:g}"))},
        )
    off_for, on_for = split_scopes(name, scopes)
    model_card_url = _web_url(info.get("model_card_url"))
    return ModelCard(
        name=name,
        aliases=tuple(info.get("aliases") or ()),
        mode=mode,
        description=_text(info.get("description")) or None,
        model_card_url=model_card_url or (repo and f"https://huggingface.co/{repo}"),
        quantization=quantization or None,
        served_model=served,
        context=_tokens(info.get("max_input_tokens")) or None,
        specs=specs,
        capabilities=capabilities,
        thinking_levels=_levels(info.get("thinkingLevels")),
        deprecation_date=_date(info.get("deprecation_date")),
        off_for=off_for,
        on_for=on_for,
    )


def scopes_for(
    profile: UserProfile, teams: QuerySet[Team], kind: Exclusion = Exclusion.MODELS
) -> tuple[Scope, ...]:
    personal = Scope(None, excluded(profile, kind))
    found = teams.select_related("org").order_by("name")
    return (personal, *(Scope(team.name, excluded(team, kind)) for team in found))


def usable_models(scopes: Sequence[Scope]) -> list[str]:
    return [name for name in model_names() if Availability(*split_scopes(name, scopes)).usable]


def model_kinds() -> dict[str, str | Promise]:
    return {
        name: mode_for(model_info_with_context(entry).get("mode")).label
        for name, entry in model_entries().items()
    }


def load_catalog(scopes: Sequence[Scope]) -> Catalog:
    router_parallel = get_router_config().get("default_max_parallel_requests")
    cards = (parse_card(entry, scopes, router_parallel) for entry in model_entries().values())
    return Catalog(tuple(card for card in cards if card.usable))
