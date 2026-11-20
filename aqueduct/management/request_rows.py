from dataclasses import dataclass

from django.utils.translation import gettext

from gateway.config import model_names
from management.formatting import duration, grouped
from management.models import Request
from management.outcomes import Outcome
from management.request_filters import Key


@dataclass(frozen=True)
class Row:
    request: Request
    key: Key
    is_open: bool

    @property
    def unknown_model(self) -> bool:
        return bool(self.request.model) and self.request.model not in model_names()

    @property
    def outcome(self) -> Outcome | None:
        return Outcome.of(self.request.status_code)

    @property
    def error(self) -> str | None:
        outcome = self.outcome
        if outcome is None:
            return gettext("No status was recorded.")
        if self.unknown_model and outcome is Outcome.CLIENT_ERROR:
            return gettext("This gateway does not serve a model with that name.")
        return None if outcome.meaning is None else str(outcome.meaning)

    @property
    def response_time(self) -> str:
        return duration(self.request.response_time_ms)

    @property
    def overhead(self) -> str | None:
        ms = self.request.processing_time_ms
        return None if ms is None else duration(ms)

    @property
    def output_speed(self) -> str | None:
        tokens, ms = self.request.output_tokens, self.request.response_time_ms
        if not tokens or not ms:
            return None
        return gettext("%(speed)s tok/s") % {"speed": grouped(round(tokens * 1000 / ms))}
