from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import ClassVar

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import dateformat
from django.utils.timezone import now
from django.utils.translation import gettext, ngettext
from django.utils.translation import gettext_lazy as _

PRESET_FIELD = "expiry_preset"


class Preset(StrEnum):
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "1y"
    NEVER = "never"
    LONGEST = "max"
    DATE = "date"

    @property
    def lifetime(self) -> timedelta | None:
        return _LIFETIMES.get(self)

    @classmethod
    def parse(cls, raw: object) -> "Preset":
        """What a submitted value means; a form posted without presets sets the date itself."""
        try:
            return cls(raw)
        except ValueError:
            return cls.DATE


_LABELS = {
    Preset.MONTH: _("30 days"),
    Preset.QUARTER: _("90 days"),
    Preset.YEAR: _("1 year"),
    Preset.NEVER: _("Never"),
    Preset.DATE: _("Pick a date"),
}
_LIFETIMES = {
    Preset.MONTH: timedelta(days=30),
    Preset.QUARTER: timedelta(days=90),
    Preset.YEAR: timedelta(days=365),
}


def offered(max_days: int) -> tuple[Preset, ...]:
    """The presets a form shows; a maximum lifetime of `max_days` leaves only those within it."""
    if max_days == -1:
        return tuple(preset for preset in Preset if preset is not Preset.LONGEST)
    within = (preset for preset, lifetime in _LIFETIMES.items() if lifetime.days < max_days)
    return (*within, Preset.LONGEST, Preset.DATE)


def label(preset: Preset, max_days: int) -> str:
    if preset is Preset.LONGEST:
        return ngettext("%(count)s day", "%(count)s days", max_days) % {"count": max_days}
    return str(_LABELS[preset])


def expires_sentence(at: datetime | None) -> str:
    if at is None:
        return gettext("Never expires")
    when = dateformat.format(at.astimezone(UTC), "j M Y, H:i")
    return gettext("Expires %(date)s") % {"date": f"{when} UTC"}


def sentence_template() -> str:
    """The sentence with `{date}` left for the browser to fill in."""
    return gettext("Expires %(date)s") % {"date": "{date}"}


@dataclass(frozen=True)
class PresetOption:
    preset: Preset
    label: str
    sentence: str
    checked: bool


class ExpiryForm(forms.ModelForm):
    """A form whose expiry date can also be chosen as a lifetime from the moment it is saved."""

    expiration_field: ClassVar[str]
    expiry_preset = forms.ChoiceField(
        required=False, choices=[(preset.value, preset.value) for preset in Preset]
    )

    def clean(self) -> dict:
        cleaned_data = super().clean()
        preset = Preset.parse(cleaned_data.get(PRESET_FIELD))
        if preset is not Preset.DATE:
            # Without a date the key gets the longest lifetime the server allows.
            lifetime = preset.lifetime
            cleaned_data[self.expiration_field] = now() + lifetime if lifetime else None
        elif cleaned_data.get(PRESET_FIELD) == Preset.DATE and not cleaned_data.get(
            self.expiration_field
        ):
            # A form posted without the preset field keeps the old contract: no date, no expiry.
            if self.expiration_field not in self.errors:
                self.add_error(self.expiration_field, _("Pick a date, or choose another option."))
        return cleaned_data

    @property
    def expiry_template(self) -> str:
        return sentence_template()

    @property
    def expiry_options(self) -> tuple[PresetOption, ...]:
        max_days = settings.MAX_ACCESS_KEY_EXPIRATION_DAYS
        presets = offered(max_days)
        current, picked = now(), self._picked(presets)
        return tuple(
            PresetOption(
                preset,
                label(preset, max_days),
                self._sentence(preset, current, max_days),
                preset is picked,
            )
            for preset in presets
        )

    def _picked(self, presets: tuple[Preset, ...]) -> Preset:
        if self.is_bound:
            # The date box holds the field's error, so a refused preset opens it.
            posted = Preset.parse(self.data.get(self.add_prefix(PRESET_FIELD)))
            return posted if posted in presets else Preset.DATE
        if self.initial.get(self.expiration_field):
            return Preset.DATE
        return Preset.NEVER if Preset.NEVER in presets else Preset.LONGEST

    def _sentence(self, preset: Preset, current: datetime, max_days: int) -> str:
        if preset is Preset.DATE:
            typed = self._typed_date()
            return expires_sentence(typed) if typed else ""
        lifetime = timedelta(days=max_days) if preset is Preset.LONGEST else preset.lifetime
        return expires_sentence(current + lifetime if lifetime else None)

    def _typed_date(self) -> datetime | None:
        raw = self[self.expiration_field].value()
        if isinstance(raw, datetime):
            return raw
        try:
            return self.fields[self.expiration_field].to_python(raw) if raw else None
        except ValidationError:
            return None
