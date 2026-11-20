from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from django.utils.formats import date_format, get_format, number_format
from django.utils.translation import gettext, ngettext, pgettext, pgettext_lazy
from django.utils.translation import gettext_lazy as _

MISSING = "—"
NO_ORG = _("No organization")
SECOND_MS = 1000
_STEP = 1000
_SCALES = (
    pgettext_lazy("compact number", "%(number)sK"),
    pgettext_lazy("compact number", "%(number)sM"),
    pgettext_lazy("compact number", "%(number)sB"),
    pgettext_lazy("compact number", "%(number)sT"),
)


def _rounded(value: float | Decimal, places: int) -> Decimal:
    # Round the number as written, half up, so 1.15 shows as 1.2; adding 0 drops a negative zero.
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP) + 0


def grouped(count: int) -> str:
    return number_format(count, force_grouping=True)


def decimal(value: float | Decimal, places: int) -> str:
    return number_format(_rounded(value, places))


def compact(count: int) -> str:
    magnitude, scale = Decimal(abs(count)), None
    for next_scale in _SCALES:
        if _rounded(magnitude, 1) < _STEP:
            break
        magnitude, scale = magnitude / _STEP, next_scale
    digits = decimal(magnitude, 1).removesuffix(f"{get_format('DECIMAL_SEPARATOR')}0")
    number = ("-" if count < 0 else "") + digits
    return number if scale is None else scale % {"number": number}


def and_join(items: list[str]) -> str:
    *head, last = items or [""]
    if not head:
        return last
    return gettext("%(items)s and %(last)s") % {"items": ", ".join(head), "last": last}


def as_percent(number: float, places: int = 0) -> str:
    """`number` as a percent, with `places` decimals that a whole number leaves out."""
    if places:
        shown = decimal(number, places).removesuffix(
            f"{get_format('DECIMAL_SEPARATOR')}{'0' * places}"
        )
    else:
        shown = grouped(int(_rounded(number, 0)))
    return pgettext("percent", "%(number)s%%") % {"number": shown}


def percent(part: int, whole: int) -> str:
    return as_percent(100 * part / whole) if whole else MISSING


def as_tokens(count: float) -> str:
    n = round(count)
    return ngettext("%(count)s token", "%(count)s tokens", n) % {"count": compact(n)}


def as_input(count: float) -> str:
    n = round(count)
    return ngettext("%(count)s input token", "%(count)s input tokens", n) % {"count": compact(n)}


def as_output(count: float) -> str:
    n = round(count)
    return ngettext("%(count)s output token", "%(count)s output tokens", n) % {"count": compact(n)}


def as_requests(count: float) -> str:
    n = round(count)
    return ngettext("%(count)s request", "%(count)s requests", n) % {"count": grouped(n)}


def duration(ms: int | None) -> str:
    if ms is None:
        return gettext("n/a")
    return f"{grouped(ms)} ms" if ms < SECOND_MS else f"{decimal(Decimal(ms) / SECOND_MS, 1)} s"


def day_label(day: date) -> str:
    return date_format(day, "j M")
