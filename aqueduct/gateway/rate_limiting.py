"""Cache-backed rate limiting using fixed minute/hour/day buckets.

Per-request rate-limit checks run against the Django cache API instead of SQL
aggregates over the ``Request`` table (which remains the source of truth for
usage analytics only).

A bucket is keyed by ``(token_id, window)`` and holds a dict::

    {"req": 0.0, "in": 0, "out": 0}

``req`` is the weighted request budget (fractional, ``1/multiplier(model)`` per
request), ``in``/``out`` are input/output token counts. Buckets are *not*
sliding windows: one dedicated bucket per window with a TTL of ``2 x window`` so
a bucket created late in a window still outlives it. New windows get new
``window_id``s; stale buckets just expire.

All read-modify-write of a token's buckets happens under a per-token advisory
lock (``cache_lock``), so different tokens never contend. Floats are never
``cache.incr``'d (Redis can't ``INCR`` pickled floats); the dict + lock avoids
that limitation entirely.

Reservation vs. recording:
- ``req`` is *reserved* at check time (in ``check_and_reserve``), into all three
  windows, by adding ``1/multiplier(model)``.
- input/output tokens are *recorded* at completion time (in
  ``record_token_usage``), into the then-current windows, since they are unknown
  until the upstream call returns.
"""

import logging
from contextlib import AbstractContextManager
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from gateway.config import get_model_request_limit_multiplier
from management.models import LimitSet, Usage

log = logging.getLogger("aqueduct")

# (window_name, window_seconds) — constants, independent of the configured limits.
# Order is finest-first, which ``_evaluate`` relies on to report only the finest
# exceeded window per metric.
WINDOWS: tuple[tuple[str, int], ...] = (("min", 60), ("hour", 3600), ("day", 86400))
_WINDOW_FMT: dict[str, str] = {"min": "%Y%m%d%H%M", "hour": "%Y%m%d%H", "day": "%Y%m%d"}
_PREFIX: dict[str, str] = {"min": "rl:m", "hour": "rl:h", "day": "rl:d"}
# Message suffix per window, matching the historical "Request limit (N/min)" shape.
_SUFFIX: dict[str, str] = {"min": "min", "hour": "hour", "day": "day"}
# (bucket field, message label) — order is finest-metric reporting order.
_METRICS: tuple[tuple[str, str], ...] = (
    ("req", "Request limit"),
    ("in", "Input token limit"),
    ("out", "Output token limit"),
)


def _window_id(name: str, now: Any = None) -> str:
    return (now or timezone.now()).strftime(_WINDOW_FMT[name])


def _bucket_key(token_id: int, name: str, now: Any = None) -> str:
    return f"{_PREFIX[name]}:{token_id}:{_window_id(name, now)}"


def _lock_key(token_id: int) -> str:
    return f"rl:lock:{token_id}"


def _new_bucket() -> dict[str, float]:
    return {"req": 0.0, "in": 0, "out": 0}


def _ttl(window_seconds: int) -> int:
    return 2 * window_seconds


def _cache_lock(lock_id: str, ttl: int) -> AbstractContextManager[bool]:
    """Lazy import of ``cache_lock`` to avoid a circular import at module load.

    ``gateway.views.utils`` lives in the ``gateway.views`` package whose ``__init__``
    eagerly imports the view modules (which, in turn, import this module). Importing
    it at module top-level would therefore create a load-order cycle. By the time
    these functions run (request time) the app is fully loaded, so the import is a
    cheap dict lookup.
    """
    from gateway.views.utils import cache_lock

    return cache_lock(lock_id, ttl)


def _evaluate(
    windows: list[tuple[str, int, int | None, int | None, int | None]],
    buckets: dict[str, dict[str, float]],
    keys: list[str],
) -> list[str]:
    """Return exceeded-message list, reporting only the finest exceeded window per metric.

    ``windows`` is finest-first (min, hour, day); for each metric we stop at the
    first (finest) window whose limit is exceeded and do not report coarser ones
    for that metric.
    """
    messages: list[str] = []
    for field, label in _METRICS:
        for (name, _secs, rpm, itpm, otpm), key in zip(windows, keys, strict=True):
            limit = {"req": rpm, "in": itpm, "out": otpm}[field]
            if limit is not None and buckets[key][field] >= limit:
                messages.append(f"{label} ({limit}/{_SUFFIX[name]})")
                break  # finest exceeded window for this metric; skip coarser windows
    return messages


def check_and_reserve(limits: LimitSet, token_id: int, model: str | None) -> tuple[bool, list[str]]:
    """Read all-window buckets, enforce limits, reserve the request slot in every window.

    Returns ``(allowed, exceeded_messages)``. On rare same-token lock contention we
    fail open (allow the request) to avoid latency spikes — no worse than today's
    read races.
    """
    windows = limits.windows()
    if not any(
        rpm is not None or itpm is not None or otpm is not None
        for _name, _secs, rpm, itpm, otpm in windows
    ):
        return True, []

    now = timezone.now()
    keys = [_bucket_key(token_id, name, now) for name, _secs, *_rest in windows]
    delta = 1.0 / get_model_request_limit_multiplier(model) if model else 1.0

    with _cache_lock(_lock_key(token_id), settings.AQUEDUCT_RATE_LIMIT_LOCK_TTL_SECONDS) as got:
        if not got:
            log.warning("Rate-limit lock contention for token %s; failing open.", token_id)
            return True, []

        raw = cache.get_many(keys)
        buckets = {k: (raw.get(k) or _new_bucket()) for k in keys}

        exceeded = _evaluate(windows, buckets, keys)
        if exceeded:
            return False, exceeded

        for key in keys:
            buckets[key]["req"] += delta
        for (_name, secs, _rpm, _itpm, _otpm), key in zip(windows, keys, strict=True):
            cache.set(key, buckets[key], _ttl(secs))  # per-window TTL -> individual sets
        return True, []


def record_token_usage(token_id: int, usage: Usage) -> None:
    """Add a completed request's token usage to all current-window buckets.

    Tokens are recorded into the *completion* window (not the request-start
    window), so they are never lost to limiting for requests spanning a window
    boundary. On rare same-token lock contention we skip the recording (slight
    under-count, logged).
    """
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        return

    now = timezone.now()
    keys = [_bucket_key(token_id, name, now) for name, _secs in WINDOWS]

    with _cache_lock(_lock_key(token_id), settings.AQUEDUCT_RATE_LIMIT_LOCK_TTL_SECONDS) as got:
        if not got:
            log.warning(
                "Rate-limit lock contention for token %s while recording usage; skipping.", token_id
            )
            return

        raw = cache.get_many(keys)
        buckets = {k: (raw.get(k) or _new_bucket()) for k in keys}
        for key in keys:
            buckets[key]["in"] += usage.input_tokens
            buckets[key]["out"] += usage.output_tokens
        for (_name, secs), key in zip(WINDOWS, keys, strict=True):
            cache.set(key, buckets[key], _ttl(secs))
