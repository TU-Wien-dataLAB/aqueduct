"""Unit tests for the cache-backed rate limiter (``gateway/rate_limiting.py``).

These tests use the in-process LocMemCache (the configured cache under TESTING)
and patch ``gateway.rate_limiting.timezone.now`` to drive window rollover and
multi-minute hour/day accumulation deterministically.
"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from gateway.rate_limiting import WINDOWS, _bucket_key, check_and_reserve, record_token_usage
from management.models import LimitSet, Usage

TOKEN_ID = 42


class _Clock:
    """Mutable clock used to drive ``rate_limiting.timezone.now``."""

    def __init__(self, start: datetime):
        self.t = start

    def now(self) -> datetime:
        return self.t

    def advance_minutes(self, n: int) -> None:
        self.t = self.t + timedelta(minutes=n)


def _patch_time(clock: _Clock):
    return patch("gateway.rate_limiting.timezone.now", side_effect=clock.now)


@contextmanager
def _yielding(value: bool):
    """Context manager that yields a fixed lock-acquisition status."""
    yield value


def _lock_disabled():
    """Patch that makes the per-token lock appear contended (never acquired)."""
    return patch("gateway.rate_limiting._cache_lock", return_value=_yielding(False))


class RateLimitingBase(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def tearDown(self):
        cache.clear()


class CheckAndReserveTest(RateLimitingBase):
    def test_no_limits_is_noop_and_allows(self):
        limits = LimitSet()  # all None
        allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertTrue(allowed)
        self.assertEqual(exceeded, [])
        # No bucket should have been written.
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        for name, _secs in WINDOWS:
            self.assertIsNone(cache.get(_bucket_key(TOKEN_ID, name, now)))

    def test_reserves_request_slot_and_then_blocks(self):
        limits = LimitSet(requests_per_minute=3)
        # Requests 1-3 allowed (delta=1.0 each: checks see 0, 1.0, 2.0 < 3), 4th blocked (3.0 >= 3).
        for i in range(3):
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
            self.assertTrue(allowed, f"request {i + 1} should be allowed")
            self.assertEqual(exceeded, [])
        allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertFalse(allowed)
        self.assertEqual(exceeded, ["Request limit (3/min)"])

    def test_weighted_budget_uses_model_multiplier(self):
        limits = LimitSet(requests_per_minute=3)
        # multiplier 2.0 -> delta 0.5 per request; 6 allowed (0..2.5 < 3), 7th blocked (3.0 >= 3).
        with patch("gateway.rate_limiting.get_model_request_limit_multiplier", return_value=2.0):
            for i in range(6):
                allowed, _ = check_and_reserve(limits, TOKEN_ID, model="gpt-4.1-nano")
                self.assertTrue(allowed, f"request {i + 1} should be allowed")
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model="gpt-4.1-nano")
        self.assertFalse(allowed)
        self.assertEqual(exceeded, ["Request limit (3/min)"])

    def test_mixed_model_weighted_budget_accumulates(self):
        limits = LimitSet(requests_per_minute=3)
        multipliers = {"expensive": 0.5, "cheap": 2.0}  # deltas 2.0 and 0.5
        with patch(
            "gateway.rate_limiting.get_model_request_limit_multiplier",
            side_effect=lambda m: multipliers[m],
        ):
            # expensive: delta 2.0 -> bucket 2.0 (<3) allowed
            allowed, _ = check_and_reserve(limits, TOKEN_ID, model="expensive")
            self.assertTrue(allowed)
            # cheap: delta 0.5 -> bucket 2.5 (<3) allowed
            allowed, _ = check_and_reserve(limits, TOKEN_ID, model="cheap")
            self.assertTrue(allowed)
            # cheap again: bucket 2.5 (<3) allowed -> bucket 3.0
            allowed, _ = check_and_reserve(limits, TOKEN_ID, model="cheap")
            self.assertTrue(allowed)
            # cheap again: bucket 3.0 (>=3) blocked
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model="cheap")
        self.assertFalse(allowed)
        self.assertEqual(exceeded, ["Request limit (3/min)"])

    def test_input_token_limit_blocks(self):
        limits = LimitSet(input_tokens_per_minute=100)
        # Pre-seed the current minute bucket with 100 input tokens.
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            record_token_usage(TOKEN_ID, Usage(input_tokens=100, output_tokens=0))
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertFalse(allowed)
        self.assertEqual(exceeded, ["Input token limit (100/min)"])

    def test_lock_contention_fails_open_for_check(self):
        limits = LimitSet(requests_per_minute=3)
        with _lock_disabled():
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertTrue(allowed)
        self.assertEqual(exceeded, [])
        # Nothing reserved since we failed open.
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        self.assertIsNone(cache.get(_bucket_key(TOKEN_ID, "min", now)))


class RecordTokenUsageTest(RateLimitingBase):
    def test_zero_usage_is_noop(self):
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            record_token_usage(TOKEN_ID, Usage(input_tokens=0, output_tokens=0))
        for name, _secs in WINDOWS:
            self.assertIsNone(cache.get(_bucket_key(TOKEN_ID, name, now)))

    def test_increments_all_three_windows(self):
        now = datetime(2026, 7, 2, 10, 5, 30, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            record_token_usage(TOKEN_ID, Usage(input_tokens=7, output_tokens=11))
        for name, _secs in WINDOWS:
            bucket = cache.get(_bucket_key(TOKEN_ID, name, now))
            self.assertIsNotNone(bucket, f"{name} bucket should exist")
            assert bucket is not None  # for mypy
            self.assertEqual(bucket["in"], 7)
            self.assertEqual(bucket["out"], 11)
            self.assertEqual(bucket["req"], 0.0)

    def test_lock_contention_skips_recording(self):
        with _lock_disabled():
            record_token_usage(TOKEN_ID, Usage(input_tokens=5, output_tokens=5))
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        for name, _secs in WINDOWS:
            self.assertIsNone(cache.get(_bucket_key(TOKEN_ID, name, now)))

    def test_ttl_set_on_first_write(self):
        """Buckets are written with a per-window TTL of 2 x window seconds."""
        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            record_token_usage(TOKEN_ID, Usage(input_tokens=1, output_tokens=1))
        # We cannot easily expire LocMem by TTL within a test, but clearing the
        # cache makes the buckets disappear (confirms they live in the cache, not
        # in some module-level state), and the keys are window-scoped.
        cache.clear()
        for name, _secs in WINDOWS:
            self.assertIsNone(cache.get(_bucket_key(TOKEN_ID, name, now)))


class GetAggregateUsageTest(RateLimitingBase):
    def test_empty_token_list_returns_zero_buckets_for_every_window(self):
        from gateway.rate_limiting import get_aggregate_usage

        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            usage = get_aggregate_usage([])
        self.assertEqual(set(usage), {name for name, _secs in WINDOWS})
        for name, _secs in WINDOWS:
            self.assertEqual(usage[name], {"req": 0.0, "in": 0, "out": 0})

    def test_sums_reserved_requests_and_recorded_tokens_across_tokens(self):
        from gateway.rate_limiting import get_aggregate_usage

        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        limits = LimitSet(requests_per_minute=10)
        token_a, token_b = TOKEN_ID, TOKEN_ID + 1
        with _patch_time(_Clock(now)):
            # Token A: reserve 2 request slots + record 7 in / 11 out tokens.
            self.assertTrue(check_and_reserve(limits, token_a, model=None)[0])
            self.assertTrue(check_and_reserve(limits, token_a, model=None)[0])
            record_token_usage(token_a, Usage(input_tokens=7, output_tokens=11))
            # Token B: reserve 3 request slots + record 4 in / 5 out tokens.
            for _ in range(3):
                self.assertTrue(check_and_reserve(limits, token_b, model=None)[0])
            record_token_usage(token_b, Usage(input_tokens=4, output_tokens=5))
            usage = get_aggregate_usage([token_a, token_b])
        for name, _secs in WINDOWS:
            self.assertEqual(usage[name]["req"], 5.0)  # 2 + 3
            self.assertEqual(usage[name]["in"], 11)  # 7 + 4
            self.assertEqual(usage[name]["out"], 16)  # 11 + 5

    def test_single_token_matches_its_own_buckets(self):
        from gateway.rate_limiting import get_aggregate_usage

        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        limits = LimitSet(requests_per_minute=10)
        with _patch_time(_Clock(now)):
            self.assertTrue(check_and_reserve(limits, TOKEN_ID, model=None)[0])
            record_token_usage(TOKEN_ID, Usage(input_tokens=7, output_tokens=11))
            usage = get_aggregate_usage([TOKEN_ID])
        for name, _secs in WINDOWS:
            self.assertEqual(usage[name]["req"], 1.0)
            self.assertEqual(usage[name]["in"], 7)
            self.assertEqual(usage[name]["out"], 11)

    def test_separate_minute_buckets_per_window_id(self):
        limits = LimitSet(requests_per_minute=2)
        clock = _Clock(datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC))
        with _patch_time(clock):
            # Two requests in minute 10:00 -> both allowed, minute bucket at 2.0.
            self.assertTrue(check_and_reserve(limits, TOKEN_ID, model=None)[0])
            self.assertTrue(check_and_reserve(limits, TOKEN_ID, model=None)[0])
            # 3rd in same minute -> blocked.
            self.assertFalse(check_and_reserve(limits, TOKEN_ID, model=None)[0])

            clock.advance_minutes(1)  # new minute window_id, same hour bucket
            # Minute bucket is fresh, so a new request is allowed even though the
            # previous minute's bucket was at its limit.
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertTrue(allowed)
        self.assertEqual(exceeded, [])


class GetPerTokenUsageTest(RateLimitingBase):
    def test_empty_token_list_returns_no_entries(self):
        from gateway.rate_limiting import get_per_token_usage

        self.assertEqual(get_per_token_usage([]), {})

    def test_returns_per_token_buckets_without_summing(self):
        from gateway.rate_limiting import get_per_token_usage

        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        limits = LimitSet(requests_per_minute=10)
        token_a, token_b = TOKEN_ID, TOKEN_ID + 1
        with _patch_time(_Clock(now)):
            # Token A: reserve 2 request slots + record 7 in / 11 out tokens.
            self.assertTrue(check_and_reserve(limits, token_a, model=None)[0])
            self.assertTrue(check_and_reserve(limits, token_a, model=None)[0])
            record_token_usage(token_a, Usage(input_tokens=7, output_tokens=11))
            # Token B: reserve 3 request slots + record 4 in / 5 out tokens.
            for _ in range(3):
                self.assertTrue(check_and_reserve(limits, token_b, model=None)[0])
            record_token_usage(token_b, Usage(input_tokens=4, output_tokens=5))
            per_token = get_per_token_usage([token_a, token_b])
        self.assertEqual(set(per_token), {token_a, token_b})
        for name, _secs in WINDOWS:
            self.assertEqual(per_token[token_a][name]["req"], 2.0)
            self.assertEqual(per_token[token_a][name]["in"], 7)
            self.assertEqual(per_token[token_a][name]["out"], 11)
            self.assertEqual(per_token[token_b][name]["req"], 3.0)
            self.assertEqual(per_token[token_b][name]["in"], 4)
            self.assertEqual(per_token[token_b][name]["out"], 5)

    def test_token_with_no_recorded_usage_gets_zero_buckets(self):
        from gateway.rate_limiting import get_per_token_usage

        now = datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC)
        with _patch_time(_Clock(now)):
            per_token = get_per_token_usage([TOKEN_ID])
        self.assertEqual(set(per_token), {TOKEN_ID})
        for name, _secs in WINDOWS:
            self.assertEqual(per_token[TOKEN_ID][name], {"req": 0.0, "in": 0, "out": 0})


class HourDayLimitTest(RateLimitingBase):
    def test_hour_limit_blocks_after_sustained_burst(self):
        """requests_per_minute=10, hourly_limit_multiplier=6 -> 60/hour.

        Drive 60 requests across 6 different minutes (minute cap never hit), then
        a 61st request in a fresh minute is blocked by the hour bucket.
        """
        limits = LimitSet(requests_per_minute=10, hourly_limit_multiplier=6)
        clock = _Clock(datetime(2026, 7, 2, 10, 0, 0, tzinfo=UTC))
        with _patch_time(clock):
            for _minute in range(6):
                for _ in range(10):
                    allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
                    self.assertTrue(allowed, f"minute {_minute} request should be allowed")
                    self.assertEqual(exceeded, [])
                clock.advance_minutes(1)  # still within hour 10

            # 61st request in a fresh minute: minute ok, hour bucket (60) >= 60 -> blocked.
            allowed, exceeded = check_and_reserve(limits, TOKEN_ID, model=None)
        self.assertFalse(allowed)
        self.assertEqual(exceeded, ["Request limit (60/hour)"])

    def test_day_limit_derived_from_per_minute(self):
        limits = LimitSet(requests_per_minute=10, daily_limit_multiplier=6)
        windows = limits.windows()
        # min, hour, day
        self.assertEqual(windows[0][2], 10)  # rpm
        self.assertEqual(windows[1][2], 10 * 60)  # hour uses settings default 60
        self.assertEqual(windows[2][2], 10 * 6)  # day uses the set multiplier 6


class LimitSetResolutionTest(SimpleTestCase):
    def test_from_objects_resolves_multipliers_specific_over_org(self):
        org = SimpleNamespace(
            requests_per_minute=10,
            input_tokens_per_minute=None,
            output_tokens_per_minute=None,
            hourly_limit_multiplier=6,
            daily_limit_multiplier=1440,
        )
        team = SimpleNamespace(
            requests_per_minute=None,
            input_tokens_per_minute=None,
            output_tokens_per_minute=None,
            hourly_limit_multiplier=3,  # overrides org
            daily_limit_multiplier=None,  # falls back to org
        )
        limits = LimitSet.from_objects(team, org)
        self.assertEqual(limits.requests_per_minute, 10)  # from org
        self.assertEqual(limits.hourly_limit_multiplier, 3)  # from team
        self.assertEqual(limits.daily_limit_multiplier, 1440)  # from org

    @override_settings(AQUEDUCT_HOURLY_LIMIT_MULTIPLIER=60, AQUEDUCT_DAILY_LIMIT_MULTIPLIER=1440)
    def test_windows_uses_settings_defaults_when_multiplier_unset(self):
        limits = LimitSet(requests_per_minute=10)
        windows = limits.windows()
        self.assertEqual(windows[0][2], 10)  # min rpm
        self.assertEqual(windows[1][2], 10 * 60)  # hour default
        self.assertEqual(windows[2][2], 10 * 1440)  # day default

    def test_windows_none_per_minute_implies_none_derived(self):
        limits = LimitSet(input_tokens_per_minute=5)  # requests/output unset
        windows = limits.windows()
        # requests_per_minute is None -> hour/day rpm also None (tied design).
        self.assertIsNone(windows[0][2])
        self.assertIsNone(windows[1][2])
        self.assertIsNone(windows[2][2])
        # input_tokens scales.
        self.assertEqual(windows[0][3], 5)
        self.assertEqual(windows[1][3], 5 * 60)  # settings default hourly mult
        self.assertEqual(windows[2][3], 5 * 1440)  # settings default daily mult

    def test_has_no_limit_fast_path(self):
        # Re-import the helper used by the decorator to gate the cache call.
        from gateway.views.decorators import _has_any_limit

        self.assertFalse(_has_any_limit(LimitSet()))
        self.assertTrue(_has_any_limit(LimitSet(requests_per_minute=1)))
        self.assertTrue(_has_any_limit(LimitSet(input_tokens_per_minute=1)))
