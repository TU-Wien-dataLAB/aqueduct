import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase, override_settings
from redis.asyncio import Redis

from gateway.decorators.limits import RedisConcurrencyLimiter, concurrency_key


@override_settings(
    AQUEDUCT_DEFAULT_CONCURRENT_REQUESTS_LIMIT=3, AQUEDUCT_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS=120
)
class RedisConcurrencyLimiterTests(SimpleTestCase):
    """Unit tests for RedisConcurrencyLimiter.

    No real Redis server is used. The module-level Lua scripts (``acquire_script``
    / ``release_script``) and the redis client are mocked.
    """

    def setUp(self):
        self.redis = MagicMock(spec=Redis)
        # The limiter's TTL-watchdog awaits redis.expire(), so it must be an AsyncMock.
        self.redis.expire = AsyncMock(return_value=True)
        self.limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=30)
        self.user_id = 42
        self.key = concurrency_key(self.user_id)

    # ------------------------------ acquire ------------------------------
    @patch("gateway.decorators.limits.acquire_script", new_callable=AsyncMock)
    def test_acquire_returns_true_when_script_allows(self, acquire_script):
        acquire_script.return_value = 1
        self.assertTrue(asyncio.run(self.limiter.acquire(self.user_id)))

    @patch("gateway.decorators.limits.acquire_script", new_callable=AsyncMock)
    def test_acquire_returns_false_when_script_rejects(self, acquire_script):
        acquire_script.return_value = 0
        self.assertFalse(asyncio.run(self.limiter.acquire(self.user_id)))

    @patch("gateway.decorators.limits.acquire_script", new_callable=AsyncMock)
    def test_acquire_uses_user_key_limit_and_ttl(self, acquire_script):
        acquire_script.return_value = 1
        asyncio.run(self.limiter.acquire(self.user_id))
        acquire_script.assert_awaited_once_with(keys=[self.key], args=[3, 120])

    # ------------------------------ release ------------------------------
    @patch("gateway.decorators.limits.release_script", new_callable=AsyncMock)
    def test_release_returns_true_when_script_allows(self, release_script):
        release_script.return_value = 1
        self.assertTrue(asyncio.run(self.limiter.release(self.user_id)))

    @patch("gateway.decorators.limits.release_script", new_callable=AsyncMock)
    def test_release_returns_false_when_script_rejects(self, release_script):
        release_script.return_value = 0
        self.assertFalse(asyncio.run(self.limiter.release(self.user_id)))

    @patch("gateway.decorators.limits.release_script", new_callable=AsyncMock)
    def test_release_uses_user_key(self, release_script):
        release_script.return_value = 1
        asyncio.run(self.limiter.release(self.user_id))
        release_script.assert_awaited_once_with(keys=[self.key])

    # ---------------------------- wrap_stream ----------------------------
    async def _collect_stream(self, stream):
        return [chunk async for chunk in self.limiter.wrap_stream(stream, self.user_id)]

    @patch("gateway.decorators.limits.release_script", new_callable=AsyncMock)
    def test_wrap_stream_yields_all_chunks_and_releases(self, release_script):
        release_script.return_value = 1

        async def stream():
            yield b"chunk-1"
            yield b"chunk-2"

        chunks = asyncio.run(self._collect_stream(stream()))
        self.assertEqual(chunks, [b"chunk-1", b"chunk-2"])
        release_script.assert_awaited_once_with(keys=[self.key])

    @patch("gateway.decorators.limits.release_script", new_callable=AsyncMock)
    def test_wrap_stream_releases_if_stream_raises(self, release_script):
        release_script.return_value = 1

        async def stream():
            yield b"chunk-1"
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            asyncio.run(self._collect_stream(stream()))

        release_script.assert_awaited_once_with(keys=[self.key])

    # ------------------------- _refresh_ttl_loop -------------------------
    @async_to_sync
    async def test_refresh_ttl_loop_extends_ttl(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)

        stop_event = asyncio.Event()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, stop_event))
        await asyncio.sleep(0.05)
        calls = self.redis.expire.await_count
        stop_event.set()
        await task

        self.assertGreater(calls, 0)
        self.redis.expire.assert_awaited_with(self.key, limiter.ttl)

    @async_to_sync
    async def test_refresh_ttl_loop_stops_when_event_set(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)

        stop_event = asyncio.Event()
        stop_event.set()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, stop_event))
        await task
        calls = self.redis.expire.await_count

        self.assertEqual(calls, 0)
