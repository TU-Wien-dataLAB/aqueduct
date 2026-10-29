import asyncio

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase, override_settings
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from gateway.decorators.limits import RedisConcurrencyLimiter, concurrency_key


@override_settings(
    AQUEDUCT_DEFAULT_CONCURRENT_REQUESTS_LIMIT=3, AQUEDUCT_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS=120
)
class RedisConcurrencyLimiterTests(SimpleTestCase):
    """Unit tests for RedisConcurrencyLimiter, using fakeredis.

    The Lua scripts (``acquire_script`` / ``release_script``) are executed
    by fakeredis, so no further mocks are needed.
    """

    def setUp(self) -> None:
        self.server = FakeServer()
        self.redis = FakeRedis(server=self.server, decode_responses=True)
        self.limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=30)
        self.user_id = 42
        self.key = concurrency_key(self.user_id)
        self.user_id_2 = 43
        self.key_2 = concurrency_key(self.user_id_2)

    # ------------------------------ acquire ------------------------------
    @async_to_sync
    async def test_acquire_allows_up_to_limit(self):
        limit = 3
        for _ in range(limit):
            self.assertTrue(await self.limiter.acquire(self.user_id))
        self.assertFalse(await self.limiter.acquire(self.user_id))

    @async_to_sync
    async def test_acquire_counter_does_not_exceed_limit(self):
        limit = 3
        for _ in range(limit + 1):
            await self.limiter.acquire(self.user_id)
        # The over-limit INCR is undone by the script; counter stays at limit.
        self.assertEqual(int(await self.redis.get(self.key)), limit)

    @async_to_sync
    async def test_acquire_sets_ttl(self):
        await self.limiter.acquire(self.user_id)
        self.assertEqual(await self.redis.ttl(self.key), 120)

    @async_to_sync
    async def test_acquire_users_are_independent(self):
        limit = 3
        for _ in range(limit):
            await self.limiter.acquire(self.user_id)
        self.assertTrue(await self.limiter.acquire(self.user_id_2))
        self.assertEqual(int(await self.redis.get(self.key)), limit)
        self.assertEqual(int(await self.redis.get(self.key_2)), 1)

    # ------------------------------ release ------------------------------
    @async_to_sync
    async def test_release_decrements_counter(self):
        await self.limiter.acquire(self.user_id)
        release = await self.limiter.release(self.user_id)
        self.assertTrue(release)
        self.assertEqual(int(await self.redis.get(self.key)), 0)

    @async_to_sync
    async def test_release_does_not_go_negative(self):
        # acquire once, release twice -> release returns False and counter stays at 0.
        await self.limiter.acquire(self.user_id)
        await self.limiter.release(self.user_id)

        release = await self.limiter.release(self.user_id)
        self.assertFalse(release)

        counter = await self.redis.get(self.key)
        self.assertEqual(int(counter), 0)

    @async_to_sync
    async def test_release_decrements_its_own_key_counter(self):
        await self.limiter.acquire(self.user_id)
        await self.limiter.acquire(self.user_id_2)
        await self.limiter.release(self.user_id)
        self.assertEqual(int(await self.redis.get(self.key)), 0)
        self.assertEqual(int(await self.redis.get(self.key_2)), 1)

    # ---------------------------- wrap_stream ----------------------------
    async def _collect_stream(self, stream):
        return [chunk async for chunk in self.limiter.wrap_stream(stream, self.user_id)]

    @async_to_sync
    async def test_wrap_stream_yields_all_chunks_and_releases(self):
        await self.limiter.acquire(self.user_id)
        self.assertEqual(int(await self.redis.get(self.key)), 1)

        async def stream():
            yield b"chunk-1"
            yield b"chunk-2"

        chunks = await self._collect_stream(self.limiter.wrap_stream(stream(), self.user_id))
        self.assertEqual(chunks, [b"chunk-1", b"chunk-2"])
        # The slot is released once the stream is fully consumed.
        counter = await self.redis.get(self.key)
        self.assertEqual(int(counter), 0)

    @async_to_sync
    async def test_wrap_stream_releases_if_stream_raises(self):
        await self.limiter.acquire(self.user_id)

        async def stream():
            yield b"chunk-1"
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            await self._collect_stream(self.limiter.wrap_stream(stream(), self.user_id))

        counter = await self.redis.get(self.key)
        self.assertEqual(int(counter), 0)

    # ------------------------- _refresh_ttl_loop -------------------------
    @async_to_sync
    async def test_refresh_ttl_loop_extends_ttl(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)
        await self.redis.set(self.key, 1)
        await self.redis.expire(self.key, 1)

        stop_event = asyncio.Event()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, stop_event))
        await asyncio.sleep(0.05)
        stop_event.set()
        await task

        self.assertGreater(await self.redis.ttl(self.key), 1)  # refreshed

    @async_to_sync
    async def test_refresh_ttl_loop_stops_when_event_set(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)
        await self.redis.set(self.key, 1)
        await self.redis.expire(self.key, 2)

        stop_event = asyncio.Event()
        stop_event.set()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, stop_event))
        await task

        # Loop never ran `expire`, so TTL is *not* extended back to 120.
        self.assertLessEqual(await self.redis.ttl(self.key), 2)
