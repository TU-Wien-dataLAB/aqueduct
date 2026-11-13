import asyncio

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, override_settings
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from gateway.decorators.limits import RedisConcurrencyLimiter, token_concurrency_key
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Token

User = get_user_model()


@override_settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS=2)
class RedisConcurrencyLimiterRecoveryTests(SimpleTestCase):
    async def test_orphaned_slot_expires_despite_healthy_traffic(self):
        redis = FakeRedis(server=FakeServer(), decode_responses=True)
        limiter = RedisConcurrencyLimiter(redis)
        key = "token:concurrency:orphan-recovery"
        try:
            # Simulate a worker crash: this acquisition will never be released.
            self.assertTrue(await limiter.acquire(key, 2))

            # Healthy requests use and release the remaining slot, more frequently
            # than the TTL, for longer than the orphan's entire lease lifetime.
            for _ in range(12):
                lease = await limiter.acquire(key, 2)
                self.assertIsNotNone(lease)
                try:
                    await asyncio.sleep(0.2)
                finally:
                    await limiter.release(key, lease)

            # Both slots must now be available, even though traffic never stopped.
            acquired = []
            try:
                for _ in range(2):
                    lease = await limiter.acquire(key, 2)
                    self.assertIsNotNone(lease, "Orphaned slot did not recover under traffic")
                    acquired.append(lease)
            finally:
                for lease in acquired:
                    await limiter.release(key, lease)
        finally:
            await redis.aclose()

    async def test_heartbeat_renews_only_its_own_lease(self):
        redis = FakeRedis(server=FakeServer(), decode_responses=True)
        limiter = RedisConcurrencyLimiter(redis, refresh_interval=0.1)
        key = "token:concurrency:heartbeat-recovery"
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def stream():
            entered.set()
            await finish.wait()
            yield b"done"

        async def consume(lease):
            return [chunk async for chunk in limiter.wrap_stream(stream(), key, lease)]

        task = None
        try:
            orphan = await limiter.acquire(key, 2)
            healthy = await limiter.acquire(key, 2)
            self.assertIsNotNone(orphan)
            self.assertIsNotNone(healthy)
            task = asyncio.create_task(consume(healthy))
            await asyncio.wait_for(entered.wait(), timeout=1)
            self.assertIsNone(await limiter.acquire(key, 2))

            await asyncio.sleep(2.2)
            self.assertFalse(task.done())
            replacement = await limiter.acquire(key, 2)
            self.assertIsNotNone(replacement, "Healthy heartbeat kept the orphan alive")
            try:
                # A late release from the dead worker cannot remove the new lease.
                self.assertFalse(await limiter.release(key, orphan))
                self.assertIsNone(await limiter.acquire(key, 2))
            finally:
                await limiter.release(key, replacement)
        finally:
            finish.set()
            if task is not None:
                await asyncio.wait_for(task, timeout=1)
            await redis.aclose()


@override_settings(
    AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=3,
    AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS=120,
)
class RedisConcurrencyLimiterTests(GatewayIntegrationTestCase):
    """Unit tests for RedisConcurrencyLimiter, using fakeredis.

    The Lua scripts (``acquire_script`` / ``release_script``) are executed
    by fakeredis, so no further mocks are needed.
    """

    def setUp(self) -> None:
        self.server = FakeServer()
        self.redis = FakeRedis(server=self.server, decode_responses=True)
        self.limiter = RedisConcurrencyLimiter(self.redis)
        self.token = Token.objects.get(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN))
        self.key = token_concurrency_key(self.token)
        new_user = User.objects.create_user(username="OtherUser", email="other@example.com")
        token_2 = Token(name="TestToken", user=new_user)
        token_2._set_new_key()
        token_2.save()
        self.key_2 = token_concurrency_key(token_2)

    # ------------------------------ acquire ------------------------------
    @async_to_sync
    async def test_acquire_allows_up_to_limit(self):
        limit = 3
        for _ in range(limit):
            self.assertTrue(await self.limiter.acquire(self.key, limit))
        self.assertFalse(await self.limiter.acquire(self.key, limit))

    @async_to_sync
    async def test_acquire_counter_does_not_exceed_limit(self):
        limit = 3
        for _ in range(limit + 1):
            await self.limiter.acquire(self.key, limit)
        # Rejection does not add another lease.
        self.assertEqual(await self.redis.zcard(self.key), limit)

    @async_to_sync
    async def test_acquire_sets_ttl(self):
        await self.limiter.acquire(self.key, 42)
        self.assertEqual(await self.redis.ttl(self.key), 120)

    @async_to_sync
    async def test_acquire_tokens_are_independent(self):
        limit = 3
        for _ in range(limit):
            await self.limiter.acquire(self.key, limit)
        self.assertTrue(await self.limiter.acquire(self.key_2, limit))
        self.assertEqual(await self.redis.zcard(self.key), limit)
        self.assertEqual(await self.redis.zcard(self.key_2), 1)

    # ------------------------------ release ------------------------------
    @async_to_sync
    async def test_release_decrements_counter(self):
        lease = await self.limiter.acquire(self.key, 42)
        release = await self.limiter.release(self.key, lease)
        self.assertTrue(release)
        self.assertEqual(await self.redis.zcard(self.key), 0)

    @async_to_sync
    async def test_release_does_not_go_negative(self):
        lease = await self.limiter.acquire(self.key, 42)
        await self.limiter.release(self.key, lease)

        release = await self.limiter.release(self.key, lease)
        self.assertFalse(release)
        self.assertEqual(await self.redis.zcard(self.key), 0)

    @async_to_sync
    async def test_release_decrements_its_own_key_counter(self):
        lease = await self.limiter.acquire(self.key, 42)
        await self.limiter.acquire(self.key_2, 137)
        await self.limiter.release(self.key, lease)
        self.assertEqual(await self.redis.zcard(self.key), 0)
        self.assertEqual(await self.redis.zcard(self.key_2), 1)

    # ---------------------------- wrap_stream ----------------------------
    async def _collect_stream(self, stream, lease):
        return [chunk async for chunk in self.limiter.wrap_stream(stream, self.key, lease)]

    @async_to_sync
    async def test_wrap_stream_yields_all_chunks_and_releases(self):
        lease = await self.limiter.acquire(self.key, 42)
        self.assertEqual(await self.redis.zcard(self.key), 1)

        async def stream():
            yield b"chunk-1"
            yield b"chunk-2"

        chunks = await self._collect_stream(stream(), lease)
        self.assertEqual(chunks, [b"chunk-1", b"chunk-2"])
        # The slot is released once the stream is fully consumed.
        self.assertEqual(await self.redis.zcard(self.key), 0)

    @async_to_sync
    async def test_wrap_stream_releases_if_stream_raises(self):
        lease = await self.limiter.acquire(self.key, 42)

        async def stream():
            yield b"chunk-1"
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            await self._collect_stream(stream(), lease)

        self.assertEqual(await self.redis.zcard(self.key), 0)

    # ------------------------- _refresh_ttl_loop -------------------------
    @async_to_sync
    async def test_refresh_ttl_loop_extends_ttl(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)
        lease = await limiter.acquire(self.key, 1)
        await self.redis.expire(self.key, 1)

        stop_event = asyncio.Event()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, lease, stop_event))
        try:
            await asyncio.sleep(0.05)
            self.assertGreater(await self.redis.ttl(self.key), 1)  # refreshed
        finally:
            stop_event.set()
            await task

    @async_to_sync
    async def test_refresh_ttl_loop_stops_when_event_set(self):
        limiter = RedisConcurrencyLimiter(self.redis, refresh_interval=0.01)
        lease = await limiter.acquire(self.key, 1)
        await self.redis.expire(self.key, 2)

        stop_event = asyncio.Event()
        stop_event.set()
        task = asyncio.create_task(limiter._refresh_ttl_loop(self.key, lease, stop_event))
        await task

        # Stopping the heartbeat also releases its lease.
        self.assertEqual(await self.redis.zcard(self.key), 0)
