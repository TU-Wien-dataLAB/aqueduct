import asyncio

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase, override_settings
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from gateway.decorators.limits import RedisConcurrencyLimiter


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


@override_settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS=120)
class RedisConcurrencyLimiterTests(SimpleTestCase):
    """Unit tests for RedisConcurrencyLimiter, using fakeredis.

    The Lua scripts (``acquire_script`` / ``release_script``) are executed
    by fakeredis, so no further mocks are needed.
    """

    def setUp(self) -> None:
        self.redis = FakeRedis(server=FakeServer(), decode_responses=True)
        self.limiter = RedisConcurrencyLimiter(self.redis)
        self.key = "token:concurrency:1:leases"
        self.key_2 = "token:concurrency:2:leases"
        self.addCleanup(async_to_sync(self.redis.aclose))

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
