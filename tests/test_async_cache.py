"""Async single-flight and TTL cache tests. No network."""
import asyncio
import unittest

from sibyl.async_cache import AsyncSingleFlightTTL


class TestAsyncSingleFlightTTL(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_callers_share_one_factory(self):
        cache = AsyncSingleFlightTTL[str, object](30, 4)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0
        value = object()

        async def factory():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return value

        first = asyncio.create_task(cache.get_or_create("key", factory))
        await started.wait()
        second = asyncio.create_task(cache.get_or_create("key", factory))
        await asyncio.sleep(0)
        release.set()

        self.assertIs(await first, value)
        self.assertIs(await second, value)
        self.assertEqual(calls, 1)

    async def test_cached_value_expires(self):
        now = 10.0
        cache = AsyncSingleFlightTTL[str, int](
            5,
            4,
            clock=lambda: now,
        )
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            return calls

        self.assertEqual(await cache.get_or_create("key", factory), 1)
        self.assertEqual(await cache.get_or_create("key", factory), 1)
        now = 16.0
        self.assertEqual(await cache.get_or_create("key", factory), 2)

    async def test_rejected_value_is_not_cached(self):
        cache = AsyncSingleFlightTTL[str, int](30, 4)
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            return calls

        should_cache = lambda value: value > 1
        self.assertEqual(
            await cache.get_or_create("key", factory, should_cache),
            1,
        )
        self.assertEqual(
            await cache.get_or_create("key", factory, should_cache),
            2,
        )
        self.assertEqual(await cache.get_or_create("key", factory), 2)

    async def test_cancelled_waiter_does_not_cancel_shared_work(self):
        cache = AsyncSingleFlightTTL[str, int](30, 4)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return 42

        waiter = asyncio.create_task(cache.get_or_create("key", factory))
        await started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        release.set()

        self.assertEqual(await cache.get_or_create("key", factory), 42)
        self.assertEqual(calls, 1)

    async def test_factory_exception_does_not_poison_the_key(self):
        cache = AsyncSingleFlightTTL[str, int](30, 4)
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            return 42

        with self.assertRaisesRegex(RuntimeError, "temporary"):
            await cache.get_or_create("key", factory)

        self.assertEqual(await cache.get_or_create("key", factory), 42)
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
