import os
import asyncio
import httpx
import time

# Config from environment
TOKEN = os.getenv("BENCH_TOKEN", "")
BASE_URL = os.getenv("BENCH_URL", "http://localhost:8000")
ENDPOINT = os.getenv("BENCH_ENDPOINT", "vllm")
NUM_RUNS = int(os.getenv("BENCH_RUNS", 5))
MODE = os.getenv("BENCH_MODE", "sequential").lower()

URL = f"{BASE_URL.rstrip('/')}/{ENDPOINT.strip('/')}/models"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


async def fetch(client: httpx.AsyncClient, i: int) -> float:
    start = time.perf_counter()
    try:
        response = await client.get(URL, headers=HEADERS, timeout=10.0)
        response.raise_for_status()
    except Exception as e:
        print(f"Run {i + 1}: Error - {e}")
        return None
    duration = time.perf_counter() - start
    print(f"Run {i + 1}: {duration:.3f}s")
    return duration


async def run_sequential():
    async with httpx.AsyncClient() as client:
        times = []
        for i in range(NUM_RUNS):
            t = await fetch(client, i)
            if t is not None:
                times.append(t)
        return times


async def run_parallel():
    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, i) for i in range(NUM_RUNS)]
        results = await asyncio.gather(*tasks)
        return [t for t in results if t is not None]


async def main():
    print(f"Benchmarking {NUM_RUNS} {MODE} requests to: {URL}")
    times = await run_sequential() if MODE == "sequential" else await run_parallel()

    if times:
        avg = sum(times) / len(times)
        print(f"\nAverage response time: {avg:.3f}s over {len(times)} runs")
    else:
        print("No valid responses received.")


if __name__ == "__main__":
    asyncio.run(main())
