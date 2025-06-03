import os
import asyncio
import httpx
import time

# Config from environment
TOKEN = os.getenv("BENCH_TOKEN", "")
BASE_URL = os.getenv("BENCH_URL", "http://localhost:8000")
NUM_RUNS = int(os.getenv("BENCH_RUNS", 5))
MODE = os.getenv("BENCH_MODE", "sequential").lower()

URL = f"{BASE_URL.rstrip('/')}/models"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

TARGET = os.getenv("BENCH_TARGET", "completions").lower()  # "models" or "completions"
MODEL = os.getenv("BENCH_MODEL", "Qwen-32B")
if TARGET == "completions":
    ENDPOINT = f"{BASE_URL.rstrip('/')}/completions"
    REQUEST_TYPE = "POST"
    COMPLETION_PAYLOAD = {
        "model": MODEL,
        "prompt": "Write a short poem about the ocean.",
        "max_tokens": 32,
    }
else:
    ENDPOINT = f"{BASE_URL.rstrip('/')}/models"
    REQUEST_TYPE = "GET"
    COMPLETION_PAYLOAD = None


async def fetch(client: httpx.AsyncClient, i: int) -> float:
    start = time.perf_counter()
    try:
        if REQUEST_TYPE == "GET":
            response = await client.get(ENDPOINT, headers=HEADERS, timeout=15.0)
        else:
            response = await client.post(ENDPOINT, headers=HEADERS, json=COMPLETION_PAYLOAD, timeout=15.0)
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
    start = time.perf_counter()
    times = await run_sequential() if MODE == "sequential" else await run_parallel()

    if times:
        avg = sum(times) / len(times)
        print(f"\nAverage response time: {avg:.3f}s over {len(times)} runs")
        print(f"Total run time: {time.perf_counter() - start:.3f}s")
    else:
        print("No valid responses received.")


if __name__ == "__main__":
    asyncio.run(main())
