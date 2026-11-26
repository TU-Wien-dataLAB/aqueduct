"""Prometheus metrics for `/metrics`, read from the database, Redis and the pool on each scrape.

Nothing is counted in-process. Gateway-wide metrics read shared state, so every replica
reports the same values for them.
"""

import asyncio
import logging
import time
import tomllib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from functools import cache
from typing import Any

from asgiref.sync import async_to_sync
from django.conf import settings
from django.db import connection
from django.db.models import Count
from django.utils import timezone
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector

from gateway.config import get_router_config
from gateway.decorators import limits
from management.models import Request

log = logging.getLogger("aqueduct")

# Seconds a scrape gives Redis before reporting it down.
REDIS_TIMEOUT = 1.0

PER_POD = "Per pod."
GATEWAY_WIDE = "Gateway-wide: every pod reports the same value, so aggregate with max, not sum."


def render() -> bytes:
    """The Aqueduct, process and platform metrics in the Prometheus text format."""
    return generate_latest(REGISTRY)


@cache
def _version() -> str:
    try:
        with (settings.BASE_DIR.parent / "pyproject.toml").open("rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    except (OSError, tomllib.TOMLDecodeError, KeyError):
        return "unknown"


def _models() -> list[str]:
    """The router config's models, by the name their concurrency leases use."""
    try:
        names = (entry["model_name"] for entry in get_router_config().get("model_list", []))
        return list(dict.fromkeys(limits.model_concurrency_name(name) for name in names))
    except Exception as e:
        log.warning("Metrics: no models, router config unavailable: %r", e)
        return []


@contextmanager
def _bounded_cursor() -> Iterator[Any]:
    """A cursor on a single connection; a pool connection is waited for HEALTH_DB_TIMEOUT."""
    pool = getattr(connection, "pool", None)
    if pool is None:
        with connection.cursor() as cursor:
            yield cursor
        return
    # As in the readiness check: the pool opens lazily, and this may be the first access.
    pool.open()
    with (
        pool.connection(timeout=settings.HEALTH_DB_TIMEOUT) as conn,
        conn.transaction(),
        conn.cursor() as cursor,
    ):
        # Ends with the transaction, so the connection returns to the pool unchanged.
        cursor.execute("SET LOCAL statement_timeout = '1s'")
        yield cursor


def _read_database() -> tuple[bool, list[tuple[str, int | None, int]] | None]:
    """Whether the database answers, and the last minute's request counts by model and status."""
    recent = (
        Request.objects.filter(timestamp__gte=timezone.now() - timedelta(seconds=60))
        .values_list("model", "status_code")
        .annotate(Count("id"))
        .order_by()
    )
    up = False
    try:
        with _bounded_cursor() as cursor:
            cursor.execute("SELECT 1")
            up = True
            cursor.execute(*recent.query.sql_with_params())
            return up, cursor.fetchall()
    except Exception as e:
        log.warning("Metrics: database %s: %r", "query failed" if up else "unavailable", e)
        return up, None


async def _read_redis(keys: list[str]) -> tuple[bool, list[int] | None]:
    """Whether Redis answers, and the number of unexpired leases under each key."""
    limiter = limits.get_limiter()
    up = False
    try:
        async with asyncio.timeout(REDIS_TIMEOUT):
            await limiter.redis.ping()
            up = True
            return up, await limiter.count_unexpired(keys)
    except Exception as e:
        log.warning("Metrics: Redis %s: %r", "query failed" if up else "unavailable", e)
        return up, None


def _pool_metrics(stats: dict[str, int]) -> list[Metric]:
    def gauge(name: str, key: str, text: str) -> Metric:
        return GaugeMetricFamily(name, f"{text} {PER_POD}", value=stats.get(key, 0))

    def counter(name: str, key: str, text: str, scale: float = 1) -> Metric:
        return CounterMetricFamily(name, f"{text} {PER_POD}", value=stats.get(key, 0) * scale)

    return [
        gauge("aqueduct_db_pool_size", "pool_size", "Connections open in the database pool."),
        gauge(
            "aqueduct_db_pool_available",
            "pool_available",
            "Idle connections in the database pool; in use is size minus available.",
        ),
        gauge("aqueduct_db_pool_max", "pool_max", "Maximum size of the database pool."),
        gauge(
            "aqueduct_db_pool_requests_waiting",
            "requests_waiting",
            "Requests waiting for a database pool connection.",
        ),
        counter(
            "aqueduct_db_pool_requests_total",
            "requests_num",
            "Connections requested from the database pool.",
        ),
        counter(
            "aqueduct_db_pool_requests_errors_total",
            "requests_errors",
            "Database pool connection requests that failed, for example by timing out.",
        ),
        counter(
            "aqueduct_db_pool_connections_lost_total",
            "connections_lost",
            "Database pool connections found broken.",
        ),
        counter(
            "aqueduct_db_pool_wait_seconds_total",
            "requests_wait_ms",
            "Time requests waited for a database pool connection.",
            scale=0.001,
        ),
    ]


def _requests_last_minute(rows: list[tuple[str, int | None, int]], models: list[str]) -> Metric:
    counts: Counter[tuple[str, str]] = Counter()
    # Without models, the router config is unavailable to resolve names with.
    for requested, status_code, count in rows if models else ():
        # Requests store the name the client sent, which may be an alias.
        model = limits.model_concurrency_name(requested)
        if model in models:
            counts[model, "pending" if status_code is None else str(status_code)] += count
    family = GaugeMetricFamily(
        "aqueduct_requests_last_minute",
        "Requests for the model in the last 60 seconds by status code, 'pending' while "
        f"unfinished. {GATEWAY_WIDE}",
        labels=["model", "status"],
    )
    for labels, count in counts.items():
        family.add_metric(labels, count)
    return family


def _collect() -> Iterator[Metric]:
    build_info = GaugeMetricFamily(
        "aqueduct_build_info", f"Running Aqueduct version, always 1. {PER_POD}", labels=["version"]
    )
    build_info.add_metric([_version()], 1)
    yield build_info

    pool = getattr(connection, "pool", None)
    if pool is not None:
        # Read before this scrape takes a connection, so it does not count as in use.
        yield from _pool_metrics(pool.get_stats())

    models = _models()
    db_up, rows = _read_database()
    yield GaugeMetricFamily(
        "aqueduct_db_up", f"Whether the database answers a query. {PER_POD}", value=int(db_up)
    )
    if rows is not None:
        yield _requests_last_minute(rows, models)

    redis_up, inflight = async_to_sync(_read_redis)(
        [limits.model_concurrency_key(model) for model in models]
    )
    yield GaugeMetricFamily(
        "aqueduct_redis_up",
        f"Whether the Redis used for concurrency limiting answers. {PER_POD}",
        value=int(redis_up),
    )
    if inflight is not None:
        family = GaugeMetricFamily(
            "aqueduct_inflight_requests",
            f"Requests holding a concurrency slot of the model. {GATEWAY_WIDE}",
            labels=["model"],
        )
        for model, count in zip(models, inflight, strict=True):
            family.add_metric([model], count)
        yield family

    limit = GaugeMetricFamily(
        "aqueduct_model_concurrency_limit",
        f"Concurrent requests allowed for the model. {GATEWAY_WIDE}",
        labels=["model"],
    )
    for model in models:
        limit.add_metric([model], limits.model_concurrency_limit(model))
    yield limit


class AqueductCollector(Collector):
    def collect(self) -> list[Metric]:
        start = time.perf_counter()
        metrics = list(_collect())
        duration = GaugeMetricFamily(
            "aqueduct_scrape_duration_seconds",
            f"Time spent collecting the Aqueduct metrics. {PER_POD}",
            value=time.perf_counter() - start,
        )
        return [*metrics, duration]

    def describe(self) -> list[Metric]:
        # Without this, registering would collect, and so query the database and Redis.
        return []


REGISTRY.register(AqueductCollector())
