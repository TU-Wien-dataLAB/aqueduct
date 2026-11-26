import tempfile
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import yaml
from asgiref.sync import async_to_sync
from django.db import OperationalError
from django.test import override_settings
from django.utils import timezone
from prometheus_client.parser import text_string_to_metric_families

from gateway.decorators.limits import model_concurrency_key
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request, Token

TOKEN = "metrics-token"
PATHS = ("/metrics", "/metrics/")

AQUEDUCT_FAMILIES = {
    "aqueduct_build_info",
    "aqueduct_db_up",
    "aqueduct_redis_up",
    "aqueduct_inflight_requests",
    "aqueduct_model_concurrency_limit",
    "aqueduct_requests_last_minute",
    "aqueduct_scrape_duration_seconds",
}
POOL_FAMILIES = {
    "aqueduct_db_pool_size",
    "aqueduct_db_pool_available",
    "aqueduct_db_pool_max",
    "aqueduct_db_pool_requests_waiting",
    "aqueduct_db_pool_requests",
    "aqueduct_db_pool_requests_errors",
    "aqueduct_db_pool_connections_lost",
    "aqueduct_db_pool_wait_seconds",
}


class Scrape:
    """A parsed `/metrics` response: families by name, and samples by name and labels."""

    def __init__(self, response):
        self.families = {
            family.name: family
            for family in text_string_to_metric_families(response.content.decode())
        }

    def samples(self, name: str) -> dict[tuple[str, ...], float]:
        """Sample values of a metric, keyed by their label values."""
        return {
            tuple(sample.labels.values()): sample.value
            for family in self.families.values()
            for sample in family.samples
            if sample.name == name
        }

    def value(self, name: str) -> float:
        return self.samples(name)[()]


@override_settings(AQUEDUCT_METRICS_TOKEN=TOKEN)
class MetricsEndpointTests(GatewayIntegrationTestCase):
    """`/metrics` reports gateway health and load to Prometheus holding the metrics token."""

    def scrape(self, path="/metrics", token=TOKEN):
        headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
        return self.client.get(path, headers=headers)

    def scrape_ok(self, path="/metrics") -> Scrape:
        response = self.scrape(path)
        self.assertEqual(response.status_code, 200)
        return Scrape(response)

    @override_settings(AQUEDUCT_METRICS_TOKEN="")
    def test_without_a_configured_token_answers_like_a_path_without_a_route(self):
        unknown = self.client.get("/no-such-path")
        self.assertEqual(unknown.status_code, 404)
        for path in PATHS:
            for token in (None, "", TOKEN):
                with self.subTest(path=path, token=token):
                    response = self.scrape(path, token)
                    self.assertEqual(response.status_code, 404)
                    self.assertEqual(response.content, unknown.content)
                    self.assertEqual(dict(response.headers), dict(unknown.headers))

    def test_unauthorized_without_the_token(self):
        for path in PATHS:
            for token in (None, "", "wrong", TOKEN[:-1], f"{TOKEN}x", f"{TOKEN} x"):
                with self.subTest(path=path, token=token):
                    response = self.scrape(path, token)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.content, b"unauthorized")
                    self.assertEqual(response["WWW-Authenticate"], "Bearer")
                    self.assertNotIn(b"aqueduct_", response.content)

    def test_reports_every_family_with_the_token(self):
        for path in PATHS:
            with self.subTest(path=path):
                response = self.scrape(path)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response["Content-Type"].startswith("text/plain; version="))
                scrape = Scrape(response)
                self.assertLessEqual(AQUEDUCT_FAMILIES, scrape.families.keys())
                # The library's own collectors are kept
                self.assertIn("python_info", scrape.families)
                # The test database has no connection pool
                self.assertFalse(POOL_FAMILIES & scrape.families.keys())

                self.assertEqual(scrape.value("aqueduct_db_up"), 1)
                self.assertEqual(scrape.value("aqueduct_redis_up"), 1)
                ((version,),) = scrape.samples("aqueduct_build_info")
                self.assertRegex(version, r"^\d+\.\d+\.\d+")
                self.assertEqual(scrape.samples("aqueduct_build_info"), {(version,): 1})
                self.assertGreater(scrape.value("aqueduct_scrape_duration_seconds"), 0)

    def test_scrapes_do_not_create_sessions_or_request_records(self):
        response = self.scrape()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.cookies)
        self.assertFalse(Request.objects.exists())

    def test_inflight_counts_unexpired_leases_under_the_accounting_name(self):
        now = time.time()
        # Requests for an alias hold leases under the model it points to.
        key = model_concurrency_key("main")
        leases = {"stream": now + 600, "call": now + 60, "crashed": now - 1}

        async def seed():
            await self.redis.zadd(key, leases)

        async_to_sync(seed)()

        inflight = self.scrape_ok().samples("aqueduct_inflight_requests")

        self.assertEqual(inflight[("gpt-4.1-nano",)], 2)
        self.assertEqual(inflight[("gpt-6-luna",)], 0)
        self.assertNotIn(("main",), inflight)

    def test_requests_last_minute_by_model_and_status(self):
        token = Token.objects.first()
        now = timezone.now()
        seeded = [
            ("gpt-4.1-nano", 200, now),
            ("gpt-4.1-nano", 200, now - timedelta(seconds=30)),
            ("main", 200, now),  # alias of gpt-4.1-nano
            ("gpt-4.1-nano", 499, now),
            ("gpt-4.1-nano", None, now),
            ("text-embedding-ada-002", 429, now),
            ("gpt-4.1-nano", 200, now - timedelta(seconds=61)),
            ("", 200, now),  # an endpoint without a model, like /files
            ("not-configured", 404, now),
        ]
        Request.objects.bulk_create(
            Request(token=token, model=model, status_code=status, timestamp=timestamp)
            for model, status, timestamp in seeded
        )

        requests = self.scrape_ok().samples("aqueduct_requests_last_minute")

        self.assertEqual(
            requests,
            {
                ("gpt-4.1-nano", "200"): 3,
                ("gpt-4.1-nano", "499"): 1,
                ("gpt-4.1-nano", "pending"): 1,
                ("text-embedding-ada-002", "429"): 1,
            },
        )

    def test_concurrency_limit_from_router_config_then_settings_default(self):
        config = {
            "model_list": [
                {"model_name": "limited", "model_info": {"max_parallel_requests": 3}},
                {"model_name": "limited", "model_info": {"id": "limited-2"}},
                {"model_name": "default", "model_info": {"aliases": ["alias"]}},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router_config.yaml"
            path.write_text(yaml.safe_dump(config))
            with override_settings(
                LITELLM_ROUTER_CONFIG_FILE_PATH=path,
                AQUEDUCT_DEFAULT_MODEL_CONCURRENT_REQUESTS_LIMIT=7,
            ):
                scrape = self.scrape_ok()

        self.assertEqual(
            scrape.samples("aqueduct_model_concurrency_limit"), {("limited",): 3, ("default",): 7}
        )
        self.assertEqual(
            scrape.samples("aqueduct_inflight_requests"), {("limited",): 0, ("default",): 0}
        )

    def test_redis_down_leaves_out_inflight_requests(self):
        self.redis.connection_pool.connection_kwargs["server"].connected = False

        scrape = self.scrape_ok()

        self.assertEqual(scrape.value("aqueduct_redis_up"), 0)
        self.assertNotIn("aqueduct_inflight_requests", scrape.families)
        self.assertEqual(scrape.value("aqueduct_db_up"), 1)
        self.assertIn("aqueduct_model_concurrency_limit", scrape.families)

    def test_database_down_leaves_out_request_counts(self):
        with patch("management.metrics.connection") as connection:
            connection.pool = None
            connection.cursor.side_effect = OperationalError("database is locked")
            scrape = self.scrape_ok()

        self.assertEqual(scrape.value("aqueduct_db_up"), 0)
        self.assertNotIn("aqueduct_requests_last_minute", scrape.families)
        self.assertEqual(scrape.value("aqueduct_redis_up"), 1)
        self.assertIn("aqueduct_inflight_requests", scrape.families)

    @override_settings(HEALTH_DB_TIMEOUT=2)
    def test_pool_statistics_and_a_single_bounded_pool_connection(self):
        stats = {
            "pool_min": 20,
            "pool_max": 50,
            "pool_size": 23,
            "pool_available": 4,
            "requests_waiting": 1,
            "requests_num": 1200,
            "requests_errors": 3,
            "requests_wait_ms": 2500,
            "usage_ms": 99999,
            # `connections_lost` is left out until the pool loses a connection
        }
        with patch("management.metrics.connection") as connection:
            connection.pool.get_stats.return_value = stats
            scrape = self.scrape_ok()

        pool = connection.pool
        pool.connection.assert_called_once_with(timeout=2)
        pooled = pool.connection.return_value.__enter__.return_value
        cursor = pooled.cursor.return_value.__enter__.return_value
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(statements[:2], ["SET LOCAL statement_timeout = '1s'", "SELECT 1"])
        # The connection is given back before the response
        pool.connection.return_value.__exit__.assert_called_once()

        self.assertEqual(scrape.value("aqueduct_db_up"), 1)
        self.assertEqual(
            {name: scrape.value(name) for name in self.pool_sample_names(scrape)},
            {
                "aqueduct_db_pool_size": 23,
                "aqueduct_db_pool_available": 4,
                "aqueduct_db_pool_max": 50,
                "aqueduct_db_pool_requests_waiting": 1,
                "aqueduct_db_pool_requests_total": 1200,
                "aqueduct_db_pool_requests_errors_total": 3,
                "aqueduct_db_pool_connections_lost_total": 0,
                "aqueduct_db_pool_wait_seconds_total": 2.5,
            },
        )

    @staticmethod
    def pool_sample_names(scrape: Scrape) -> set[str]:
        return {
            sample.name
            for family in scrape.families.values()
            for sample in family.samples
            if sample.name.startswith("aqueduct_db_pool_")
        }
