from typing import ClassVar
from unittest.mock import patch

from django.db import OperationalError
from django.test import SimpleTestCase, override_settings


class ReadinessCheckTests(SimpleTestCase):
    """`/health/ready` reports whether the database answers within a short wait."""

    # The check runs a real query and then closes the connection, which a
    # `TestCase` transaction would not survive.
    databases: ClassVar[set[str]] = {"default"}

    def test_ok_when_database_answers(self):
        for path in ("/health/ready", "/health/ready/"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"ok")
                self.assertEqual(response["Content-Type"], "text/plain")

    def test_unavailable_when_query_fails(self):
        with patch("management.middleware.connection") as connection:
            connection.pool = None
            connection.cursor.side_effect = OperationalError("database is locked")
            response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.content, b"db unavailable")
        connection.close.assert_called_once()

    @override_settings(HEALTH_DB_TIMEOUT=2)
    def test_pool_connection_waits_only_for_health_timeout(self):
        with patch("management.middleware.connection") as connection:
            response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"ok")
        pool = connection.pool
        # The first probe can arrive before any request has opened the pool
        pool.open.assert_called_once()
        pool.connection.assert_called_once_with(timeout=2)
        pooled = pool.connection.return_value.__enter__.return_value
        pooled.execute.assert_called_once_with("SELECT 1")

    def test_unavailable_when_pool_has_no_connection(self):
        with patch("management.middleware.connection") as connection:
            # Like `psycopg_pool.PoolTimeout`
            connection.pool.connection.side_effect = OperationalError(
                "couldn't get a connection after 2.00 sec"
            )
            response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.content, b"db unavailable")
