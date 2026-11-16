"""Unit checks for migration startup; PostgreSQL behavior is tested separately."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/migrate.py"
spec = importlib.util.spec_from_file_location("migration_entrypoint", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class MigrationEntrypointTests(unittest.TestCase):
    def test_invalid_configuration_prevents_database_access(self):
        config = Mock()
        config.get_router_config.side_effect = RuntimeError("invalid config")
        connection = Mock(vendor="postgresql", settings_dict={"OPTIONS": {}})
        connections = MagicMock()
        connections.__getitem__.return_value = connection
        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
              patch.object(migration.django, "setup"),
              patch.object(migration, "connections", connections),
              self.assertRaisesRegex(RuntimeError, "invalid config")):
            migration.main()
        connection.ensure_connection.assert_not_called()
        connection.close.assert_called_once()

    def test_database_wait_is_bounded(self):
        connection = Mock()
        connection.ensure_connection.side_effect = migration.OperationalError("offline")
        with (patch.object(migration.time, "monotonic", side_effect=[0, 601]),
              self.assertRaisesRegex(RuntimeError, "Timed out waiting")):
            migration.wait_for_database(connection)

    def test_sqlite_is_rejected_without_migration_or_database_access(self):
        connection = Mock(vendor="sqlite")
        config = Mock()
        connections = MagicMock()
        connections.__getitem__.return_value = connection
        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
              patch.object(migration.django, "setup"),
              patch.object(migration, "connections", connections),
              patch.object(migration, "call_command") as command,
              self.assertRaisesRegex(RuntimeError, "requires PostgreSQL")):
            migration.main()
        connection.ensure_connection.assert_not_called()
        command.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "requires PostgreSQL"):
            migration.migrate(connection)

    def test_main_enforces_unpooled_connection_and_tcp_timeouts(self):
        connection = Mock(vendor="postgresql", settings_dict={"OPTIONS": {"pool": {"max_size": 10}}})
        connections = MagicMock()
        connections.__getitem__.return_value = connection
        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": Mock()}),
              patch.object(migration.django, "setup", side_effect=lambda: self.assertNotIn(
                  "pool", connection.settings_dict["OPTIONS"])),
              patch.object(migration, "connections", connections),
              patch.object(migration, "wait_for_database"),
              patch.object(migration, "migrate")):
            migration.main()
        options = connection.settings_dict["OPTIONS"]
        self.assertNotIn("pool", options)
        self.assertEqual(options["connect_timeout"], 5)
        self.assertEqual(options["tcp_user_timeout"], 30000)
        self.assertEqual((options["keepalives_idle"], options["keepalives_interval"],
                          options["keepalives_count"]), (10, 5, 3))
        connection.close.assert_called_once()

    def test_stalled_lock_query_is_interrupted(self):
        connection = MagicMock(vendor="postgresql")
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = lambda *args: migration.time.sleep(5)
        with (patch.object(migration, "WAIT_SECONDS", 0.05),
              patch.object(migration, "call_command") as command,
              self.assertRaisesRegex(RuntimeError, "Timed out waiting.*lock")):
            migration.migrate(connection)
        command.assert_not_called()

    def test_session_loss_never_reconnects(self):
        connection = MagicMock(vendor="postgresql")
        connection.connection.closed = False
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value = [True]
        original = connection.ensure_connection

        def lost(*args, **kwargs):
            connection.connection = None
            connection.ensure_connection()

        with (patch.object(migration, "call_command", side_effect=lost),
              self.assertRaisesRegex(RuntimeError, "refusing to reconnect")):
            migration.migrate(connection)
        original.assert_not_called()
        connection.close.assert_called_once()

    def test_failure_closes_locked_connection(self):
        connection = MagicMock(vendor="postgresql")
        connection.connection.closed = False
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value = [True]
        with (patch.object(migration, "call_command", side_effect=ValueError("failed")),
              self.assertRaisesRegex(ValueError, "failed")):
            migration.migrate(connection)
        connection.close.assert_called_once()

    def test_lock_wait_is_bounded(self):
        connection = MagicMock(vendor="postgresql")
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value = [False]
        with (patch.object(migration.time, "monotonic", side_effect=[0, 601]),
              self.assertRaisesRegex(RuntimeError, "migration lock")):
            migration.migrate(connection)


if __name__ == "__main__":
    unittest.main()
