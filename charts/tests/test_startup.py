"""Unit checks for migration startup; PostgreSQL behavior is tested separately."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/migrate.py"
spec = importlib.util.spec_from_file_location("migration_entrypoint", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class MigrationEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock(vendor="postgresql", settings_dict={"OPTIONS": {"pool": {"max_size": 10}}})
        self.connections = MagicMock()
        self.connections.__getitem__.return_value = self.connection
        self.config = Mock()

    def test_migration_locks_then_resets_timeout_before_django(self):
        events = Mock()
        cursor = self.connection.cursor.return_value.__enter__.return_value
        events.attach_mock(cursor, "cursor")
        with patch.object(migration, "call_command", events.command):
            migration.migrate(self.connection)
        self.assertEqual(events.mock_calls, [
            call.cursor.execute("SELECT set_config('lock_timeout', %s, false)", ["10min"]),
            call.cursor.execute("SELECT pg_advisory_lock(%s)", [migration.LOCK_KEY]),
            call.cursor.execute("RESET lock_timeout"),
            call.command("migrate", interactive=False, database="default"),
        ])

    def test_invalid_configuration_prevents_database_access(self):
        self.config.get_router_config.side_effect = RuntimeError("invalid config")
        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": self.config}),
              patch.object(migration.django, "setup"),
              patch.object(migration, "connections", self.connections),
              patch.object(migration, "migrate") as migrate,
              self.assertRaisesRegex(RuntimeError, "invalid config")):
            migration.main()
        migrate.assert_not_called()
        self.connection.close.assert_called_once()

    def test_sqlite_is_rejected_before_migration(self):
        self.connection.vendor = "sqlite"
        with (patch.object(migration, "connections", self.connections),
              patch.object(migration, "call_command") as command,
              self.assertRaisesRegex(RuntimeError, "requires PostgreSQL")):
            migration.main()
        self.connection.cursor.assert_not_called()
        command.assert_not_called()

    def test_unpooled_connection_is_configured_before_setup_and_closed_on_failure(self):
        def setup():
            self.assertNotIn("pool", self.connection.settings_dict["OPTIONS"])
            self.assertEqual(self.connection.settings_dict["OPTIONS"]["connect_timeout"], 5)

        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": self.config}),
              patch.object(migration.django, "setup", side_effect=setup),
              patch.object(migration, "connections", self.connections),
              patch.object(migration, "migrate", side_effect=RuntimeError("failed")),
              self.assertRaisesRegex(RuntimeError, "failed")):
            migration.main()
        self.config.get_router_config.assert_called_once()
        self.connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
