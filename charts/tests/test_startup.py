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
        with (patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
              patch.object(migration.django, "setup"),
              patch.object(migration, "connections") as connections,
              self.assertRaisesRegex(RuntimeError, "invalid config")):
            migration.main()
        connections.__getitem__.assert_not_called()

    def test_database_wait_is_bounded(self):
        connection = Mock()
        connection.ensure_connection.side_effect = migration.OperationalError("offline")
        with (patch.object(migration.time, "monotonic", side_effect=[0, 601]),
              self.assertRaisesRegex(RuntimeError, "Timed out waiting")):
            migration.wait_for_database(connection)

    def test_sqlite_runs_normal_migration(self):
        connection = Mock(vendor="sqlite")
        with patch.object(migration, "call_command") as command:
            migration.migrate(connection)
        command.assert_called_once_with("migrate", interactive=False, database="default")

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
