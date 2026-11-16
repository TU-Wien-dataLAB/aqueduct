"""Exercise startup gate behavior against Kubernetes API responses."""

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch
from urllib.error import HTTPError

SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/wait_for_migrations.py"
spec = importlib.util.spec_from_file_location("migration_gate", SCRIPT)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

MIGRATION_SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/migrate.py"
spec = importlib.util.spec_from_file_location("migration_entrypoint", MIGRATION_SCRIPT)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def response(*conditions):
    return io.StringIO(json.dumps({"status": {"conditions": list(conditions)}}))


class MigrationStartupTests(unittest.TestCase):
    def setUp(self):
        for mock in (
            patch.dict("os.environ", {
                "KUBERNETES_SERVICE_HOST": "kubernetes.default.svc",
                "KUBERNETES_SERVICE_PORT_HTTPS": "443",
            }),
            patch.object(Path, "read_text", return_value="test"),
            patch.object(gate.ssl, "create_default_context"),
            patch.object(gate.time, "sleep"),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_only_a_completed_job_releases_startup(self):
        with patch.object(gate, "urlopen", return_value=response(
            {"type": "Complete", "status": "True"},
        )):
            gate.wait_for_migrations("test-migrate-1")

    def test_missing_running_and_false_complete_conditions_do_not_release_startup(self):
        missing = HTTPError("https://kubernetes.default.svc", 404, "Not Found", {}, None)
        for pending in (missing, response(), response({"type": "Complete", "status": "False"})):
            with self.subTest(pending=pending), patch.object(gate, "urlopen", side_effect=[
                pending, response({"type": "Failed", "status": "True"}),
            ]):
                with self.assertRaisesRegex(RuntimeError, "Migration Job test-migrate-1 failed"):
                    gate.wait_for_migrations("test-migrate-1")

    def test_failed_job_prevents_startup(self):
        with patch.object(gate, "urlopen", return_value=response(
            {"type": "Failed", "status": "True"},
        )):
            with self.assertRaisesRegex(RuntimeError, "Migration Job test-migrate-1 failed"):
                gate.wait_for_migrations("test-migrate-1")

    def test_unreadable_job_prevents_startup(self):
        forbidden = HTTPError("https://kubernetes.default.svc", 403, "Forbidden", {}, None)
        with patch.object(gate, "urlopen", side_effect=forbidden):
            with self.assertRaises(HTTPError):
                gate.wait_for_migrations("test-migrate-1")


class MigrationEntrypointTests(unittest.TestCase):
    def test_validation_and_database_wait_precede_migration(self):
        events = Mock()
        config = Mock(get_router_config=events.validate)
        connection = Mock(ensure_connection=events.connect)
        connections = MagicMock(close_all=events.close)
        connections.__getitem__ = Mock(return_value=connection)
        events.connect.side_effect = [migration.OperationalError("not ready"), None]
        with (
            patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
            patch.object(sys, "path", sys.path.copy()),
            patch.object(migration.django, "setup", events.setup),
            patch.object(migration, "connections", connections),
            patch.object(migration.time, "sleep", events.sleep),
            patch.object(migration.os, "execv", events.execv),
        ):
            migration.main()
        self.assertEqual(events.mock_calls, [
            call.setup(), call.validate(), call.connect(), call.sleep(2), call.connect(),
            call.close(), call.execv(sys.executable, [sys.executable, "manage.py", "migrate", "--noinput"]),
        ])

    def test_invalid_configuration_prevents_database_access_and_migration(self):
        config = Mock()
        config.get_router_config.side_effect = RuntimeError("invalid config")
        with (
            patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
            patch.object(sys, "path", sys.path.copy()),
            patch.object(migration.django, "setup"),
            patch.object(migration, "connections") as connections,
            patch.object(migration.os, "execv") as execute,
            self.assertRaisesRegex(RuntimeError, "invalid config"),
        ):
            migration.main()
        connections.__getitem__.assert_not_called()
        execute.assert_not_called()

    def test_migration_exec_failure_is_not_retried(self):
        config = Mock()
        connections = MagicMock()
        connections.__getitem__ = Mock(return_value=Mock())
        with (
            patch.dict(sys.modules, {"gateway": Mock(), "gateway.config": config}),
            patch.object(sys, "path", sys.path.copy()),
            patch.object(migration.django, "setup"),
            patch.object(migration, "connections", connections),
            patch.object(migration.os, "execv", side_effect=OSError("exec failed")) as execute,
            self.assertRaisesRegex(OSError, "exec failed"),
        ):
            migration.main()
        execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
