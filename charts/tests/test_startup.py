"""Exercise startup gate behavior against Kubernetes API responses."""

import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/wait_for_migrations.py"
spec = importlib.util.spec_from_file_location("migration_gate", SCRIPT)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


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


if __name__ == "__main__":
    unittest.main()
