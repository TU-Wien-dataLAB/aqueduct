"""Offline Helm regression tests: python3 -m unittest discover -s charts/tests -v."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts/aqueduct"
OVERLAY = ROOT / "docs/examples/cnpg-bitnami-bootstrap.yaml"
HOST_ERROR = (
    "cnpg.bootstrapFromExternal.connection.host is required "
    "when bootstrapFromExternal is enabled"
)


class CnpgBootstrapTests(unittest.TestCase):
    def render(self, *args):
        return subprocess.run(
            [
                "helm", "template", ".", "--name-template", "aqueduct-staging",
                "--namespace", "aqueduct-staging", "--kube-version", "1.36.2",
                "--include-crds", "--show-only", "templates/cnpg-cluster.yaml",
                *args,
            ],
            cwd=CHART, capture_output=True, text=True, check=False,
        )

    def test_fresh_cluster_uses_initdb(self):
        result = self.render()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("    initdb:", result.stdout)
        self.assertNotIn("pg_basebackup:", result.stdout)
        self.assertNotIn("externalClusters:", result.stdout)

    def test_external_bootstrap_requires_explicit_host(self):
        # Minimal reproduction of staging's retained migration settings.
        result = self.render("--set", "cnpg.bootstrapFromExternal.enabled=true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(HOST_ERROR, result.stderr)

    def test_legacy_bootstrap_correction_preserves_source(self):
        result = self.render(
            "--set", "cnpg.bootstrapFromExternal.connection.existingSecret=source-credentials",
            "--set", "cnpg.bootstrapFromExternal.connection.secretKeys.password=source-password",
            "--values", str(OVERLAY),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('      source: "bitnami"', result.stdout)
        self.assertIn('    - name: "bitnami"', result.stdout)
        self.assertIn('        host: "postgresql"', result.stdout)
        self.assertIn('        port: "5432"', result.stdout)
        self.assertIn("        name: source-credentials", result.stdout)
        self.assertIn("        key: source-password", result.stdout)
        self.assertNotIn("kind: Secret", result.stdout)
        self.assertNotIn("    initdb:", result.stdout)

    def test_explicit_external_source_is_not_replaced(self):
        result = self.render(
            "--set", "cnpg.bootstrapFromExternal.enabled=true",
            "--set", "cnpg.bootstrapFromExternal.connection.host=source.example.org",
            "--set-string", "cnpg.bootstrapFromExternal.connection.port=5433",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('        host: "source.example.org"', result.stdout)
        self.assertIn('        port: "5433"', result.stdout)

    def test_legacy_overlay_still_rejects_empty_host(self):
        result = self.render(
            "--values", str(OVERLAY),
            "--set", "cnpg.bootstrapFromExternal.connection.host=",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(HOST_ERROR, result.stderr)


if __name__ == "__main__":
    unittest.main()
