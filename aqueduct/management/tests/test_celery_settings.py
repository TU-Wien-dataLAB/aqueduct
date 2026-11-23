import os
import subprocess
import sys
from pathlib import Path

from django.test import SimpleTestCase


class CeleryBrokerSettingsTests(SimpleTestCase):
    def test_effective_broker_uses_redis_url_or_explicit_override(self):
        repo_root = Path(__file__).resolve().parents[3]
        chart_settings = repo_root / "charts" / "aqueduct" / "files"
        redis_url = "redis://shared-valkey:6379/0"
        override_url = "redis://other-valkey:6379/1"
        for settings_module, extra_path in (
            ("aqueduct.settings", ""),
            ("settings", str(chart_settings)),
        ):
            for override in (None, override_url):
                with self.subTest(settings=settings_module, override=override):
                    env = os.environ.copy()
                    env.pop("CELERY_BROKER_URL", None)
                    env.update(DJANGO_SETTINGS_MODULE=settings_module, REDIS_URL=redis_url)
                    if override is not None:
                        env["CELERY_BROKER_URL"] = override
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "import sys; sys.path.insert(0, sys.argv[1]); "
                            "from aqueduct.celery import app; "
                            "print(app.connection_for_write().as_uri(include_password=True))",
                            extra_path,
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(result.stdout.strip(), override or redis_url)
