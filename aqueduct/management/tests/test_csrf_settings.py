import os
from runpy import run_path
from unittest.mock import patch

from django.test import SimpleTestCase

from aqueduct import settings


class CsrfSettingsTests(SimpleTestCase):
    def test_cookie_domain_from_environment(self):
        with patch.dict(os.environ, CSRF_COOKIE_DOMAIN=".example.com"):
            configured = run_path(settings.__file__)
        self.assertEqual(configured.get("CSRF_COOKIE_DOMAIN"), ".example.com")

    def test_host_only_cookie_by_default(self):
        for env in ({}, {"CSRF_COOKIE_DOMAIN": ""}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertIsNone(run_path(settings.__file__).get("CSRF_COOKIE_DOMAIN"))
