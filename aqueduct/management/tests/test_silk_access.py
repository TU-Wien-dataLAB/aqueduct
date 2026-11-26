import json
import os
import subprocess
import sys
from pathlib import Path

from django.test import SimpleTestCase

PROBE = """
import json
from unittest import mock

import django

django.setup()

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse
from django.test import RequestFactory
from django.urls import resolve

User = get_user_model()
users = {
    "staff": User(username="staff", is_staff=True, is_active=True),
    "inactive superuser": User(username="gone", is_staff=True, is_superuser=True, is_active=False),
    "superuser": User(username="root", is_staff=True, is_superuser=True, is_active=True),
}
view = resolve("/silk/cleardb/").func
outcome = {}
for name, user in users.items():
    request = RequestFactory().get("/silk/cleardb/")
    request.user = user
    with mock.patch("silk.views.clear_db.render", return_value=HttpResponse("ok")):
        try:
            outcome[name] = view(request).status_code
        except PermissionDenied:
            outcome[name] = 403
print(json.dumps(outcome))
"""


class SilkAccessTests(SimpleTestCase):
    """The settings leave Silk off under test, so a fresh interpreter loads them with it on."""

    def test_only_active_superusers_open_silk(self):
        project = Path(__file__).resolve().parents[2]
        env = {**os.environ, "DJANGO_SETTINGS_MODULE": "aqueduct.settings", "SILKY_ENABLED": "True"}
        run = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(run.stdout.splitlines()[-1]),
            {"staff": 403, "inactive superuser": 403, "superuser": 200},
        )
