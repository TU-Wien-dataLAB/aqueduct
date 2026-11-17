"""Demo-only settings: use the application with isolated local state."""

# Standalone settings module loaded through PYTHONPATH, not a package.
# ruff: noqa: INP001

import os
from pathlib import Path

# Force SQLite before importing application settings, including when the caller
# normally uses PostgreSQL.
os.environ["DATABASE_ENGINE"] = "django.db.backends.sqlite3"

from aqueduct.settings import *  # noqa: F403

# Keep UI caches and queued tasks away from the caller's Redis/Celery services.
CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "cache+memory://"
CELERY_TASK_ALWAYS_EAGER = False

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(Path(os.environ["AQUEDUCT_DEMO_STATE_DIR"]) / "db.sqlite3"),
    }
}
