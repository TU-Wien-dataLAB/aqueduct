"""Benchmark settings: the demo's settings on bench.sh's throwaway PostgreSQL database."""

# Standalone settings module loaded through PYTHONPATH, not a package.
# ruff: noqa: INP001

import os
import warnings

# Silk saves every request and EXPLAINs every SELECT, which would swamp the timings.
os.environ["SILKY_ENABLED"] = "False"
os.environ["DJANGO_DEBUG"] = "False"

from demo_settings import *  # noqa: F403

ALLOWED_HOSTS = ["localhost"]
# Production serves the collected, hashed files; the benchmark has none to collect.
warnings.filterwarnings("ignore", "No directory at")
STORAGES = {
    **STORAGES,  # noqa: F405
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "aqueduct",
        "USER": "aqueduct",
        "PASSWORD": "aqueduct",
        "HOST": "127.0.0.1",
        "PORT": os.environ["AQUEDUCT_BENCH_DB_PORT"],
    }
}
