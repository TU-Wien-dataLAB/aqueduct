"""Allow a fresh release's database to bootstrap before running migrations once."""

import os
import sys
import time
from pathlib import Path

import django
from django.db import OperationalError, connections

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aqueduct.settings")
django.setup()

# Reuse application validation before schema changes or any new serving process.
from gateway.config import get_router_config

get_router_config()

while True:
    try:
        connections["default"].ensure_connection()
    except OperationalError as error:
        print(f"Waiting for PostgreSQL: {error}", flush=True)
        time.sleep(2)
    else:
        connections.close_all()
        break
