"""Run Django migrations under a PostgreSQL session advisory lock."""

import os
import sys
from pathlib import Path

import django
from django.core.management import call_command
from django.db import connections

# Stable across releases and installations targeting the same database.
LOCK_KEY = 4707761683377112148
LOCK_TIMEOUT = "10min"


def migrate(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT set_config('lock_timeout', %s, false)", [LOCK_TIMEOUT])
        cursor.execute("SELECT pg_advisory_lock(%s)", [LOCK_KEY])
        cursor.execute("RESET lock_timeout")
    call_command("migrate", interactive=False, database="default")


def main():
    sys.path.insert(0, str(Path.cwd()))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aqueduct.settings")
    # Configure the backend before app initialization can open a connection.
    connection = connections["default"]
    if connection.vendor != "postgresql":
        raise RuntimeError("The Aqueduct Helm chart requires PostgreSQL; SQLite is not supported")
    options = connection.settings_dict.setdefault("OPTIONS", {})
    options.pop("pool", None)
    options["connect_timeout"] = 5
    try:
        django.setup()
        from gateway.config import get_router_config

        get_router_config()
        migrate(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
