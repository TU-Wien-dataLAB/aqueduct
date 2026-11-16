"""Run Django migrations under a PostgreSQL session advisory lock."""

import os
import sys
import time
from pathlib import Path

import django
from django.core.management import call_command
from django.db import OperationalError, connections

# Stable across releases and installations targeting the same database.
LOCK_KEY = 4707761683377112148
WAIT_SECONDS = 600
RETRY_SECONDS = 2


def wait_for_database(connection):
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        try:
            connection.ensure_connection()
            return
        except OperationalError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the migration database") from error
            print(f"Waiting for migration database: {error}", flush=True)
            time.sleep(min(RETRY_SECONDS, max(0, deadline - time.monotonic())))


def migrate(connection):
    if connection.vendor != "postgresql":
        call_command("migrate", interactive=False, database="default")
        return

    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", [LOCK_KEY])
            acquired = cursor.fetchone()[0]
        if acquired:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for the PostgreSQL migration lock")
        print("Waiting for PostgreSQL migration lock", flush=True)
        time.sleep(min(RETRY_SECONDS, max(0, deadline - time.monotonic())))

    session = connection.connection
    original_ensure = connection.ensure_connection
    original_connect = connection.connect

    def ensure_locked_session():
        if connection.connection is not session or session.closed:
            raise RuntimeError("Migration lock session was lost; refusing to reconnect")
        original_ensure()

    def refuse_connect():
        raise RuntimeError("Migration lock session was lost; refusing to reconnect")

    # Django normally reconnects transparently. Never permit that while migrating:
    # the advisory lock belongs to this exact physical PostgreSQL session.
    connection.ensure_connection = ensure_locked_session
    connection.connect = refuse_connect
    try:
        call_command("migrate", interactive=False, database="default")
        ensure_locked_session()
    finally:
        # Closing a non-pooled connection releases the session lock, including on
        # exceptions or a broken session. Never return a locked session to a pool.
        connection.ensure_connection = original_ensure
        connection.connect = original_connect
        try:
            connection.close()
        finally:
            # Also close the original session if anything replaced the wrapper's
            # connection reference during the migration.
            session.close()


def main():
    sys.path.insert(0, str(Path.cwd()))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aqueduct.settings")
    django.setup()

    from gateway.config import get_router_config

    get_router_config()
    connection = connections["default"]
    if connection.vendor == "postgresql":
        # Use a dedicated physical session, not a pooled connection. Bound each
        # connection attempt independently of the overall retry deadline.
        connection.settings_dict["OPTIONS"].pop("pool", None)
        connection.settings_dict["OPTIONS"]["connect_timeout"] = 5
    try:
        wait_for_database(connection)
        migrate(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
