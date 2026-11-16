"""Run Django migrations under a PostgreSQL session advisory lock."""

import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import django
from django.core.management import call_command
from django.db import OperationalError, connections

# Stable across releases and installations targeting the same database.
LOCK_KEY = 4707761683377112148
WAIT_SECONDS = 600
RETRY_SECONDS = 2


@contextmanager
def wait_deadline(message):
    # The init runs on Linux in the main thread. Interrupt even a stalled
    # synchronous libpq query, not only the sleeps between attempts.
    def timed_out(signum, frame):
        raise RuntimeError(message)

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, WAIT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def wait_for_database(connection):
    message = "Timed out waiting for the migration database"
    with wait_deadline(message):
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            try:
                connection.ensure_connection()
                return
            except OperationalError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(message) from error
                print(f"Waiting for migration database: {error}", flush=True)
                time.sleep(min(RETRY_SECONDS, max(0, deadline - time.monotonic())))


def migrate(connection):
    if connection.vendor != "postgresql":
        raise RuntimeError("The Aqueduct Helm chart requires PostgreSQL; SQLite is not supported")

    message = "Timed out waiting for the PostgreSQL migration lock"
    with wait_deadline(message):
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(%s)", [LOCK_KEY])
                acquired = cursor.fetchone()[0]
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(message)
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
    # Configure the backend before app initialization can open a connection.
    connection = connections["default"]
    if connection.vendor != "postgresql":
        raise RuntimeError("The Aqueduct Helm chart requires PostgreSQL; SQLite is not supported")
    # Dedicated physical session; Linux TCP timers limit detection of half-open
    # connections, including during migrations (without a statement timeout).
    options = connection.settings_dict.setdefault("OPTIONS", {})
    options.pop("pool", None)
    options.update(connect_timeout=5, keepalives=1, keepalives_idle=10,
                   keepalives_interval=5, keepalives_count=3, tcp_user_timeout=30000)
    try:
        django.setup()
        from gateway.config import get_router_config

        get_router_config()
        wait_for_database(connection)
        migrate(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
