"""Opt-in disposable PostgreSQL test: AQUEDUCT_TEST_POSTGRES_URL=... unittest discover.

The database is modified by Django's contenttypes migrations; never use production.
"""

import importlib.util
import io
import time
from contextlib import nullcontext
import multiprocessing
import os
import queue
import select
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import urlparse

SCRIPT = Path(__file__).resolve().parents[1] / "aqueduct/files/migrate.py"
URL = os.environ.get("AQUEDUCT_TEST_POSTGRES_URL")


def contender(url, events, release, behavior):
    import django
    from django.conf import settings
    from django.core.management import call_command
    from django.db import connections

    target = urlparse(url)
    host, port = target.hostname, target.port
    stop_proxy = threading.Event()
    blackhole = threading.Event()
    if behavior == "blackhole":
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        host, port = listener.getsockname()

        def proxy():
            client, _ = listener.accept()
            upstream = socket.create_connection((target.hostname, target.port))
            try:
                while not stop_proxy.is_set():
                    readable, _, _ = select.select([client, upstream], [], [], 0.1)
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        if not blackhole.is_set():
                            (upstream if source is client else client).sendall(data)
            finally:
                client.close()
                upstream.close()

        proxy_thread = threading.Thread(target=proxy, daemon=True)
        proxy_thread.start()
    apps = ["django.contrib.contenttypes"]
    temporary = tempfile.TemporaryDirectory()
    if behavior == "fail":
        root = Path(temporary.name)
        migrations = root / "failing_app" / "migrations"
        migrations.mkdir(parents=True)
        (root / "failing_app" / "__init__.py").touch()
        (migrations / "__init__.py").touch()
        (migrations / "0001_failure.py").write_text(
            "from django.db import migrations\n"
            "class Migration(migrations.Migration):\n"
            "    initial = True\n"
            "    dependencies = []\n"
            "    operations = [migrations.RunSQL('SELECT 1 / 0')]\n"
        )
        sys.path.insert(0, temporary.name)
        apps.append("failing_app")
    settings.configure(
        SECRET_KEY="test", INSTALLED_APPS=apps,
        DATABASES={"default": {"ENGINE": "django.db.backends.postgresql", "NAME": target.path[1:],
                               "USER": target.username, "PASSWORD": target.password,
                               "HOST": host, "PORT": port}},
        USE_TZ=True,
    )
    django.setup()
    spec = importlib.util.spec_from_file_location("migration_entrypoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connection = connections["default"]
    module.wait_for_database(connection)

    def run(*args, **kwargs):
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            pid = cursor.fetchone()[0]
        events.put(("locked", behavior, pid))
        if behavior in ("hold", "terminate"):
            if not release.wait(20):
                raise RuntimeError("test release timed out")
        if behavior == "terminate":
            # A closed/broken session must not be silently replaced.
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
            except Exception:
                connection.close()
            connection.ensure_connection()
        if behavior == "slow-migration":
            time.sleep(1.2)
        output = io.StringIO()
        call_command(*args, stdout=output, **kwargs)
        events.put(("output", behavior, output.getvalue()))

    def stall_lock_query(execute, sql, params, many, context):
        if "pg_try_advisory_lock" in sql:
            if behavior == "blackhole":
                blackhole.set()
                return execute(sql, params, many, context)
            return execute("SELECT pg_sleep(10)", None, many, context)
        return execute(sql, params, many, context)

    module.call_command = run
    if behavior in ("slow-query", "slow-migration", "blackhole"):
        module.WAIT_SECONDS = 1
    wrapper = connection.execute_wrapper(stall_lock_query) if behavior in ("slow-query", "blackhole") else nullcontext()
    try:
        with wrapper:
            module.migrate(connection)
        events.put(("done", behavior, ""))
    except Exception as error:
        events.put(("error", behavior, str(error)))
    finally:
        connection.close()
        if behavior == "blackhole":
            stop_proxy.set()
            listener.close()
            proxy_thread.join(2)
        temporary.cleanup()


@unittest.skipUnless(URL, "set AQUEDUCT_TEST_POSTGRES_URL to a disposable PostgreSQL database")
class PostgreSQLMigrationTests(unittest.TestCase):
    def setUp(self):
        self.ctx = multiprocessing.get_context("spawn")
        self.events = self.ctx.Queue()
        self.release = self.ctx.Event()
        self.processes = []

    def tearDown(self):
        self.release.set()
        for process in self.processes:
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()
            self.assertEqual(process.exitcode, 0)

    def start(self, behavior):
        process = self.ctx.Process(target=contender, args=(URL, self.events, self.release, behavior))
        process.start()
        self.processes.append(process)

    def event(self):
        return self.events.get(timeout=30)

    def test_contention_then_normal_noop(self):
        self.start("hold")
        self.assertEqual(self.event()[:2], ("locked", "hold"))
        self.start("second")
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=3)
        self.release.set()
        results = [self.event() for _ in range(5)]
        pairs = [event[:2] for event in results]
        self.assertEqual(set(pairs), {("output", "hold"), ("done", "hold"),
                                     ("locked", "second"), ("output", "second"), ("done", "second")})
        self.assertLess(pairs.index(("output", "hold")), pairs.index(("locked", "second")))
        output = next(event[2] for event in results if event[:2] == ("output", "second"))
        self.assertIn("No migrations to apply", output)

    def test_lock_query_deadline_does_not_limit_migration(self):
        self.start("slow-query")
        event = self.event()
        self.assertEqual(event[:2], ("error", "slow-query"))
        self.assertIn("Timed out waiting", event[2])
        self.start("blackhole")
        event = self.event()
        self.assertEqual(event[:2], ("error", "blackhole"))
        self.assertIn("Timed out waiting", event[2])
        self.start("slow-migration")
        self.assertEqual(self.event()[:2], ("locked", "slow-migration"))
        self.assertEqual(self.event()[:2], ("output", "slow-migration"))
        self.assertEqual(self.event()[:2], ("done", "slow-migration"))

    def test_failure_and_terminated_session_release_lock(self):
        import psycopg

        self.start("fail")
        self.assertEqual(self.event()[:2], ("locked", "fail"))
        failure = self.event()
        self.assertEqual(failure[:2], ("error", "fail"))
        self.assertIn("division by zero", failure[2])
        self.start("terminate")
        kind, _, pid = self.event()
        self.assertEqual(kind, "locked")
        with psycopg.connect(URL, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", [pid])
        self.release.set()
        event = self.event()
        self.assertEqual(event[:2], ("error", "terminate"))
        self.assertIn("refusing to reconnect", event[2])
        self.start("after")
        self.assertEqual(self.event()[:2], ("locked", "after"))
        self.assertEqual(self.event()[:2], ("output", "after"))
        self.assertEqual(self.event()[:2], ("done", "after"))
