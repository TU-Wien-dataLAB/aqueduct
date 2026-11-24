"""Time the management pages and snapshot what they show, as three roles; bench.sh runs it.

profile and snapshot sign in as the platform admin, the org admin and a plain member of the
demo's seed and request every management page through Django's test client, in-process: no
server and no network. Both freeze the clock at the newest seeded request, so time windows
such as "last 24 h" cover the same rows on every run, however old the database is.
"""

# Standalone script run by bench.sh, not a package.
# ruff: noqa: INP001, T201

import argparse
import dataclasses
import datetime as dt
import enum
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

ROLES = {"admin": "admin@example.com", "org-admin": "you@example.com", "member": "ada@example.com"}
"""The signed-in users of scripts/seed.py, by role."""
REPO_ROOT = os.environ.get("AQUEDUCT_BENCH_REPO", ".")
"""The checkout whose commit the results record; bench.sh sets it."""
ROWS_LIMIT = 1000
"""The most items the snapshot keeps of one queryset the page did not already cut."""


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", REPO_ROOT, *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def commit() -> str:
    """The checkout's short commit, with `-dirty` while tracked files have changes."""
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def default_out(kind: str) -> Path:
    return Path(os.environ["AQUEDUCT_BENCH_DIR"]) / f"{kind}-{commit()}.json"


def setup_django() -> None:
    import django

    django.setup()


def frozen_at() -> dt.datetime:
    """The newest request's minute, rounded up: the instant both profile and snapshot see."""
    from django.db.models import Max
    from django.utils import timezone

    from management.models import Request

    newest = Request.objects.aggregate(newest=Max("timestamp"))["newest"] or timezone.now()
    return newest.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)


@contextmanager
def frozen_clock(at: dt.datetime) -> Iterator[None]:
    """Every `timezone.now()`, and so `localdate()`, returns `at`."""
    with mock.patch("django.utils.timezone.now", return_value=at):
        yield


def pages(role: str) -> list[tuple[str, str]]:
    """Each management page the role opens, with the filtered variants worth timing."""
    from django.contrib.auth import get_user_model
    from django.db.models import Count
    from django.urls import reverse

    from management.models import Request

    profile = get_user_model().objects.get(email=ROLES[role]).profile
    busiest = (
        Request.objects.filter(token__in=profile.visible_tokens())
        .values("token")
        .annotate(n=Count("id"))
        .order_by("-n", "token")
        .first()
    )
    team = profile.visible_teams(profile.org).order_by("pk").first()
    log, usage = reverse("requests"), reverse("usage")
    found = [
        ("API keys", reverse("tokens")),
        ("Requests, default", log),
        ("Requests, last 24 h", f"{log}?span=1d"),
        ("Requests, failed", f"{log}?status=failed"),
        ("Requests, sort by input", f"{log}?sort=-input"),
        ("Requests, search 'chat'", f"{log}?q=chat"),
        ("Requests, page 200", f"{log}?page=200"),
        ("Usage, 24 h", f"{usage}?span=1d"),
        ("Usage, 7 days", f"{usage}?span=1w"),
        ("Usage, 30 days", f"{usage}?span=1m"),
        ("Models", reverse("models")),
        ("MCP servers", reverse("mcp_servers")),
        ("Files", reverse("files")),
        ("Batches", reverse("batches")),
        ("Vector stores", reverse("vector_stores")),
        ("Settings", reverse("settings")),
        ("Teams", reverse("org")),
    ]
    if busiest:
        key = busiest["token"]
        found += [
            ("Key page", reverse("token_detail", args=[key])),
            ("Requests, one key", f"{log}?key={key}"),
            ("Usage, one key, 7 days", f"{usage}?span=1w&token={key}"),
        ]
    if team:
        found.append(("Team page", reverse("team", args=[team.id])))
    if profile.is_org_admin(profile.org):
        found.append(("All keys", reverse("all_keys")))
    if profile.is_admin():
        found.append(("Usage, all orgs, 7 days", f"{usage}?span=1w&org=all"))
    return found


def visit(each: Callable[[str, str, str, Any], dict[str, Any]]) -> tuple[dt.datetime, list]:
    """Call `each(role, page, url, client)` for every page of every role, signed in as the
    role, under the frozen clock; return the instant and the results."""
    from django.contrib.auth import get_user_model
    from django.test import Client

    at = frozen_at()
    results = []
    with frozen_clock(at):
        for role, email in ROLES.items():
            client = Client(HTTP_HOST="localhost")
            client.force_login(get_user_model().objects.get(email=email))
            for page, url in pages(role):
                results.append(
                    {"role": role, "page": page, "url": url, **each(role, page, url, client)}
                )
    return at, results


def header(at: dt.datetime) -> dict[str, Any]:
    from django.db import connection

    from management.models import Request

    return {
        "commit": commit(),
        "subject": git("log", "-1", "--format=%s"),
        "database": f"{connection.vendor} {connection.pg_version // 10000}"
        if connection.vendor == "postgresql"
        else connection.vendor,
        "requests": Request.objects.count(),
        "frozen_at": at.isoformat(),
    }


# --- profile --------------------------------------------------------------------------------


def profile(args: argparse.Namespace) -> None:
    from django.db import connection

    def each(role: str, page: str, url: str, client: Any) -> dict[str, Any]:
        client.get(url)  # Warm up: connection, templates, router config.
        runs = []
        for _ in range(args.runs):
            sql: list[float] = []

            def timed(
                execute: Callable, sql_text: str, params: Any, many: bool, context: Any
            ) -> Any:
                started = time.perf_counter()
                try:
                    return execute(sql_text, params, many, context)
                finally:
                    sql.append(time.perf_counter() - started)  # noqa: B023

            with connection.execute_wrapper(timed):
                started = time.perf_counter()
                response = client.get(url)
                wall = time.perf_counter() - started
            runs.append((wall, sum(sql), len(sql), response.status_code))
        wall, sql_time, queries, status = min(runs)
        found = {
            "status": status,
            "wall_ms": round(wall * 1000, 1),
            "sql_ms": round(sql_time * 1000, 1),
            "queries": queries,
            "wall_median_ms": round(statistics.median(run[0] for run in runs) * 1000, 1),
        }
        print(
            f"{role:<9} {page:<26} {status}  wall {found['wall_ms']:>7.1f} ms"
            f"  sql {found['sql_ms']:>7.1f} ms  queries {queries:>3}",
            flush=True,
        )
        return found

    at, results = visit(each)
    out = args.out or default_out("profile")
    out.write_text(json.dumps({**header(at), "runs": args.runs, "results": results}, indent=1))
    print(f"\nWrote {out}")


def table(args: argparse.Namespace) -> None:
    """A Markdown table of one metric: a row per page (and role), a column per profile."""
    profiles = [json.loads(Path(name).read_text()) for name in args.files]
    labels = [
        found.get("commit") or Path(name).stem
        for found, name in zip(profiles, args.files, strict=True)
    ]
    cells: dict[tuple[str, str], list[str]] = {}
    for column, found in enumerate(profiles):
        for row in found["results"]:
            if args.role and row["role"] != args.role:
                continue
            value = row[args.metric]
            text = f"{value:.0f}" if isinstance(value, float) else str(value)
            if row["status"] != 200:  # noqa: PLR2004
                text += f" ({row['status']})"
            cells.setdefault((row["role"], row["page"]), [""] * len(profiles))[column] = text
    lead = ["Page"] if args.role else ["Role", "Page"]
    print("| " + " | ".join(lead + labels) + " |")
    print("|" + "---|" * len(lead) + "--:|" * len(labels))
    for (role, page), values in cells.items():
        print("| " + " | ".join(([page] if args.role else [role, page]) + values) + " |")


# --- snapshot -------------------------------------------------------------------------------


def plain(value: Any, seen: frozenset[int] = frozenset()) -> Any:  # noqa: C901
    """`value` as JSON: models as their fields plus what the view set on them, querysets as
    lists, sets sorted, floats to six places; other objects as their public attributes."""
    from django.core.paginator import Page, Paginator
    from django.db.models import Model, QuerySet
    from django.forms import BaseForm
    from django.utils.functional import Promise

    if isinstance(value, enum.Enum):
        return plain(value.value, seen)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float | Decimal):
        return round(float(value), 6)
    if isinstance(value, Promise):
        return str(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if id(value) in seen:
        return "<cycle>"
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return {str(key): plain(item, seen) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((plain(item, seen) for item in value), key=json.dumps)
    if isinstance(value, list | tuple):
        return [plain(item, seen) for item in value]
    if isinstance(value, QuerySet):
        return [plain(item, seen) for item in value[:ROWS_LIMIT]]
    if isinstance(value, Paginator):
        return {"count": value.count, "num_pages": value.num_pages, "per_page": value.per_page}
    if isinstance(value, Page):
        return {
            "number": value.number,
            "has_previous": value.has_previous(),
            "has_next": value.has_next(),
            "start_index": value.start_index(),
            "end_index": value.end_index(),
        }
    if isinstance(value, Model):
        fields = {field.attname for field in value._meta.concrete_fields}
        return {
            "model": value._meta.label,
            **{name: plain(getattr(value, name), seen) for name in sorted(fields)},
            **{
                name: plain(item, seen)
                for name, item in vars(value).items()
                if not name.startswith("_") and name not in fields
            },
        }
    if isinstance(value, BaseForm):
        return {"form": type(value).__name__, "initial": plain(value.initial, seen)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name), seen) for f in dataclasses.fields(value)}
    if hasattr(value, "__dict__") and not callable(value):
        public = {name: item for name, item in vars(value).items() if not name.startswith("_")}
        return {"type": type(value).__name__, **plain(public, seen)}
    return f"<{type(value).__name__}>"


def snapshot(args: argparse.Namespace) -> None:
    def each(role: str, page: str, url: str, client: Any) -> dict[str, Any]:
        response = client.get(url)
        # The dict the view hands its template: response.context without the context
        # processors' and the included templates' layers.
        context = getattr(response, "context_data", None) or {}
        context = {key: value for key, value in context.items() if key != "view"}
        print(f"{role:<9} {page:<26} {response.status_code}  {len(context)} keys", flush=True)
        return {"status": response.status_code, "context": plain(context)}

    at, results = visit(each)
    out = args.out or default_out("snapshot")
    out.write_text(json.dumps({**header(at), "pages": results}, indent=1, sort_keys=True))
    print(f"\nWrote {out}")


def differences(base: Any, new: Any, path: str) -> Iterator[tuple[str, Any, Any]]:
    if isinstance(base, dict) and isinstance(new, dict):
        for key in sorted(base.keys() | new.keys()):
            yield from differences(
                base.get(key, "<missing>"), new.get(key, "<missing>"), f"{path}.{key}"
            )
    elif isinstance(base, list) and isinstance(new, list):
        if len(base) != len(new):
            yield f"{path} length", len(base), len(new)
        for index, (old, now) in enumerate(zip(base, new, strict=False)):
            yield from differences(old, now, f"{path}[{index}]")
    elif base != new:
        yield path, base, new


def short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 80 else f"{text[:77]}..."  # noqa: PLR2004


def compare(args: argparse.Namespace) -> None:
    """Print every page, role and context key where NEW differs from BASE; exit 1 if any."""
    base, new = (json.loads(Path(name).read_text()) for name in (args.base, args.new))
    if base["frozen_at"] != new["frozen_at"] or base["requests"] != new["requests"]:
        print(
            f"The snapshots saw different data: {base['requests']} requests at"
            f" {base['frozen_at']} against {new['requests']} at {new['frozen_at']}."
        )
        sys.exit(1)
    pages = {(page["role"], page["page"]): page for page in base["pages"]}
    newer = {(page["role"], page["page"]): page for page in new["pages"]}
    found = 0
    for where in [*pages, *(where for where in newer if where not in pages)]:
        if where not in pages or where not in newer:
            print(f"{where[0]} | {where[1]}: only in {'NEW' if where in newer else 'BASE'}")
            found += 1
            continue
        old, now = pages[where], newer[where]
        diffs = list(differences(old["status"], now["status"], "status"))
        for key in sorted(old["context"].keys() | now["context"].keys()):
            if key not in args.ignore:
                missing = "<missing>"
                diffs += differences(
                    old["context"].get(key, missing), now["context"].get(key, missing), key
                )
        for path, was, became in diffs[: args.limit]:
            print(f"{where[0]} | {where[1]} | {path}: {short(was)} -> {short(became)}")
        if len(diffs) > args.limit:
            print(f"{where[0]} | {where[1]}: {len(diffs) - args.limit} more differences")
        found += len(diffs)
    ignored = f", ignoring {', '.join(args.ignore)}" if args.ignore else ""
    if found:
        print(f"\n{found} differences{ignored}.")
        sys.exit(1)
    print(f"Snapshots match: {len(pages)} pages{ignored}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    found = commands.add_parser("profile", help="time every page as each role")
    found.add_argument("--runs", type=int, default=5, help="timed runs per page, after a warm-up")
    found.add_argument("--out", type=Path, help="JSON file (default: in the results directory)")
    found = commands.add_parser("snapshot", help="record what every page shows as each role")
    found.add_argument("--out", type=Path, help="JSON file (default: in the results directory)")
    found = commands.add_parser("compare", help="diff two snapshots")
    found.add_argument("base", type=Path)
    found.add_argument("new", type=Path)
    found.add_argument(
        "--ignore", action="append", default=[], help="a context key a change alters on purpose"
    )
    found.add_argument("--limit", type=int, default=10, help="differences shown per page")
    found = commands.add_parser("table", help="print profiles as a Markdown table")
    found.add_argument("files", nargs="+", type=Path)
    found.add_argument("--metric", default="wall_ms", choices=["wall_ms", "sql_ms", "queries"])
    found.add_argument("--role", choices=list(ROLES), help="only this role, without a role column")
    args = parser.parse_args()
    if args.command in ("profile", "snapshot"):
        setup_django()
    {"profile": profile, "snapshot": snapshot, "compare": compare, "table": table}[args.command](
        args
    )


if __name__ == "__main__":
    main()
