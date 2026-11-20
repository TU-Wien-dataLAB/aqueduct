from datetime import timedelta

from django.core.cache import cache
from django.db import connection
from django.db.models import F
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from django.utils.http import urlencode

from management.models import Request, TeamMembership, UserGroup
from management.request_paging import PAGE_SIZE, Place
from management.tests.helpers import CHAT, EMBEDDING, RequestsPageTestBase, join, listed


class RequestsPaginationTest(RequestsPageTestBase):
    def test_pages_hold_page_size_rows_newest_first(self):
        now = timezone.now()
        for n in range(PAGE_SIZE + 1):
            tied = timedelta(minutes=min(n, PAGE_SIZE - 1))
            Request.objects.create(
                token=self.solo_key,
                path=f"/bulk-{n}",
                model=CHAT,
                status_code=200,
                timestamp=now - tied,
            )
        first = self.get(self.solo)
        content = first.content.decode()
        self.assertEqual(len(first.context["log"].rows), PAGE_SIZE)
        self.assertLess(content.index(">/bulk-0</div>"), content.index(">/bulk-1</div>"))
        self.assertIn(f"1\u2013{PAGE_SIZE} of {PAGE_SIZE + 2}", content)
        self.assertNotIn(">Previous</a>", content)

        second = self.client.get(self.pager_link(content, "Next"))
        self.assertEqual(self.paths_on(second), {"/own"})
        self.assertIn(f">/bulk-{PAGE_SIZE}</div>", content)
        self.assertIn(f">/bulk-{PAGE_SIZE - 1}</div>", second.content.decode())
        self.assertNotIn(">Next</a>", second.content.decode())

        back = self.client.get(self.pager_link(second.content.decode(), "Previous"))
        self.assertEqual(
            [row.request.path for row in back.context["log"].rows],
            [row.request.path for row in first.context["log"].rows],
        )

    def test_a_linked_request_opens_on_a_page_that_holds_it(self):
        for minutes in range(PAGE_SIZE + 5):
            self.log(self.solo_key, f"/bulk-{minutes}", ago=timedelta(minutes=minutes))
        own = Request.objects.get(path="/own")
        response = self.get(self.solo, key=self.solo_key.id, request=own.id)
        log = response.context["log"]
        self.assertEqual([row.request.id for row in log.rows if row.is_open], [own.id])
        content = response.content.decode()
        self.assertIn(">Previous</a>", content)
        self.assertIn(urlencode({"before": log.pager.here.before}), content)
        total = PAGE_SIZE + 6
        self.assertIn(f"{total}\u2013{total} of {total}", content)
        closed = self.client.get(self.pager_link(content, f"Hide details of request {own.id}"))
        self.assertEqual(
            closed.context["log"].rows,
            tuple(type(row)(row.request, row.key, False) for row in log.rows),
        )

    def test_a_linked_request_outside_the_viewers_scope_stays_closed(self):
        foreign = Request.objects.get(path="/foreign")
        log = self.get(self.solo, request=foreign.id).context["log"]
        self.assertEqual(log.pager.here, Place())
        self.assertFalse(any(row.is_open for row in log.rows))

    def test_a_malformed_cursor_or_id_shows_the_newest_page(self):
        huge = "9" * 30
        cursors = ("abc", "1,", ",1", "-5,3", "\N{SUPERSCRIPT ONE},1", "1,\N{SUPERSCRIPT ONE}")
        for cursor in (*cursors, f"{huge},1", f"1,{huge}", f"{10**17},1"):
            for direction in ("before", "after"):
                with self.subTest(cursor=cursor, direction=direction):
                    response = self.get(self.solo, **{direction: cursor})
                    self.assertEqual(self.paths_on(response), {"/own"})
        endless = "9" * 5000
        for cursor in (f"{endless},1", f"1,{endless}"):
            with self.subTest(cursor=cursor[:12]):
                self.assertEqual(self.paths_on(self.get(self.solo, before=cursor)), {"/own"})
        self.assertEqual(self.get(self.solo, before=f"1,1,{endless}").status_code, 200)
        for page in (endless, str(2**63 - 2), "0", "-1", "abc"):
            with self.subTest(page=page[:12]):
                self.assertEqual(self.get(self.solo, sort="-response", page=page).status_code, 200)
        for raw in ("\N{SUPERSCRIPT ONE}", huge, endless):
            with self.subTest(request=raw):
                self.assertEqual(self.paths_on(self.get(self.solo, request=raw)), {"/own"})
            with self.subTest(key=raw):
                self.assertEqual(self.get(self.solo, key=raw).status_code, 404)

    def test_a_page_past_the_oldest_request_links_back_to_the_newest(self):
        content = self.get(self.solo, before="1,1").content.decode()
        self.assertEqual(self.paths_on(self.get(self.solo, before="1,1")), set())
        self.assertIn(">Previous</a>", content)
        self.assertEqual(
            self.paths_on(self.client.get(self.pager_link(content, "Previous"))), {"/own"}
        )

    def pager_link(self, content: str, label: str) -> str:
        if label.startswith("Hide"):
            anchor = content.split(f'aria-label="{label}"', maxsplit=1)[0].rsplit("<a ", 1)[1]
        else:
            anchor = content.split(f">{label}</a>", maxsplit=1)[0].rsplit("<a ", 1)[1]
        href = anchor.split('href="', 1)[1].split('"', 1)[0].replace("&amp;", "&")
        return reverse("requests") + href

    def test_paginator_reuses_the_tab_counts(self):
        self.log(self.solo_key, "/limited", status_code=429)
        self.client.force_login(self.org_admin)
        for params in ({}, {"status": "429"}):
            cache.clear()
            with self.subTest(**params), CaptureQueriesContext(connection) as queries:
                log = self.client.get(reverse("requests"), params).context["log"]
            tab = next(tab for tab in log.tabs if tab.value == params.get("status"))
            self.assertEqual(log.count, tab.count)
            counts = [q["sql"] for q in queries if "COUNT(" in q["sql"].upper()]
            self.assertEqual(len(counts), 1, counts)

    def test_query_count_does_not_grow_with_rows(self):
        self.client.force_login(self.org_admin)
        with CaptureQueriesContext(connection) as few:
            self.client.get(reverse("requests"))
        for minutes in range(30):
            self.log(self.side_key, f"/bulk-{minutes}")
            self.log(self.others_key, f"/more-{minutes}")
        cache.clear()
        with CaptureQueriesContext(connection) as many:
            self.client.get(reverse("requests"))
        self.assertEqual(len(many), len(few))


@override_settings(REQUEST_RETENTION_DAYS=30)
class RequestsPagerTest(RequestsPageTestBase):
    TOTAL = 2 * PAGE_SIZE + 8

    def setUp(self):
        super().setUp()
        now = timezone.now()
        for n in range(self.TOTAL - 1):
            Request.objects.create(
                token=self.solo_key,
                path=f"/bulk-{n}",
                model=CHAT,
                status_code=200,
                timestamp=now - timedelta(minutes=n // 3),
                response_time_ms=None if n % 7 == 0 else n % 4,
            )
        self.client.force_login(self.solo)

    def open(self, url: str):
        return self.client.get(url)

    def follow(self, response, label: str):
        content = response.content.decode()
        if f">{label}</a>" not in content:
            return None
        anchor = content.split(f">{label}</a>", maxsplit=1)[0].rsplit("<a ", 1)[1]
        href = anchor.split('href="', 1)[1].split('"', 1)[0].replace("&amp;", "&")
        return self.client.get(reverse("requests") + href)

    def expected(self, *ordering) -> list[str]:
        own = Request.objects.filter(token=self.solo_key)
        return list(own.order_by(*ordering).values_list("path", flat=True))

    def label(self, response) -> str:
        log = response.context["log"]
        if log.pager.start is None:
            return f"{log.count} requests"
        return f"{log.pager.start}\u2013{log.end} of {log.count}"

    def test_next_and_previous_walk_every_request_once_in_order(self):
        for sort, ordering in (
            ({}, ("-timestamp", "-id")),
            ({"sort": "time"}, ("timestamp", "id")),
            ({"sort": "-response"}, (F("response_time_ms").desc(nulls_last=True), "-id")),
        ):
            with self.subTest(**sort):
                expected = self.expected(*ordering)
                page = self.get(self.solo, **sort)
                walked, labels, pages = [], [], []
                while page is not None:
                    walked += listed(page)
                    labels.append(self.label(page))
                    pages.append(listed(page))
                    page = self.follow(page, "Next")
                self.assertEqual(walked, expected)
                t = self.TOTAL
                self.assertEqual(
                    labels, [f"1\u201350 of {t}", f"51\u2013100 of {t}", f"101\u2013{t} of {t}"]
                )
                back = self.get(self.solo, **sort)
                for _ in range(2):
                    back = self.follow(back, "Next")
                for want in reversed(pages[:-1]):
                    back = self.follow(back, "Previous")
                    self.assertEqual(listed(back), want)
                self.assertIsNone(self.follow(back, "Previous"))

    def test_last_and_first_open_the_ends_of_the_log(self):
        t = self.TOTAL
        for sort, ordering, last_label, before_last in (
            ({}, ("-timestamp", "-id"), f"{t - 49}\u2013{t} of {t}", slice(t - 100, t - 50)),
            (
                {"sort": "time"},
                ("timestamp", "id"),
                f"{t - 49}\u2013{t} of {t}",
                slice(t - 100, t - 50),
            ),
            (
                {"sort": "-tokens"},
                (F("input_tokens").desc(), "-id"),
                f"101\u2013{t} of {t}",
                slice(50, 100),
            ),
        ):
            with self.subTest(**sort):
                expected = self.expected(*ordering)
                last = self.follow(self.get(self.solo, **sort), "Last")
                self.assertEqual(self.label(last), last_label)
                self.assertEqual(listed(last), expected[-len(listed(last)) :])
                self.assertIsNone(self.follow(last, "Next"))
                self.assertIsNone(self.follow(last, "Last"))
                previous = self.follow(last, "Previous")
                self.assertEqual(listed(previous), expected[before_last])
                first = self.follow(last, "First")
                self.assertEqual(listed(first), expected[:PAGE_SIZE])
                self.assertEqual(self.label(first), f"1\u201350 of {t}")

    def test_retention_note_shows_on_the_last_page_and_an_empty_log_only(self):
        note = "Requests are kept for 30 days"
        first = self.get(self.solo)
        self.assertNotContains(first, note)
        self.assertNotContains(self.follow(first, "Next"), note)
        self.assertContains(self.follow(first, "Last"), note)
        self.assertContains(self.get(self.solo, model=EMBEDDING), note)

    def test_a_deep_link_mid_log_shows_the_total_until_the_position_is_known(self):
        expected = self.expected("-timestamp", "-id")
        middle = Request.objects.get(path=expected[55])
        linked = self.get(self.solo, request=middle.id)
        self.assertEqual(listed(linked), expected[55:105])
        self.assertEqual(
            [r.request.id for r in linked.context["log"].rows if r.is_open], [middle.id]
        )
        self.assertEqual(self.label(linked), f"{self.TOTAL} requests")
        self.assertIn(f"{self.TOTAL} requests", linked.content.decode())
        end = self.follow(linked, "Next")
        self.assertEqual(listed(end), expected[105:])
        self.assertEqual(self.label(end), f"106\u2013{self.TOTAL} of {self.TOTAL}")
        self.assertEqual(listed(self.follow(linked, "Previous")), expected[5:55])

    def test_requests_newer_than_the_cached_count_stay_reachable(self):
        first = self.get(self.solo, sort="-tokens")
        last = self.follow(first, "Last")
        self.assertIsNone(self.follow(last, "Next"))
        for n in range(PAGE_SIZE):
            self.log(self.solo_key, f"/late-{n}", input_tokens=0)
        late = self.follow(self.follow(last, "First"), "Last")
        self.assertEqual(len(listed(late)), PAGE_SIZE)
        self.assertEqual(self.label(late), f"{self.TOTAL} requests")
        beyond = self.follow(late, "Next")
        self.assertEqual(len(listed(beyond)), self.TOTAL - 2 * PAGE_SIZE)
        self.assertIsNone(self.follow(beyond, "Next"))

    def test_closing_a_row_keeps_the_page(self):
        last = self.follow(self.get(self.solo), "Last")
        row = last.context["log"].rows[3].request
        opened = self.client.get(reverse("requests"), {"page": "last", "request": row.id})
        self.assertEqual(listed(opened), listed(last))
        self.assertTrue(opened.context["log"].rows[3].is_open)


class RequestsCountCacheTest(RequestsPageTestBase):
    def all_count(self, user, **params) -> int:
        return self.get(user, **params).context["log"].tabs[0].count

    def test_cached_counts_are_never_shared_between_viewers(self):
        team_admin = self.user("side-admin")
        join(team_admin, self.side_team, is_admin=True)
        for user, params, count in (
            (self.solo, {}, 1),
            (self.other, {}, 2),
            (self.member, {}, 1),
            (team_admin, {}, 1),
            (self.org_admin, {}, 5),
            (self.other_org_admin, {}, 1),
            (self.admin, {}, 5),
            (self.org_admin, {"scope": "mine"}, 0),
            (self.solo, {"q": "o"}, 1),
            (self.other, {"q": "o"}, 2),
            (self.org_admin, {"q": "o"}, 3),
        ):
            with self.subTest(user=user.username, **params):
                self.assertEqual(self.all_count(user, **params), count)

    def test_a_new_role_or_team_reads_its_own_counts(self):
        self.assertEqual(self.all_count(self.solo), 1)
        TeamMembership.objects.create(user_profile=self.solo.profile, team=self.team)
        self.assertEqual(self.all_count(self.solo), 2)
        self.solo.profile.group = UserGroup.ORG_ADMIN
        self.assertEqual(self.all_count(self.solo), 5)

    def test_counts_are_reused_until_the_selection_changes(self):
        self.assertEqual(self.all_count(self.solo), 1)
        self.log(self.solo_key, "/newer")
        self.assertEqual(self.all_count(self.solo), 1)
        self.assertEqual(self.all_count(self.solo, status="failed"), 1)
        self.assertEqual(self.all_count(self.solo, span="1w"), 2)
        self.assertEqual(self.all_count(self.solo, q="/"), 2)
