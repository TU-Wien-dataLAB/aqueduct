from datetime import timedelta

from django.db import connection
from django.template import Context, Template
from django.test import SimpleTestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from management.models import Request
from management.tests.helpers import RoleWorld


class UsagePageTest(RoleWorld):
    def page(self, user, **params):
        response = self.as_user(user).get(reverse("usage"), params)
        self.assertEqual(response.status_code, 200)
        return response

    def test_totals_count_the_viewers_requests(self):
        Request.objects.create(
            token=self.solo_key, model="gpt-4.1-nano", status_code=200, input_tokens=1200
        )
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", status_code=500)
        Request.objects.create(token=self.member_key, model="gpt-4.1-nano", status_code=200)
        page = self.page(self.solo)
        self.assertEqual(page.context["total_requests"], 2)
        cards = {kpi["label"]: (kpi["value"], kpi["note"]) for kpi in page.context["kpis"]}
        self.assertEqual(cards["Input tokens"], ("1.2K", "0% from cache"))
        self.assertEqual(cards["Failed"], ("50%", "0 rate-limited"))
        self.assertContains(page, '<span class="text-2xl font-semibold tabular-nums">1.2K</span>')

    def test_renders_for_every_role(self):
        for user in (self.solo, self.member, self.org_admin, self.admin):
            with self.subTest(user=user.username):
                self.page(user)
                self.page(user, org="all", rows="all", span="1m")

    def test_usage_by_model_sums_tokens(self):
        Request.objects.create(
            token=self.solo_key,
            model="gpt-4.1-nano",
            input_tokens=1200,
            cached_input_tokens=300,
            output_tokens=30,
            response_time_ms=900,
        )
        Request.objects.create(
            token=self.solo_key,
            model="gpt-4.1-nano",
            input_tokens=800,
            status_code=429,
            response_time_ms=1500,
        )
        Request.objects.create(token=self.member_key, model="gpt-4.1-nano", input_tokens=5)
        page = self.page(self.solo)
        (row,) = page.context["per_model"]
        self.assertEqual(
            {name: row[name] for name in ("model", "requests", "input", "cached", "output")},
            {"model": "gpt-4.1-nano", "requests": 2, "input": 2000, "cached": 300, "output": 30},
        )
        self.assertEqual((row["rate_limited"], row["avg_ms"], row["mode"]), (1, 1200, "chat"))
        self.assertContains(page, '<td class="text-right tabular-nums">2K</td>', html=True)
        self.assertContains(page, '<h2 id="per-model-title" class="card-title">By model</h2>')
        self.assertContains(
            page, '<td class="whitespace-nowrap text-right tabular-nums">1.2 s</td>'
        )

    def requests_in_two_orgs(self):
        """Requests of four keys of both orgs over three days, with two models and statuses."""
        now = timezone.now()
        for key, model, status, hours_ago, tokens, response_ms in [
            (self.solo_key, "gpt-4.1-nano", 200, 1, 100, 900),
            (self.solo_key, "gpt-4.1-nano", 429, 30, 0, None),
            (self.member_key, "whisper-1", 200, 2, 40, 300),
            (self.member_key, "gpt-4.1-nano", 500, 50, 0, None),
            (self.pipeline_key, "gpt-4.1-nano", 200, 3, 70, 1500),
            (self.foreign_key, "whisper-1", 200, 4, 999, 100),
        ]:
            Request.objects.create(
                token=key,
                model=model,
                status_code=status,
                timestamp=now - timedelta(hours=hours_ago),
                input_tokens=tokens,
                output_tokens=tokens // 10,
                response_time_ms=response_ms,
                user_id="alice" if key == self.pipeline_key else "",
            )

    def test_every_table_adds_up_to_the_totals(self):
        self.requests_in_two_orgs()
        for user, params in [
            (self.org_admin, {"span": "1w"}),
            (self.admin, {"span": "1w", "org": "all"}),
            (self.member, {"span": "1w", "token": self.pipeline_key.id}),
            (self.solo, {"span": "1d"}),
        ]:
            with self.subTest(user=user.username, **params):
                context = self.page(user, **params).context
                totals = context["totals"]
                for table in ("per_model", "top_items"):
                    for name in ("requests", "input", "output", "failed", "rate_limited"):
                        rows = context[table]
                        self.assertEqual(sum(row[name] for row in rows), totals[name], table)

    def test_all_orgs_rank_the_orgs_by_tokens(self):
        self.requests_in_two_orgs()
        rows = self.page(self.admin, span="1w", org="all").context["top_items"]
        self.assertEqual(
            [(row["name"], row["org_id"], row["requests"], row["tokens"]) for row in rows],
            [("other-org", self.other_org.id, 1, 1098), ("lab-org", self.org.id, 5, 231)],
        )
        rows = self.page(self.org_admin, span="1w").context["top_items"]
        self.assertEqual(
            [(row["name"], row["team_name"], row["tokens"]) for row in rows],
            [("solo-laptop", None, 110), ("pipeline", "NLP Lab", 77), ("member-laptop", None, 44)],
        )

    def test_response_time_averages_the_requests_that_have_one(self):
        self.requests_in_two_orgs()
        Request.objects.create(token=self.solo_key, model="gpt-image-1", status_code=500)
        per_model = self.page(self.org_admin, span="1w").context["per_model"]
        self.assertEqual(
            {row["model"]: row["avg_ms"] for row in per_model},
            {"gpt-4.1-nano": 1200, "whisper-1": 300, "gpt-image-1": None},
        )

    def test_reads_the_requests_in_one_pass(self):
        self.requests_in_two_orgs()
        key_page = reverse("token_detail", args=[self.solo_key.id])
        for url, params in [
            (reverse("usage"), {"span": "1w"}),
            (reverse("usage"), {"span": "1w", "org": "all"}),
            (reverse("usage"), {"span": "1w", "token": self.pipeline_key.id}),
            (key_page, {}),
        ]:
            with self.subTest(url=url, **params), CaptureQueriesContext(connection) as queries:
                self.assertEqual(self.as_user(self.admin).get(url, params).status_code, 200)
            passes = [
                query for query in queries if 'SUM("management_request"."input' in query["sql"]
            ]
            self.assertEqual(len(passes), 1)

    def test_filters_by_model(self):
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", input_tokens=10)
        Request.objects.create(token=self.solo_key, model="whisper-1", input_tokens=20)
        self.assertEqual(self.page(self.solo, model="whisper-1").context["totals"]["input"], 20)
        unknown = self.page(self.solo, model="not-on-this-gateway")
        self.assertEqual(
            (unknown.context["selected_model"], unknown.context["totals"]["input"]), ("", 30)
        )

    def test_key_menu_and_links_offer_only_keys_the_viewer_may_open(self):
        for key in (self.solo_key, self.member_key, self.pipeline_key, self.foreign_key):
            Request.objects.create(token=key, model="gpt-4.1-nano", input_tokens=10)
        # Main's scope also counts another person's key's requests sent with the viewer's email.
        Request.objects.create(
            token=self.member_key, model="gpt-4.1-nano", user_id=self.solo.email, input_tokens=5
        )
        for user, names in (
            (self.solo, ["solo-laptop"]),
            (self.member, ["member-laptop", "pipeline"]),
            (self.org_admin, ["member-laptop", "pipeline", "solo-laptop"]),
        ):
            with self.subTest(user=user.username):
                self.assertEqual([key.name for key in self.page(user).context["keys"]], names)
        page = self.page(self.solo)
        self.assertEqual(
            [row["name"] for row in page.context["top_items"]], ["solo-laptop", "member-laptop"]
        )
        self.assertContains(page, reverse("token_detail", args=[self.solo_key.id]))
        self.assertNotContains(page, reverse("token_detail", args=[self.member_key.id]))

    def test_a_picked_key_shows_its_user_ids(self):
        Request.objects.create(token=self.pipeline_key, model="gpt-4.1-nano", user_id="alice")
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", user_id="bob")
        page = self.page(self.member, token=self.pipeline_key.id)
        self.assertEqual([row["user_id"] for row in page.context["top_items"]], ["alice"])
        self.assertContains(page, "By user ID")

    def test_the_time_frame_marks_custom_only_while_a_custom_range_is_shown(self):
        current = 'id="custom-toggle" class="btn btn-sm join-item" aria-current="true"'
        self.assertNotContains(self.page(self.solo), current)
        custom = self.page(
            self.solo, span="custom", start="2026-01-01T00:00", end="2026-01-02T00:00"
        )
        self.assertEqual(custom.context["selected_span"], "custom")
        self.assertContains(custom, current)

    def test_status_rows_link_to_the_request_log_under_the_same_filters(self):
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", status_code=503)
        page = self.page(self.solo, span="1w", model="gpt-4.1-nano")
        server = page.context["statuses"][-1]
        self.assertEqual((server["count"], server["share"]), (1, "100%"))
        self.assertEqual(
            server["url"], f"{reverse('requests')}?span=1w&model=gpt-4.1-nano&status=server_error"
        )


class UsageFiltersTest(SimpleTestCase):
    def render(self, source: str, **context) -> str:
        return Template("{% load ui %}" + source).render(Context(context))

    def test_compact_counts(self):
        shown = [
            self.render("{{ n|compact }}", n=n) for n in (0, 950, 1234, 33_840, 999_960, 4.1e9)
        ]
        self.assertEqual(shown, ["0", "950", "1.2K", "33.8K", "1M", "4.1B"])

    def test_percent_and_duration(self):
        self.assertEqual(self.render("{{ 6|percent:100 }} {{ 6|percent:0 }}"), "6% —")
        self.assertEqual(
            self.render(
                "{{ a|duration }} {{ b|duration }} {{ c|duration }}", a=258, b=4560, c=None
            ),
            "258 ms 4.6 s —",
        )

    def test_expiry(self):
        now = timezone.now()
        days = [None, *(now + timedelta(days=n) for n in (-0.01, 13, 15))]
        self.assertEqual(
            [self.render("{{ at|expiry }}", at=at) for at in days], ["", "expired", "soon", ""]
        )
