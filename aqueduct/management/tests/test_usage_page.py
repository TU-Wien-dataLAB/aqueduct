from django.template import Context, Template
from django.test import SimpleTestCase
from django.urls import reverse

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
        self.assertContains(
            page, '<td class="whitespace-nowrap text-right tabular-nums">1.2 s</td>'
        )

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

    def test_status_rows_link_to_the_request_log_under_the_same_filters(self):
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", status_code=503)
        page = self.page(self.solo, span="1w", model="gpt-4.1-nano")
        server = page.context["statuses"][-1]
        self.assertEqual((server["count"], server["share"]), (1, "100%"))
        self.assertEqual(
            server["url"], f"{reverse('requests')}?span=7d&model=gpt-4.1-nano&status=server_error"
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
