import re
from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from management.models import Request
from management.tests.helpers import CHAT, EMBEDDING, KEY_NAMES, UsagePageTestBase, make_token

User = get_user_model()


class UsageKeySelectionTests(UsagePageTestBase):
    def selected(self, user, key, **kwargs):
        self.client.force_login(user)
        return self.client.get(reverse("usage"), {"key": key.id}, **kwargs)

    def test_member_cannot_select_another_members_key(self):
        response = self.selected(self.solo, self.member_key)
        self.assertEqual(response.status_code, 404)
        self.assertNotContains(response, self.member_key.name, status_code=404)

    def test_the_old_token_parameter_selects_nothing(self):
        self.client.force_login(self.solo)
        response = self.client.get(reverse("usage"), {"token": self.member_key.id})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["filters"].key)
        self.assertNotContains(response, self.member_key.name)

    def test_member_selects_own_key_and_team_service_account_key(self):
        for key in (self.member_key, self.pipeline_key):
            with self.subTest(key=key.name):
                response = self.selected(self.member, key)
                self.assertEqual(response.context["filters"].key.token, key)

    def test_org_admin_selects_any_key_of_the_org(self):
        response = self.selected(self.org_admin, self.solo_key)
        self.assertEqual(response.context["filters"].key.token, self.solo_key)

    def test_org_admin_cannot_select_a_key_of_another_org(self):
        self.assertEqual(self.selected(self.org_admin, self.foreign_key).status_code, 404)


class UsageAccessTest(UsagePageTestBase):
    def test_the_analytics_page_is_called_usage(self):
        response = self.page(self.solo)
        self.assertContains(response, '<h1 class="break-words text-3xl font-semibold">Usage</h1>')
        self.assertContains(response, "<title>Usage · Aqueduct AI Gateway</title>")
        self.assertNotContains(response, "Overview")

    def test_user_with_one_key_gets_no_key_picker_and_no_top_keys(self):
        response = self.page(self.solo)
        self.assert_keys(response)
        self.assertEqual(self.kpi(response, "Requests"), "1")
        self.assertNotContains(response, '<span class="sr-only">Key: </span>')
        self.assertNotContains(response, "Top keys")
        self.assertNotContains(response, "All organizations")

    def test_user_with_several_keys_picks_from_all_keys_only(self):
        make_token(self.solo, "solo-desktop")
        response = self.page(self.solo)
        self.assertContains(response, '<span class="sr-only">Key: </span>')
        self.assertContains(response, 'aria-current="true" class="menu-active">All keys</a>')
        self.assertNotContains(response, "My keys")
        self.assertNotContains(response, 'name="key_q"')
        self.assertContains(response, "Top keys")
        self.assertEqual([key.detail for key in response.context["keys"]], [""])

    def test_org_admin_sees_every_key_of_their_org(self):
        response = self.page(self.org_admin)
        self.assert_keys(response, "solo-laptop", "member-laptop", "pipeline")
        self.assertEqual(self.kpi(response, "Requests"), "3")

    def test_platform_admin_sees_the_org_they_are_in(self):
        response = self.page(self.admin)
        self.assert_keys(response, "solo-laptop", "member-laptop", "pipeline")
        self.assertEqual(self.kpi(response, "Requests"), "3")

    def test_admin_sees_every_org_split_by_org(self):
        response = self.page(self.admin, org="all")
        self.assert_keys(response, *KEY_NAMES)
        self.assertContains(response, "By organization")
        self.assertContains(response, "other-org")
        self.assertEqual(self.kpi(response, "Requests"), "4")
        self.assertNotContains(self.page(self.admin, org=self.other_org.id), "By organization")

    def test_all_orgs_names_no_org_for_a_key_whose_owner_has_no_profile(self):
        root = User.objects.create_superuser("root", "root@example.com", "pw")
        self.log(make_token(root, "root-laptop"), timedelta(hours=1), 200, input_tokens=10)
        response = self.page(self.admin, org="all")
        self.assertContains(response, "root@example.com · No organization")
        self.assertNotContains(response, "· None")

    def test_key_filter_lets_an_org_admin_name_any_key_of_their_org(self):
        response = self.page(self.org_admin, key=self.member_key.id)
        self.assertEqual(self.kpi(response, "Requests"), "1")
        self.assertContains(response, "By user ID")


class UsageFiguresTest(UsagePageTestBase):
    def test_totals_count_tokens_cache_hits_and_failures(self):
        response = self.page(self.org_admin)
        self.assertEqual(self.kpi(response, "Input tokens"), "6K")
        self.assertEqual(self.kpi(response, "Output tokens"), "1.5K")
        self.assertContains(response, "4% from cache")
        self.assertEqual(self.kpi(response, "Requests"), "3")
        self.assertEqual(self.kpi(response, "Failed"), "33%")

    def test_requests_are_split_by_status(self):
        response = self.page(self.org_admin)
        self.assertEqual(self.status_count(response, "OK"), "1")
        self.assertEqual(self.status_count(response, "Rate-limited"), "1")
        self.assertEqual(self.status_count(response, "Client error"), "0")
        self.assertEqual(self.status_count(response, "Server error"), "1")

    def test_time_frame_limits_the_requests_counted(self):
        self.log(self.solo_key, timedelta(days=3), 200)
        self.assertEqual(self.kpi(self.page(self.solo), "Requests"), "1")
        self.assertEqual(self.kpi(self.page(self.solo, span="1w"), "Requests"), "2")

    def test_model_filter_keeps_one_model(self):
        self.log(self.solo_key, timedelta(hours=1), 200, model=EMBEDDING, input_tokens=9)
        self.assertEqual(self.kpi(self.page(self.solo), "Requests"), "2")
        response = self.page(self.solo, model=EMBEDDING)
        self.assertEqual(self.kpi(response, "Requests"), "1")
        self.assertEqual(self.kpi(response, "Input tokens"), "9")

    def test_requests_to_models_the_gateway_does_not_serve_are_counted_and_grouped(self):
        self.log(self.solo_key, timedelta(hours=1), 400, model="gpt-5")
        self.log(self.solo_key, timedelta(hours=1), 400, model="gtp-4")
        response = self.page(self.solo)
        self.assertEqual(self.kpi(response, "Requests"), "3")
        self.assertEqual(self.kpi(response, "Failed"), "67%")
        self.assertEqual(self.status_count(response, "Client error"), "2")
        rows = response.context["models"]
        self.assertEqual([row.name for row in rows], [CHAT, "Unknown model names"])
        self.assertEqual(rows[1].activity.requests, 2)
        self.assertIsNone(rows[1].log_url)
        self.assertNotContains(response, "gpt-5")
        self.assertNotContains(response, "gtp-4")

    def test_the_model_filter_finds_a_name_the_gateway_does_not_serve(self):
        self.log(self.solo_key, timedelta(hours=1), 400, model="gpt-5")
        response = self.page(self.solo, model="gpt-5")
        self.assertEqual(self.kpi(response, "Requests"), "1")
        self.assertEqual(self.status_count(response, "Client error"), "1")

    def test_a_rate_limited_request_is_not_a_failed_one(self):
        response = self.page(self.org_admin)
        self.assertEqual(self.status_count(response, "Rate-limited"), "1")
        self.assertEqual(self.kpi(response, "Failed"), "33%")
        self.assertContains(response, "1 rate-limited")
        failed = {tab.value: tab.count for tab in self.log_tabs(self.org_admin)}["failed"]
        self.assertEqual(failed, 1)

    def log_tabs(self, user):
        self.client.force_login(user)
        return self.client.get(reverse("requests")).context["log"].tabs

    def test_the_page_says_which_refusals_are_not_logged(self):
        self.assertContains(
            self.page(self.solo), "Requests the gateway stops at a limit are not logged."
        )

    def test_the_four_tiles_name_their_units(self):
        labels = [kpi.label for kpi in self.page(self.org_admin).context["kpis"]]
        self.assertEqual(labels, ["Requests", "Input tokens", "Output tokens", "Failed"])

    def test_the_weekday_and_hour_heatmap_is_gone(self):
        response = self.page(self.org_admin)
        self.assertNotContains(response, "Usage by weekday and hour")
        self.assertNotContains(response, "org-activity")
        self.assertNotIn("heat", response.context)

    def test_the_chart_is_named_for_its_step_and_carries_a_scale(self):
        response = self.page(self.org_admin)
        self.assertContains(response, "Tokens per hour")
        self.assertContains(
            response, '<span class="chart-top muted text-xs tabular-nums">7.5K tokens</span>'
        )
        self.assertContains(response, ">3 requests</span>")
        week = self.page(self.org_admin, span="1w")
        self.assertContains(week, "Requests per day")

    @override_settings(REQUEST_RETENTION_DAYS=7)
    def test_retention_warning_when_the_time_frame_reaches_past_retention(self):
        self.assertContains(self.page(self.solo, span="1m"), "older ones are not counted here")
        self.assertNotContains(self.page(self.solo, span="1w"), "older ones are not counted here")

    def test_the_time_frame_is_24_hours_unless_a_known_one_is_chosen(self):
        self.log(self.solo_key, timedelta(days=2), 200)
        for span in ("", "all", "bogus"):
            response = self.page(self.solo, span=span)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.kpi(response, "Requests"), "1")
            self.assertContains(response, 'aria-current="page">24 hours</a>')
        self.assertEqual(self.kpi(self.page(self.solo, span="1w"), "Requests"), "2")

    def test_empty_time_frame_says_so_in_each_section(self):
        Request.objects.all().delete()
        response = self.page(self.member)
        self.assertContains(response, "No requests in this time frame.", count=3)

    def test_model_rows_open_the_log_on_their_model(self):
        response = self.page(self.solo, span="1d")
        log = f"{reverse('requests')}?span=1d&amp;model={CHAT}"
        self.assertContains(
            response, f'<a class="link-plain row-link block truncate font-mono" href="{log}"'
        )
        self.assertNotContains(response, f"{reverse('models')}?model=")

    def test_model_rows_show_the_lower_median_response(self):
        Request.objects.all().delete()
        for ms in (700, 100, None, 500, 300):
            Request.objects.create(
                token=self.solo_key,
                timestamp=timezone.now() - timedelta(hours=1),
                status_code=200,
                model=CHAT,
                response_time_ms=ms,
            )
        self.assertEqual([row.median for row in self.page(self.solo).context["models"]], ["300 ms"])

    def test_top_keys_count_tokens_and_requests_like_the_model_rows(self):
        self.log(self.pipeline_key, timedelta(hours=2), 200, input_tokens=1000)
        self.log(self.pipeline_key, timedelta(hours=3), 200, output_tokens=500)
        keys = {key.name: key for key in self.page(self.org_admin).context["keys"]}
        self.assertEqual(
            {name: key.activity.requests for name, key in keys.items()},
            {"pipeline": 3, "solo-laptop": 1, "member-laptop": 1},
        )
        pipeline = keys["pipeline"]
        self.assertEqual((pipeline.input, pipeline.output), ("4K", "1.5K"))
        self.assertEqual(pipeline.share, "61%")
        self.assertEqual(pipeline.activity.segments, (("c", 0), ("n", 4000), ("o", 1500)))
        self.assertEqual(
            keys["solo-laptop"].activity.segments, (("c", 250), ("n", 750), ("o", 500))
        )
        self.assertEqual((keys["solo-laptop"].cached, pipeline.cached), ("25%", "0%"))

    def test_totals_do_not_link_to_the_log(self):
        response = self.page(self.org_admin)
        self.assertNotContains(response, '<a class="card')
        self.assertNotContains(response, "open in the request log")


class UsageChartTest(UsagePageTestBase):
    def series(self, response) -> dict[str, str]:
        html = response.content.decode()
        found = {}
        opening = re.compile(r'<div class="chart-plot" data-metric="(\w+)">')
        matches = list(opening.finditer(html))
        for match, following in zip(matches, [*matches[1:], None], strict=True):
            end = following.start() if following else html.index("</section>", match.end())
            labels = re.findall(r'chart-col[^>]*aria-label="([^"]*)"', html[match.end() : end])
            found[match.group(1)] = " ".join(labels)
        return found

    def test_both_series_are_on_the_page_and_tokens_is_picked_first(self):
        response = self.page(self.org_admin)
        series = self.series(response)
        self.assertEqual(series.keys(), {"tokens", "requests"})
        self.assertIn("Output", series["tokens"])
        self.assertIn("OK 1, Rate-limited 1, Failed 1", series["requests"])
        self.assertContains(response, '<input type="radio" name="metric" value="tokens"', count=1)
        self.assertContains(response, 'aria-label="Tokens" checked>', count=1)

    def test_axis_labels_sit_in_a_row_of_their_own_and_cover_every_column_once(self):
        html = self.page(self.org_admin).content.decode()
        plot = html.split('<div class="chart-plot" data-metric="tokens">')[1]
        plot = plot.split('<div class="chart-plot" data-metric="requests">')[0]
        columns = plot.count("chart-col")
        row = plot.split('aria-hidden="true">', 1)[1]
        spans = [(int(a), int(b)) for a, b in re.findall(r"grid-column: (\d+) / span (\d+)", row)]
        self.assertIn(f"repeat({columns}, minmax(0, 1fr))", plot)
        self.assertEqual(spans[0][0], 1)
        self.assertEqual(sum(width for _, width in spans), columns)
        self.assertEqual(
            [start + width for start, width in spans[:-1]], [start for start, _ in spans[1:]]
        )

    def test_the_switch_needs_no_reload_and_links_do_not_carry_it(self):
        html = self.page(self.org_admin, span="1m").content.decode()
        self.assertIn('<input type="radio" name="metric" value="requests"', html)
        self.assertIsNone(re.search(r"[?;]metric=", html))


class UsageFilterChoicesTest(UsagePageTestBase):
    def test_models_sort_without_regard_to_case(self):
        with mock.patch(
            "management.request_filters.model_names", return_value=("b-model", "A-model", "c-model")
        ):
            filters = self.page(self.org_admin).context["filters"]
        self.assertEqual(filters.models, ("A-model", "b-model", "c-model"))

    def test_the_model_filter_hides_with_one_model(self):
        self.assertContains(self.page(self.org_admin), '<span class="sr-only">Model: </span>')
        with mock.patch("management.request_filters.model_names", return_value=(CHAT,)):
            response = self.page(self.org_admin)
        self.assertNotContains(response, '<span class="sr-only">Model: </span>')

    def test_an_org_admin_without_keys_gets_no_my_keys(self):
        response = self.page(self.org_admin)
        self.assertContains(response, 'aria-current="true" class="menu-active">All keys</a>')
        self.assertNotContains(response, ">My keys</a>")

    def test_an_org_admin_finds_their_own_keys_under_my_keys(self):
        own = make_token(self.org_admin, "admin-laptop")
        response = self.page(self.org_admin)
        html = response.content.decode()
        mine = html.index(">My keys</a>")
        self.assertGreater(html.index(f"key={own.id}", mine), mine)


class UsageKeyNamesTest(UsagePageTestBase):
    def details(self, user, **params) -> dict[str, str]:
        return {key.name: key.detail for key in self.page(user, **params).context["keys"]}

    def test_the_second_line_says_whose_key_it_is(self):
        own = make_token(self.org_admin, "admin-laptop")
        self.log(own, timedelta(hours=1), 200, input_tokens=5)
        self.assertEqual(
            self.details(self.org_admin),
            {
                "pipeline": "Service account · NLP Lab",
                "member-laptop": "member@example.com",
                "solo-laptop": "solo@example.com",
                "admin-laptop": "You",
            },
        )

    def test_a_viewer_of_their_own_keys_gets_no_second_line(self):
        self.assertEqual(self.details(self.member), {"pipeline": "", "member-laptop": ""})


class UsageModelColumnsTest(UsagePageTestBase):
    def test_models_without_tokens_show_dashes_and_follow_by_requests(self):
        for _ in range(2):
            self.log(self.solo_key, timedelta(hours=1), 200, model=EMBEDDING)
        Request.objects.filter(model=EMBEDDING).update(response_time_ms=None)
        rows = self.page(self.solo).context["models"]
        self.assertEqual([row.name for row in rows], [CHAT, EMBEDDING])
        unmetered = rows[1]
        self.assertEqual(
            (
                unmetered.input,
                unmetered.cached,
                unmetered.output,
                unmetered.share,
                unmetered.median,
            ),
            ("—", "—", "—", "—", "—"),
        )
        self.assertEqual((rows[0].input, rows[0].cached), ("1K", "25%"))

    def test_models_and_keys_show_their_failed_share_in_red_above_two_percent(self):
        Request.objects.all().delete()
        for token, model, failures in (
            (self.solo_key, CHAT, 2),
            (self.pipeline_key, CHAT, 3),
            (self.solo_key, EMBEDDING, 0),
        ):
            Request.objects.bulk_create(
                Request(
                    token=token,
                    timestamp=timezone.now() - timedelta(hours=1),
                    status_code=500 if n < failures else 200,
                    model=model,
                )
                for n in range(100)
            )
        response = self.page(self.org_admin)
        rows = {row.name: row for row in response.context["models"]}
        self.assertEqual((rows[CHAT].failed, rows[CHAT].activity.failing), ("3%", True))
        self.assertEqual((rows[EMBEDDING].failed, rows[EMBEDDING].activity.failing), ("0%", False))
        keys = {key.name: key for key in response.context["keys"]}
        self.assertEqual(
            (keys["solo-laptop"].failed, keys["solo-laptop"].activity.failing), ("1%", False)
        )
        self.assertEqual((keys["pipeline"].failed, keys["pipeline"].activity.failing), ("3%", True))
        self.assertContains(
            response, 'class="text-right tabular-nums st-err font-semibold">3%</td>', count=2
        )

    def test_day_columns_read_only_the_day_on_small_screens(self):
        response = self.page(self.org_admin, span="1w")
        day = timezone.localtime().strftime("%-d")
        self.assertContains(
            response,
            f'<span class="sm:hidden">{day}</span><span class="hidden sm:inline">'
            f"{timezone.localtime():%-d %b}</span>",
        )


class ApiKeysIsTheLandingPageTest(SimpleTestCase):

    def test_the_root_redirects_to_api_keys(self):
        response = self.client.get("/")
        self.assertRedirects(
            response, reverse("tokens"), status_code=301, fetch_redirect_response=False
        )
