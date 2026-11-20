from datetime import UTC, datetime
from http import HTTPStatus
from unittest import mock

from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from gateway.rate_limiting import check_and_reserve, get_per_token_usage, record_token_usage
from management.models import ServiceAccount, Token, Usage
from management.tests.helpers import UsagePageTestBase, limit_grid, make_token

CLOCK = "django.utils.timezone.now"
USAGE = "management.limits.get_per_token_usage"
FROZEN = datetime(2026, 3, 4, 14, 41, 41, tzinfo=UTC)


class LimitsTestBase(UsagePageTestBase):
    def use(self, token: Token, requests: int = 0, input_tokens: int = 0, output_tokens: int = 0):
        limits = token.get_limits()
        for _ in range(requests):
            allowed, _exceeded = check_and_reserve(limits, token.id, None)
            self.assertTrue(allowed)
        record_token_usage(token.id, Usage(input_tokens=input_tokens, output_tokens=output_tokens))

    def key_page(self, user, token: Token):
        return self.as_user(user).get(reverse("token_detail", args=[token.id]))

    def set_limits(self, holder, **limits):
        for field, value in limits.items():
            setattr(holder, field, value)
        holder.save()


class KeyPanelTest(LimitsTestBase):
    def test_a_key_with_limits_of_its_own_shows_use_limit_percent_and_tone_per_window(self):
        self.set_limits(
            self.solo_key,
            requests_per_minute=10,
            input_tokens_per_minute=2000,
            output_tokens_per_minute=500,
            hourly_limit_multiplier=6,
            daily_limit_multiplier=100,
        )
        with mock.patch(CLOCK, return_value=FROZEN):
            self.use(self.solo_key, requests=6, input_tokens=1600, output_tokens=500)
            grid = limit_grid(self.key_page(self.solo, self.solo_key))

        def cells(metric):
            return [(c["used"], c["limit"], c["percent"], c["meter"]) for c in grid[metric]]

        self.assertEqual(
            cells("Requests"),
            [
                ("6", "of 10", "60", "progress-primary"),
                ("6", "of 60", "10", "progress-primary"),
                ("6", "of 1,000", "0", "progress-primary"),
            ],
        )
        self.assertEqual(
            cells("Input tokens"),
            [
                ("1.6K", "of 2K", "80", "progress-warning"),
                ("1.6K", "of 12K", "13", "progress-primary"),
                ("1.6K", "of 200K", "0", "progress-primary"),
            ],
        )
        self.assertEqual(
            cells("Output tokens"),
            [
                ("500", "of 500", "100", "progress-error"),
                ("500", "of 3K", "16", "progress-primary"),
                ("500", "of 50K", "1", "progress-primary"),
            ],
        )

    def test_a_key_without_limits_of_its_own_shows_its_owners_or_the_orgs(self):
        self.set_limits(self.org, requests_per_minute=7)
        grid = limit_grid(self.key_page(self.solo, self.solo_key))
        self.assertEqual([c["limit"] for c in grid["Requests"]], ["of 7", "of 420", "of 10,080"])
        self.set_limits(self.solo.profile, requests_per_minute=9)
        grid = limit_grid(self.key_page(self.solo, self.solo_key))
        self.assertEqual([c["limit"] for c in grid["Requests"]], ["of 9", "of 540", "of 12,960"])

    def test_the_panel_shows_the_limits_the_gateway_enforces(self):
        self.set_limits(self.org, requests_per_minute=7, input_tokens_per_minute=900)
        self.set_limits(self.team, output_tokens_per_minute=40, hourly_limit_multiplier=5)
        self.set_limits(self.pipeline_key, requests_per_minute=3)
        for token in (self.solo_key, self.pipeline_key, self.member_key):
            grid = limit_grid(self.key_page(self.org_admin, token))
            for window, (_name, _seconds, requests, input_, output) in enumerate(
                token.get_limits().windows()
            ):
                with self.subTest(key=token.name, window=window):
                    self.assertEqual(grid["Requests"][window]["limit"], f"of {requests:,}")
                    self.assertEqual(grid["Input tokens"][window]["limit"] == "", input_ is None)
                    self.assertEqual(grid["Output tokens"][window]["limit"] == "", output is None)

    def test_a_metric_without_a_limit_shows_its_use_and_no_meter(self):
        self.set_limits(self.solo_key, requests_per_minute=10)
        with mock.patch(CLOCK, return_value=FROZEN):
            self.use(self.solo_key, requests=1, input_tokens=2500)
            grid = limit_grid(self.key_page(self.solo, self.solo_key))
        self.assertEqual(
            [(c["used"], c["limit"], c["meter"]) for c in grid["Input tokens"]],
            [("2.5K", "", "")] * 3,
        )
        self.assertEqual([c["meter"] for c in grid["Requests"]], ["progress-primary"] * 3)

    def test_a_key_with_no_limits_at_all_says_so_in_one_line_instead_of_a_grid(self):
        with self.settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED=False):
            response = self.key_page(self.solo, self.solo_key)
        self.assertContains(response, "No limits apply to this key.")
        self.assertEqual(limit_grid(response), {})
        self.assertNotContains(response, "<progress")

    def test_a_key_with_only_a_concurrency_limit_says_that_one_limit(self):
        with self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=4):
            response = self.key_page(self.solo, self.solo_key)
        self.assertContains(
            response, "At most 4 requests at once. No other limit applies to this key."
        )
        self.assertEqual(limit_grid(response), {})

    def test_a_metric_with_no_limit_says_so_once(self):
        self.set_limits(self.solo_key, input_tokens_per_minute=1000)
        response = self.key_page(self.solo, self.solo_key)
        self.assertContains(response, "No limit", count=2)
        for metric in ("Requests", "Output tokens"):
            self.assertContains(
                response, f'>{metric}<div class="muted text-xs font-normal">No limit</div>'
            )

    def test_a_limit_of_zero_is_a_full_red_meter_with_nothing_used(self):
        self.set_limits(self.solo_key, requests_per_minute=0)
        requests = limit_grid(self.key_page(self.solo, self.solo_key))["Requests"]
        self.assertEqual(
            {(c["used"], c["limit"], c["percent"], c["meter"]) for c in requests},
            {("0", "of 0", "100", "progress-error")},
        )

    def test_each_window_says_when_it_resets_at_a_fixed_clock(self):
        self.set_limits(self.solo_key, requests_per_minute=10)
        with mock.patch(CLOCK, return_value=FROZEN):
            response = self.key_page(self.solo, self.solo_key)
        for metric in ("Requests", "Input tokens", "Output tokens"):
            self.assertEqual(
                [c["resets"] for c in limit_grid(response)[metric]],
                ["resets in 19 s", "resets in 19 min", "resets in 9 h 19 min"],
            )
        for time in ("14:42", "15:00", "00:00"):
            self.assertContains(response, f'title="{time} UTC">resets in', count=1)
        self.assertContains(response, ">Today (UTC)</div>")

    def test_a_window_that_just_began_resets_in_a_whole_minute_hour_and_day(self):
        self.set_limits(self.solo_key, requests_per_minute=10)
        with mock.patch(CLOCK, return_value=datetime(2026, 3, 4, 0, 0, 0, tzinfo=UTC)):
            requests = limit_grid(self.key_page(self.solo, self.solo_key))["Requests"]
        self.assertEqual(
            [c["resets"] for c in requests],
            ["resets in 1 min", "resets in 60 min", "resets in 24 h"],
        )

    def test_the_panel_names_who_sets_the_limits_and_what_a_429_means(self):
        cases = (
            (self.org, "Set for lab-org, counted per key."),
            (self.solo.profile, "Set for solo@example.com, counted per key."),
            (self.solo_key, "Set for this key."),
        )
        for holder, source in cases:
            self.set_limits(holder, requests_per_minute=10)
            with self.subTest(source=source):
                self.assertContains(
                    self.key_page(self.solo, self.solo_key),
                    f"{source} Over a limit the gateway answers 429 until the reset.",
                )

    def test_a_service_account_key_names_its_team(self):
        self.set_limits(self.team, requests_per_minute=10)
        self.assertContains(
            self.key_page(self.team_admin, self.pipeline_key), "Set for NLP Lab, counted per key."
        )

    def test_a_team_that_sets_only_a_concurrency_limit_is_not_named_for_the_orgs_limits(self):
        self.set_limits(self.org, requests_per_minute=10)
        self.set_limits(self.team, concurrent_requests_limit=2)
        page = self.key_page(self.team_admin, self.pipeline_key)
        self.assertContains(page, "Set for lab-org, counted per key.")
        self.assertNotContains(page, "Set for NLP Lab")

    def test_limits_drawn_from_two_levels_name_both(self):
        self.set_limits(self.org, input_tokens_per_minute=1000)
        self.set_limits(self.solo_key, requests_per_minute=10)
        self.assertContains(
            self.key_page(self.solo, self.solo_key),
            "Set for this key and lab-org, counted per key.",
        )

    def test_the_panel_leads_the_page(self):
        content = self.key_page(self.solo, self.solo_key).content.decode()
        self.assertLess(content.index('id="limits-title"'), content.index("Last 7 days"))

    def test_concurrent_requests_show_only_when_enforced(self):
        self.set_limits(self.solo_key, requests_per_minute=10)
        with self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=4):
            grid = limit_grid(self.key_page(self.solo, self.solo_key))
        self.assertEqual(grid["Concurrent requests"], [{"text": "At most 4 at once"}])
        with self.settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED=False):
            response = self.key_page(self.solo, self.solo_key)
        self.assertNotIn("Concurrent requests", limit_grid(response))
        self.assertNotContains(response, "at once")


class ClosestLimitTest(LimitsTestBase):
    def setUp(self):
        super().setUp()
        self.set_limits(
            self.solo_key,
            requests_per_minute=10,
            input_tokens_per_minute=2000,
            output_tokens_per_minute=500,
        )

    def test_the_api_keys_page_names_the_limit_a_key_is_closest_to(self):
        with mock.patch(CLOCK, return_value=FROZEN):
            self.use(self.solo_key, requests=6, input_tokens=1600, output_tokens=500)
            response = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(response, "Nearest limit")
        self.assertContains(response, "Output tokens, minute")
        self.assertContains(response, "100%")
        self.assertNotContains(response, "Requests, minute")

    def test_the_first_of_equal_cells_wins(self):
        with mock.patch(CLOCK, return_value=FROZEN):
            self.use(self.solo_key, requests=5, input_tokens=1000, output_tokens=250)
            response = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(response, "Requests, minute")
        self.assertContains(response, "50%")

    def test_a_key_without_a_limit_says_so(self):
        response = self.as_user(self.member).get(reverse("tokens"))
        self.assertContains(response, "No rate limit")
        self.assertNotContains(response, "<progress")


class ListPagesTest(LimitsTestBase):
    def read_usage_of(self, user, url: str) -> list[set[int]]:
        with mock.patch(USAGE, wraps=get_per_token_usage) as usage:
            self.assertEqual(self.as_user(user).get(url).status_code, HTTPStatus.OK)
        return [set(call.args[0]) for call in usage.call_args_list]

    def test_the_api_keys_page_reads_the_counters_once_for_the_keys_it_lists(self):
        second = make_token(self.team_admin, "second")
        calls = self.read_usage_of(self.team_admin, reverse("tokens"))
        self.assertEqual(calls, [{second.id, self.pipeline_key.id}])

    def test_a_page_without_keys_reads_nothing(self):
        stranger = self.user("stranger")
        self.assertEqual(self.read_usage_of(stranger, reverse("tokens")), [])

    def test_the_api_keys_page_costs_the_same_queries_for_any_number_of_keys(self):
        client = self.as_user(self.team_admin)
        make_token(self.team_admin, "first")
        with CaptureQueriesContext(connection) as small:
            client.get(reverse("tokens"))
        for n in range(3):
            make_token(self.team_admin, f"more-{n}")
            account = ServiceAccount.objects.create(name=f"account-{n}", team=self.team)
            make_token(self.team_admin, f"account-{n}", service_account=account)
        with CaptureQueriesContext(connection) as large:
            client.get(reverse("tokens"))
        self.assertEqual(len(large), len(small))

    def test_the_team_page_costs_the_same_queries_for_any_number_of_service_accounts(self):
        client = self.as_user(self.team_admin)
        url = reverse("team", args=[self.team.id])
        with CaptureQueriesContext(connection) as small:
            client.get(url)
        for n in range(3):
            account = ServiceAccount.objects.create(name=f"account-{n}", team=self.team)
            make_token(self.team_admin, f"account-{n}", service_account=account)
        with CaptureQueriesContext(connection) as large:
            client.get(url)
        self.assertEqual(len(large), len(small))


class WhoSeesWhatTest(LimitsTestBase):
    def test_a_user_gets_no_panel_for_another_users_key(self):
        self.set_limits(self.member_key, requests_per_minute=10)
        self.assertEqual(
            self.key_page(self.solo, self.member_key).status_code, HTTPStatus.NOT_FOUND
        )
        self.assertEqual(
            self.key_page(self.foreigner, self.solo_key).status_code, HTTPStatus.NOT_FOUND
        )

    def test_the_lists_read_the_counters_of_no_key_the_viewer_may_not_open(self):
        self.set_limits(self.member_key, requests_per_minute=10)
        with mock.patch(USAGE, wraps=get_per_token_usage) as usage:
            self.as_user(self.solo).get(reverse("tokens"))
        self.assertEqual([set(call.args[0]) for call in usage.call_args_list], [{self.solo_key.id}])

    def test_an_org_admin_sees_the_panel_of_a_key_in_their_org_and_not_of_another_orgs(self):
        self.assertEqual(self.key_page(self.org_admin, self.member_key).status_code, HTTPStatus.OK)
        self.assertEqual(
            self.key_page(self.org_admin, self.foreign_key).status_code, HTTPStatus.NOT_FOUND
        )
