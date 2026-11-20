import html
import re
from datetime import UTC, datetime, timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.test import SimpleTestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from management.keys import Expiry
from management.models import Request, ServiceAccount, TeamMembership, Token
from management.tests.helpers import CHAT, RoleWorld, limit_grid, make_token, secret_on

User = get_user_model()


class ApiKeysPagesTestBase(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.other = cls.user("other")
        cls.others_key = make_token(cls.other, "others-laptop")
        hidden_sa = ServiceAccount.objects.create(name="hidden-sa", team=cls.side_team)
        make_token(cls.other, "hidden-sa", service_account=hidden_sa)

    def request_log(self, token: Token, ago: timedelta, **fields) -> Request:
        fields.setdefault("model", "gpt-4.1-nano")
        return Request.objects.create(token=token, timestamp=timezone.now() - ago, **fields)

    def request_today(self, token: Token, **fields) -> Request:
        midnight = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.request_log(token, timezone.now() - midnight, **fields)

    def get(self, user, name: str, *args, **params):
        self.client.force_login(user)
        return self.client.get(reverse(name, args=args), params)


class KeyListTest(ApiKeysPagesTestBase):
    def test_owner_sees_own_keys_linked_to_their_detail_page(self):
        response = self.get(self.solo, "tokens")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{reverse("token_detail", args=[self.solo_key.id])}"')
        self.assertContains(response, self.solo_key.key_preview)
        self.assertContains(response, "Create API key")
        self.assertContains(response, ">1/3</span>")
        self.assertNotContains(response, "others-laptop")
        self.assertNotContains(response, "pipeline")

    def test_page_never_names_another_users_key_id(self):
        response = self.get(self.solo, "tokens")
        self.assertNotContains(response, f"/tokens/{self.others_key.id}/")
        self.assertNotContains(response, f"key={self.others_key.id}")

    def test_team_admin_manages_the_teams_service_accounts_from_the_list(self):
        response = self.get(self.team_admin, "tokens")
        self.assertContains(
            response, f'href="{reverse("token_detail", args=[self.pipeline_key.id])}"'
        )
        team = reverse("team", args=[self.team.id])
        self.assertContains(
            response,
            f'Service account · <a class="link-plain relative z-10" href="{team}">NLP Lab</a>',
        )
        self.assertContains(response, reverse("token_regenerate", args=[self.pipeline_key.id]))
        self.assertContains(response, reverse("service_account_edit", args=[self.pipeline.id]))
        self.assertContains(response, reverse("service_account_delete", args=[self.pipeline.id]))
        self.assertContains(response, 'data-confirm="Delete pipeline?"')
        self.assertNotContains(response, reverse("token_edit", args=[self.pipeline_key.id]))
        self.assertNotContains(response, "hidden-sa")

    def test_plain_team_member_sees_the_service_account_without_actions(self):
        plain = self.user("plain")
        TeamMembership.objects.create(user_profile=plain.profile, team=self.team)
        response = self.get(plain, "tokens")
        self.assertContains(
            response, f'href="{reverse("token_detail", args=[self.pipeline_key.id])}"'
        )
        self.assertNotContains(response, reverse("token_regenerate", args=[self.pipeline_key.id]))
        self.assertNotContains(response, reverse("service_account_edit", args=[self.pipeline.id]))
        self.assertNotContains(response, reverse("service_account_delete", args=[self.pipeline.id]))
        self.assertNotContains(response, "Managed by the team")

    def test_rotating_from_the_list_lands_on_the_key_with_the_new_secret(self):
        detail = reverse("token_detail", args=[self.pipeline_key.id])
        self.client.force_login(self.team_admin)
        response = self.client.post(
            reverse("token_regenerate", args=[self.pipeline_key.id]), follow=True
        )
        self.assertEqual(response.redirect_chain, [(detail, 302)])
        self.assertEqual(Token.find_by_key(secret_on(response)), self.pipeline_key)

    def test_columns_count_the_last_seven_days(self):
        self.request_log(
            self.solo_key,
            timedelta(hours=1),
            input_tokens=1000,
            cached_input_tokens=250,
            output_tokens=500,
        )
        self.request_log(self.solo_key, timedelta(days=2), input_tokens=1000, output_tokens=0)
        self.request_log(self.solo_key, timedelta(days=9), input_tokens=999_000, output_tokens=0)
        response = self.get(self.solo, "tokens")
        row = response.content.decode().split(self.solo_key.key_preview, 1)[1]
        self.assertIn("1\xa0hour ago", row)
        self.assertIn("2.5K", row)
        self.assertIn('<td class="hidden text-right xl:table-cell">2</td>', row)

    def test_unused_key_reads_never_and_key_without_expiry_stays_empty(self):
        response = self.get(self.solo, "tokens")
        self.assertContains(
            response,
            '<td class="muted hidden whitespace-nowrap text-sm md:table-cell">Never</td>',
            html=True,
        )
        self.assertNotContains(response, "No expiry")

    def test_last_used_names_the_exact_utc_time(self):
        used = self.request_log(self.solo_key, timedelta(hours=2)).timestamp
        response = self.get(self.solo, "tokens")
        self.assertContains(response, f'datetime="{used.isoformat()}"')
        self.assertContains(response, f'title="{used:%Y-%m-%d %H:%M:%S} UTC"')

    def test_type_tab_and_search_filter_rows(self):
        make_token(self.team_admin, "notebook")
        response = self.get(self.team_admin, "tokens", kind="service")
        self.assertContains(response, "pipeline")
        self.assertNotContains(response, ">notebook<")
        response = self.get(self.team_admin, "tokens", q="NOTE")
        self.assertContains(response, ">notebook<")
        self.assertNotContains(response, "Service account · NLP Lab")
        response = self.get(self.team_admin, "tokens", q="nothing-like-this")
        self.assertContains(response, "No key matches.")
        self.assertNotContains(response, "No keys yet.")
        self.assertNotContains(response, "Create one")

    def test_tabs_read_all_personal_and_service_accounts(self):
        response = self.get(self.team_admin, "tokens")
        for label in ("All", "Personal", "Service accounts"):
            self.assertContains(response, f">{label}</a>")

    def test_a_key_search_that_finds_nothing_says_so(self):
        response = self.get(self.solo, "tokens", q="nothing-like-this")
        self.assertContains(response, "No key matches.")

    def test_queries_do_not_grow_with_the_keys_a_member_sees(self):
        def count() -> int:
            self.client.force_login(self.member)
            with CaptureQueriesContext(connection) as queries:
                self.assertEqual(self.client.get(reverse("tokens")).status_code, 200)
            return len(queries)

        few = count()
        for n in range(2):
            account = ServiceAccount.objects.create(name=f"sa-{n}", team=self.team)
            for key in (
                make_token(self.member, f"member-extra-{n}"),
                make_token(self.team_admin, f"sa-{n}", service_account=account),
            ):
                self.request_log(key, timedelta(minutes=n + 1))
        self.assertEqual(count(), few)

    def test_create_button_is_disabled_at_the_limit(self):
        make_token(self.solo, "two")
        make_token(self.solo, "three")
        response = self.get(self.solo, "tokens")
        self.assertNotContains(response, f'href="{reverse("token_create")}"')
        self.assertContains(
            response,
            '<button type="button" class="btn btn-primary" disabled>3 of 3 keys used</button>',
            html=True,
        )


class FirstVisitTest(ApiKeysPagesTestBase):
    def test_a_person_with_no_key_is_asked_to_create_one_with_a_working_example(self):
        response = self.get(self.user("newcomer"), "tokens")
        page = response.content.decode()
        self.assertContains(response, "Create your first API key")
        self.assertIn(f'action="{reverse("token_create")}"', page)
        self.assertIn('value="my-first-key"', page)
        self.assertIn("export OPENAI_BASE_URL=http://testserver/v1", page)
        self.assertIn("curl $OPENAI_BASE_URL/chat/completions", page)
        self.assertIn("&quot;model&quot;: &quot;main&quot;", page)
        self.assertContains(
            response,
            'It calls <span class="font-mono">main</span>, the alias for '
            f'<span class="font-mono">{CHAT}</span>',
        )
        self.assertIn("Your call then appears under Requests.", page)

    def test_the_block_stands_alone_and_replaces_the_create_button(self):
        response = self.get(self.user("newcomer"), "tokens")
        for gone in ("Create API key", "Search keys", "<table", "No key"):
            self.assertNotContains(response, gone)

    def test_the_first_key_is_made_by_the_create_view_and_then_listed(self):
        newcomer = self.user("newcomer")
        self.get(newcomer, "tokens")
        created = self.client.post(
            reverse("token_create"), {"name": "my-first-key", "expires_at": ""}
        )
        self.assertEqual(created.status_code, 302)
        response = self.get(newcomer, "tokens")
        self.assertContains(response, ">my-first-key</a>")
        self.assertContains(response, "Create API key")
        self.assertNotContains(response, "Create your first API key")

    def test_a_team_admin_with_no_personal_key_gets_the_block_above_the_team_keys(self):
        response = self.get(self.team_admin, "tokens")
        self.assertContains(response, "Create your first API key")
        self.assertContains(response, ">pipeline</a>")
        self.assertContains(response, "Search keys")
        self.assertNotContains(response, "Create API key")

    def test_a_search_or_filter_leaves_the_block_out(self):
        for params in ({"q": "pipe"}, {"kind": "service"}):
            with self.subTest(**params):
                response = self.get(self.team_admin, "tokens", **params)
                self.assertNotContains(response, "Create your first API key")
                self.assertContains(response, "Create API key")

    def test_a_platform_admin_viewing_all_orgs_gets_the_block_and_no_foreign_key(self):
        self.client.force_login(self.admin)
        self.client.get(reverse("usage"), {"org": "all"})
        response = self.client.get(reverse("tokens"))
        self.assertContains(response, "Create your first API key")
        self.assertNotContains(response, "foreign-laptop")


class KeyDetailTest(ApiKeysPagesTestBase):
    def test_owner_sees_totals_models_and_latest_requests(self):
        self.request_log(
            self.solo_key,
            timedelta(minutes=5),
            model="gpt-4.1-nano",
            input_tokens=3000,
            output_tokens=1000,
            status_code=200,
            path="/chat/completions",
            user_agent="curl/8.5.0",
        )
        self.request_log(
            self.solo_key,
            timedelta(minutes=6),
            model="gpt-4.1-nano",
            status_code=504,
            path="/responses",
            user_agent="curl/8.5.0",
        )
        response = self.get(self.solo, "token_detail", self.solo_key.id)
        self.assertEqual(response.status_code, 200)
        page = response.content.decode()
        self.assertIn("3K", page)
        self.assertIn("50%", page)
        self.assertIn("curl/8.5.0 100%", page)
        self.assertIn("/responses", page)
        self.assertIn("Last 7 days", page)
        self.assertNotIn(f"{reverse('usage')}?key={self.solo_key.id}", page)

    def test_the_figures_are_the_four_tiles_of_the_usage_page(self):
        self.request_log(
            self.solo_key, timedelta(hours=1), input_tokens=2000, cached_input_tokens=500
        )
        self.request_log(self.solo_key, timedelta(hours=2), status_code=429)
        self.request_log(self.solo_key, timedelta(hours=3), status_code=500)
        page = self.get(self.solo, "token_detail", self.solo_key.id).content.decode()
        tiles = re.findall(
            r'<span class="muted text-sm">([^<]*)</span>\s*<span class="text-2xl font-semibold">'
            r'([^<]*)</span>\s*(?:<span class="muted text-sm">([^<]*)</span>)?',
            page.split('id="totals-title"')[1],
        )
        self.assertEqual(
            tiles,
            [
                ("Requests", "3", ""),
                ("Input tokens", "2K", "25% from cache"),
                ("Output tokens", "0", ""),
                ("Failed", "33%", "1 rate-limited"),
            ],
        )
        for gone in ("Cache hits", "Cached"):
            self.assertNotIn(gone, page)

    def test_latest_requests_match_the_request_log(self):
        limited = self.request_log(
            self.solo_key,
            timedelta(minutes=5),
            model="gpt-4.1-nano",
            status_code=429,
            response_time_ms=5640,
        )
        response = self.get(self.solo, "token_detail", self.solo_key.id)
        self.assertContains(
            response, '<span class="st-warn inline-flex items-center gap-1.5" title="Rate-limited">'
        )
        self.assertContains(
            response, '<td class="whitespace-nowrap text-right max-md:hidden">5.6 s</td>', html=True
        )
        log = reverse("requests")
        self.assertContains(
            response,
            f'href="{log}?key={self.solo_key.id}&amp;span=1m&amp;request={limited.id}#request-{limited.id}"',
        )
        self.assertContains(
            response,
            f'<a class="link-plain text-sm" href="{log}?key={self.solo_key.id}&amp;span=1m">'
            "Requests</a>",
            html=True,
        )
        opened = self.client.get(f"{log}?key={self.solo_key.id}&request={limited.id}")
        self.assertContains(opened, f'aria-label="Details of request {limited.id}"')

    def test_latest_requests_are_those_the_log_opens_on(self):
        shown = self.request_log(self.solo_key, timedelta(days=29))
        self.request_log(self.solo_key, timedelta(days=31))
        detail = self.get(self.solo, "token_detail", self.solo_key.id)
        self.assertEqual([row.request for row in detail.context["latest"]], [shown])
        log = self.get(self.solo, "requests", key=self.solo_key.id, span="1m").context["log"]
        self.assertEqual([row.request for row in log.rows], [shown])
        shown.delete()
        self.assertContains(
            self.get(self.solo, "token_detail", self.solo_key.id),
            "No requests in the last 30 days.",
        )

    def test_requests_metric_charts_failed_requests(self):
        self.request_log(self.solo_key, timedelta(minutes=5), status_code=500)
        response = self.get(self.solo, "token_detail", self.solo_key.id)
        self.assertContains(response, '<span data-metric="requests">Requests per day</span>')
        self.assertContains(response, '<i class="f" style="height: 100%"></i>', html=True)

    def test_chart_holds_both_series_so_the_switch_needs_no_reload(self):
        self.request_today(self.solo_key, status_code=500, output_tokens=7)
        today = f"{timezone.localdate():%-d %b}"
        html = self.get(self.solo, "token_detail", self.solo_key.id).content.decode()
        for shown in ("tokens", "requests"):
            self.assertIn(f'<div class="chart-plot" data-metric="{shown}">', html)
            self.assertIn(f'<input type="radio" name="metric" value="{shown}"', html)
        self.assertIn(f'aria-label="{today}: 1 request (OK 0, Rate-limited 0, Failed 1)"', html)
        self.assertIn(f'aria-label="{today}: 7 tokens (', html)
        self.assertIn('aria-label="Tokens" checked>', html)

    def test_chart_column_names_its_day_and_counts(self):
        self.request_today(self.solo_key, status_code=500)
        response = self.get(self.solo, "token_detail", self.solo_key.id)
        today = f"{timezone.localdate():%-d %b}"
        self.assertContains(
            response, f'aria-label="{today}: 1 request (OK 0, Rate-limited 0, Failed 1)"'
        )

    def test_dates_name_the_full_utc_time_and_the_tab_names_the_key(self):
        used = self.request_log(self.solo_key, timedelta(minutes=5)).timestamp
        self.solo_key.expires_at = timezone.now() + timedelta(days=30)
        self.solo_key.save()
        response = self.get(self.solo, "token_detail", self.solo_key.id)
        self.assertContains(response, "<title>solo-laptop · API keys · Aqueduct AI Gateway</title>")
        self.assertContains(
            response, f'<time datetime="{used.isoformat()}" title="{used:%Y-%m-%d %H:%M:%S} UTC">'
        )
        self.assertContains(response, f">{used:%-d %b %H:%M}</time></a>")
        expires = self.solo_key.expires_at
        self.assertContains(
            response,
            f'<time datetime="{expires.isoformat()}" title="{expires:%Y-%m-%d %H:%M:%S} UTC">'
            f"Expires {expires:%-d %b %Y}</time>",
        )

    def test_unknown_key_is_not_found(self):
        response = self.get(self.solo, "token_detail", 999_999)
        self.assertEqual(response.status_code, 404)


class SevenDaysTest(ApiKeysPagesTestBase):
    def test_key_pages_count_the_same_seven_days_as_the_overview(self):
        now = datetime(2026, 10, 14, 8, 0, tzinfo=UTC)
        for ago in (timedelta(days=6, hours=12), timedelta(days=6, hours=7), timedelta(hours=1)):
            Request.objects.create(
                token=self.solo_key, timestamp=now - ago, model="gpt-4.1-nano", status_code=200
            )
        with mock.patch("django.utils.timezone.now", return_value=now):
            listed = self.get(self.solo, "tokens").context["keys"].rows[0].activity.requests
            detail = self.get(self.solo, "token_detail", self.solo_key.id).context["report"]
            overview = self.get(self.solo, "usage", key=self.solo_key.id, span="1w").context["kpis"]
        requests = next(kpi.value for kpi in overview if kpi.label == "Requests")
        self.assertEqual((listed, detail.totals.requests, requests), (2, 2, "2"))
        self.assertEqual(sum(day.activity.requests for day in detail.days), 2)

    def test_key_pages_and_the_log_show_the_same_requests_even_for_unserved_models(self):
        served = self.request_log(self.solo_key, timedelta(minutes=3), status_code=200)
        retired = self.request_log(self.solo_key, timedelta(minutes=2), model="retired-model")
        no_model = self.request_log(self.solo_key, timedelta(minutes=1), model="", path="/files")
        everything = [no_model, retired, served]
        listed = self.get(self.solo, "tokens").context["keys"].rows[0].activity
        detail = self.get(self.solo, "token_detail", self.solo_key.id).context
        overview = self.get(self.solo, "usage", key=self.solo_key.id, span="1w").context["kpis"]
        log = self.get(self.solo, "requests", key=self.solo_key.id).context["log"]
        requests = next(kpi.value for kpi in overview if kpi.label == "Requests")
        self.assertEqual((listed.requests, detail["report"].totals.requests, requests), (3, 3, "3"))
        self.assertEqual(sum(day.activity.requests for day in detail["report"].days), 3)
        self.assertEqual(
            (listed.last_used, detail["report"].totals.last_used),
            (no_model.timestamp, no_model.timestamp),
        )
        self.assertEqual([row.request for row in detail["latest"]], everything)
        self.assertEqual([row.request for row in log.rows], everything)

    def test_last_used_reaches_back_past_the_seven_days(self):
        old = self.request_log(self.pipeline_key, timedelta(days=20), model="", path="/files")
        listed = self.get(self.team_admin, "tokens").context["keys"].rows[0].activity
        detail = self.get(self.team_admin, "token_detail", self.pipeline_key.id).context["report"]
        self.assertEqual((listed.last_used, detail.totals.last_used), (old.timestamp,) * 2)
        self.assertEqual((listed.requests, detail.totals.requests), (0, 0))


class SecretShownOnceTest(ApiKeysPagesTestBase):

    def test_the_secret_shows_only_on_its_own_keys_page_and_is_not_cached(self):
        other = make_token(self.solo, "older")
        self.client.force_login(self.solo)
        self.client.post(reverse("token_create"), {"name": "fresh", "expires_at": ""})
        fresh = Token.objects.get(name="fresh")

        elsewhere = self.client.get(reverse("token_detail", args=[other.id]))
        self.assertNotContains(elsewhere, "is ready")
        self.assertNotIn("no-store", elsewhere.headers.get("Cache-Control", ""))

        self.client.post(reverse("token_regenerate", args=[fresh.id]))
        shown = self.client.get(reverse("token_detail", args=[fresh.id]))
        self.assertContains(shown, "is ready")
        self.assertIn("no-store", shown.headers["Cache-Control"])

    def test_the_panel_sits_under_the_header_and_calls_a_chat_model_by_alias(self):
        self.client.force_login(self.solo)
        response = self.client.post(
            reverse("token_create"), {"name": "fresh", "expires_at": ""}, follow=True
        )
        page = response.content.decode()
        self.assertLess(page.index("<h1"), page.index("is ready"))
        secret = secret_on(response)
        self.assertContains(response, "export OPENAI_BASE_URL=http://testserver/v1")
        self.assertContains(response, f"export OPENAI_API_KEY={secret}")
        self.assertContains(response, "curl http://testserver/v1/chat/completions")
        self.assertContains(response, "&quot;model&quot;: &quot;main&quot;")
        self.assertNotContains(response, "/models -H")

    def test_the_example_uses_a_model_the_key_may_call(self):
        self.solo.profile.excluded_models = ["gpt-4.1-nano"]
        self.solo.profile.save()
        self.client.force_login(self.solo)
        response = self.client.post(
            reverse("token_create"), {"name": "fresh", "expires_at": ""}, follow=True
        )
        self.assertContains(response, "&quot;model&quot;: &quot;decision&quot;")

    def test_a_key_that_may_call_no_chat_model_gets_the_model_list_call(self):
        self.solo.profile.excluded_models = ["gpt-4.1-nano", "gpt-6-luna"]
        self.solo.profile.save()
        self.client.force_login(self.solo)
        response = self.client.post(
            reverse("token_create"), {"name": "fresh", "expires_at": ""}, follow=True
        )
        self.assertContains(
            response,
            "curl $OPENAI_BASE_URL/models -H &quot;Authorization: Bearer $OPENAI_API_KEY&quot;",
        )
        self.assertNotContains(response, "chat/completions")

    def test_create_at_the_limit_rerenders_the_form_with_an_error(self):
        make_token(self.solo, "two")
        make_token(self.solo, "three")
        self.client.force_login(self.solo)
        response = self.client.post(reverse("token_create"), {"name": "fourth", "expires_at": ""})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You can only have 3 API keys.")
        self.assertFalse(Token.objects.filter(name="fourth").exists())


def dialog(page, title: str) -> dict[str, str]:
    """The `data-confirm-*` text a button opens the confirm dialog with."""
    tag = re.search(
        rf'<[^>]*data-confirm="{re.escape(html.escape(title))}"[^>]*>', page.content.decode()
    )
    assert tag, f"no button opens {title!r}"
    return {
        name: html.unescape(value)
        for name, value in re.findall(r'data-confirm-(\w+)="([^"]*)"', tag[0])
    }


class DialogUseTest(ApiKeysPagesTestBase):
    ROTATE = "The current key stops working immediately."
    FAILS = "Whatever uses it fails until it gets the new key."

    def use_pipeline(self, ago: timedelta, times: int = 3) -> None:
        for _ in range(times):
            self.request_log(self.pipeline_key, ago)

    def pages(self):
        return (
            (self.get(self.team_admin, "tokens"), "Rotate pipeline?", "Delete pipeline?"),
            (
                self.get(self.team_admin, "team", self.team.id),
                "Rotate the key of pipeline?",
                "Delete pipeline?",
            ),
            (
                self.get(self.team_admin, "token_detail", self.pipeline_key.id),
                "Rotate pipeline?",
                "Delete pipeline?",
            ),
        )

    def test_use_older_than_a_week_still_names_when_it_was_last_used(self):
        self.request_log(self.pipeline_key, timedelta(days=20))
        page = self.get(self.team_admin, "tokens")
        self.assertEqual(
            dialog(page, "Rotate pipeline?")["use"],
            "Last used 3\xa0weeks ago · 0 requests in the last 7 days.",
        )

    def test_a_personal_key_delete_dialog_names_its_use_above_what_is_deleted(self):
        self.request_log(self.solo_key, timedelta(minutes=5))
        page = self.get(self.solo, "tokens")
        shown = dialog(page, "Delete solo-laptop?")
        self.assertEqual(shown["use"], "Last used 5\xa0minutes ago · 1 request in the last 7 days.")
        self.assertIn(
            "its requests, files, batches, and vector stores are deleted", shown["detail"]
        )

    def test_the_numbers_come_without_a_query_per_row(self):
        def queries(name: str, *args) -> int:
            with CaptureQueriesContext(connection) as captured:
                self.get(self.team_admin, name, *args)
            return len(captured)

        queries("tokens"), queries("team", self.team.id)
        before = (queries("tokens"), queries("team", self.team.id))
        for index in range(4):
            sa = ServiceAccount.objects.create(name=f"extra-{index}", team=self.team)
            self.request_log(
                make_token(self.team_admin, f"extra-{index}", service_account=sa),
                timedelta(hours=1),
            )
        self.assertEqual((queries("tokens"), queries("team", self.team.id)), before)

    def test_a_plain_member_gets_no_dialog_and_no_numbers(self):
        plain = self.user("plain")
        TeamMembership.objects.create(user_profile=plain.profile, team=self.team)
        self.use_pipeline(timedelta(hours=2))
        page = self.get(plain, "token_detail", self.pipeline_key.id)
        self.assertNotContains(page, "data-confirm=")
        self.assertNotContains(page, "data-confirm-use")


class RotateReturnTest(ApiKeysPagesTestBase):
    def test_rotating_lands_where_the_secret_is_shown_whatever_next_says(self):
        self.client.force_login(self.solo)
        detail = reverse("token_detail", args=[self.solo_key.id])
        for target in (None, reverse("tokens"), "https://evil.example/"):
            with self.subTest(target=target):
                response = self.client.post(
                    reverse("token_regenerate", args=[self.solo_key.id]),
                    {"next": target} if target else {},
                )
                self.assertRedirects(response, detail, fetch_redirect_response=False)


class KeyEditAndDeleteReturnTest(ApiKeysPagesTestBase):

    def test_delete_returns_to_the_list_with_a_message(self):
        self.client.force_login(self.solo)
        response = self.client.post(reverse("token_delete", args=[self.solo_key.id]), follow=True)
        self.assertRedirects(response, reverse("tokens"))
        self.assertContains(response, "API key solo-laptop deleted.")
        self.assertFalse(Token.objects.filter(id=self.solo_key.id).exists())


class ExpiryTest(SimpleTestCase):
    def test_labels_and_sentences(self):
        now = datetime(2026, 10, 18, 20, tzinfo=UTC)
        later = now + timedelta(days=30)
        day = f"{timezone.localtime(later):%-d %b %Y}"
        warn, err = "st-warn font-semibold", "st-err font-semibold"
        cases = [
            (None, "No expiry", "Never expires", "muted"),
            (now - timedelta(minutes=1), "Expired", "Expired", err),
            (now + timedelta(hours=3), "Today", "Expires today", warn),
            (now + timedelta(hours=14), "In 1 day", "Expires in 1 day", warn),
            (now + timedelta(days=4, hours=14), "In 5 days", "Expires in 5 days", warn),
            (now + timedelta(days=1, hours=1), "In 1 day", "Expires in 1 day", warn),
            (now + timedelta(days=5, hours=1), "In 5 days", "Expires in 5 days", warn),
            (later, day, f"Expires {day}", "muted"),
        ]
        for expires_at, label, sentence, tone in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    Expiry.of(expires_at, now), Expiry(label, sentence, tone, expires_at)
                )


class OrgAdminKeyAccessTest(ApiKeysPagesTestBase):
    def setUp(self):
        super().setUp()
        self.request_log(self.solo_key, timedelta(hours=1), input_tokens=4200)

    def test_org_admin_sees_a_members_key_without_actions(self):
        response = self.get(self.org_admin, "token_detail", self.solo_key.id)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "4.2K")
        self.assertContains(response, "Personal · solo@example.com")
        self.assertContains(response, f"{reverse('requests')}?key={self.solo_key.id}")
        for action in ("token_regenerate", "token_edit", "token_delete"):
            self.assertNotContains(response, reverse(action, args=[self.solo_key.id]))

    def test_admin_opens_another_orgs_key_with_a_log_link_into_its_org(self):
        self.request_log(self.foreign_key, timedelta(hours=1))
        self.client.force_login(self.admin)
        response = self.client.get(reverse("token_detail", args=[self.foreign_key.id]))
        self.assertEqual(response.status_code, 200)
        query = f"key={self.foreign_key.id}&amp;span=1m&amp;org={self.other_org.id}"
        link = f"{reverse('requests')}?{query}"
        self.assertContains(response, f'href="{link}"')
        self.assertEqual(
            self.client.get(reverse("requests"), {"key": self.foreign_key.id}).status_code, 404
        )
        self.assertEqual(self.client.get(html.unescape(link)).status_code, 200)

    def test_admin_follows_a_key_into_the_log_whatever_org_the_switch_shows(self):
        self.request_log(self.pipeline_key, timedelta(hours=1))
        self.client.force_login(self.admin)
        self.client.get(reverse("tokens"), {"org": self.other_org.id})
        log = re.escape(reverse("requests"))
        for key in (self.solo_key, self.pipeline_key):
            page = self.client.get(reverse("token_detail", args=[key.id])).content.decode()
            links = re.findall(rf'href="({log}\?[^"#]*)', page)
            self.assertEqual(len(links), 2)
            for link in links:
                with self.subTest(key=key.name, link=link):
                    response = self.client.get(html.unescape(link))
                    shown = Request.objects.get(token=key)
                    self.assertContains(response, f'id="request-{shown.id}"')

    def test_admin_on_all_organizations_stays_there_after_a_key_log_link(self):
        self.client.force_login(self.admin)
        self.client.get(reverse("tokens"), {"org": "all"})
        page = self.client.get(reverse("token_detail", args=[self.solo_key.id])).content.decode()
        self.assertNotIn("org=", "".join(re.findall(r'href="[^"]*/requests/\?[^"]*"', page)))

    def test_admin_rotating_another_orgs_service_account_lands_on_its_key(self):
        detail = reverse("token_detail", args=[self.pipeline_key.id])
        self.client.force_login(self.admin)
        self.client.get(reverse("tokens"), {"org": self.other_org.id})
        response = self.client.post(
            reverse("token_regenerate", args=[self.pipeline_key.id]), {"next": detail}, follow=True
        )
        self.assertEqual(response.redirect_chain, [(detail, 302)])
        self.assertEqual(Token.find_by_key(secret_on(response)), self.pipeline_key)

    def test_org_admin_cannot_change_a_members_key(self):
        self.client.force_login(self.org_admin)
        before = Token.objects.values().get(id=self.solo_key.id)
        for action, data in (
            ("token_regenerate", {}),
            ("token_edit", {"name": "taken-over", "expires_at": ""}),
            ("token_delete", {}),
        ):
            with self.subTest(action=action):
                response = self.client.post(reverse(action, args=[self.solo_key.id]), data)
                self.assertEqual(response.status_code, 404)
        self.assertEqual(Token.objects.values().get(id=self.solo_key.id), before)


class KeyLinksTest(ApiKeysPagesTestBase):
    PAGES = ("requests", "usage", "usage_keys")

    def setUp(self):
        super().setUp()
        self.hidden_key = Token.objects.get(name="hidden-sa")
        self.members_key = make_token(self.team_admin, "members-laptop")
        for token in (
            self.solo_key,
            self.others_key,
            self.pipeline_key,
            self.hidden_key,
            self.members_key,
        ):
            self.request_log(token, timedelta(hours=1), input_tokens=100)

    def detail(self, token: Token) -> str:
        return f'href="{reverse("token_detail", args=[token.id])}"'

    def test_org_admin_sees_every_members_key_linked(self):
        for page in self.PAGES:
            response = self.get(self.org_admin, page)
            for token in (self.solo_key, self.others_key, self.pipeline_key, self.hidden_key):
                with self.subTest(page=page, key=token.name):
                    self.assertContains(response, self.detail(token))

    def test_team_member_sees_only_their_own_and_their_teams_keys_linked(self):
        for page in ("requests", "usage"):
            response = self.get(self.team_admin, page)
            with self.subTest(page=page):
                self.assertContains(response, self.detail(self.members_key))
                self.assertContains(response, self.detail(self.pipeline_key))
                for token in (self.solo_key, self.others_key, self.hidden_key):
                    self.assertNotContains(response, self.detail(token))

    def test_queries_do_not_grow_with_the_keys_listed(self):
        def queries() -> dict[str, int]:
            self.client.force_login(self.org_admin)
            counts = {}
            for page in self.PAGES:
                cache.clear()
                with CaptureQueriesContext(connection) as captured:
                    self.assertEqual(self.client.get(reverse(page)).status_code, 200)
                counts[page] = len(captured)
            return counts

        before = queries()
        for n in range(5):
            colleague = self.user(f"colleague-{n}")
            self.request_log(make_token(colleague, f"colleague-{n}"), timedelta(hours=1))
            account = ServiceAccount.objects.create(name=f"bot-{n}", team=self.side_team)
            bot = make_token(colleague, f"bot-{n}", service_account=account)
            self.request_log(bot, timedelta(hours=1))
        self.assertEqual(queries(), before)


class ServiceAccountKeyDetailTest(ApiKeysPagesTestBase):
    def setUp(self):
        super().setUp()
        self.plain_member = self.user("plain-member")
        TeamMembership.objects.create(user_profile=self.plain_member.profile, team=self.team)
        self.request_log(self.pipeline_key, timedelta(hours=1), input_tokens=4200)

    def detail(self, user):
        return self.get(user, "token_detail", self.pipeline_key.id)

    def test_team_admin_and_org_admin_see_usage_and_can_manage_the_account(self):
        edit = reverse("service_account_edit", args=[self.pipeline.id])
        for user in (self.team_admin, self.org_admin):
            with self.subTest(user=user.username):
                response = self.detail(user)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Service account · ")
                self.assertContains(response, "4.2K")
                self.assertContains(response, f"?key={self.pipeline_key.id}")
                self.assertContains(response, f'href="{edit}"')
                self.assertContains(
                    response, reverse("token_regenerate", args=[self.pipeline_key.id])
                )
                self.assertNotContains(response, reverse("token_edit", args=[self.pipeline_key.id]))

    def test_plain_team_member_sees_usage_without_management_actions(self):
        response = self.detail(self.plain_member)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "4.2K")
        self.assertNotContains(response, reverse("token_regenerate", args=[self.pipeline_key.id]))
        self.assertNotContains(response, reverse("service_account_edit", args=[self.pipeline.id]))
        self.assertNotContains(response, reverse("service_account_delete", args=[self.pipeline.id]))

    def test_limits_are_the_teams_not_the_viewers(self):
        self.team.requests_per_minute = 7
        self.team.save()
        profile = self.team_admin.profile
        profile.requests_per_minute = 999
        profile.save()
        grid = limit_grid(self.detail(self.team_admin))
        self.assertEqual(
            [cell["limit"] for cell in grid["Requests"]], ["of 7", "of 420", "of 10,080"]
        )

    def test_a_keys_own_limits_win_over_its_teams(self):
        self.team.requests_per_minute = 7
        self.team.input_tokens_per_minute = 5000
        self.team.save()
        self.pipeline_key.requests_per_minute = 2
        self.pipeline_key.save()
        grid = limit_grid(self.detail(self.team_admin))
        self.assertEqual(
            [cell["limit"] for cell in grid["Requests"]], ["of 2", "of 120", "of 2,880"]
        )
        self.assertEqual(
            [cell["limit"] for cell in grid["Input tokens"]], ["of 5K", "of 300K", "of 7.2M"]
        )

    def test_a_request_limit_of_zero_shows_every_window_as_full(self):
        self.pipeline_key.requests_per_minute = 0
        self.pipeline_key.save()
        requests = limit_grid(self.detail(self.team_admin))["Requests"]
        self.assertEqual({cell["meter"] for cell in requests}, {"progress-error"})
        self.assertEqual({cell["percent"] for cell in requests}, {"100"})
        self.assertEqual({cell["limit"] for cell in requests}, {"of 0"})

    def test_concurrent_requests_come_from_the_key_then_its_team_then_the_org(self):
        concurrent = lambda limit: self.assertContains(
            self.detail(self.team_admin), f"At most {limit} requests at once."
        )
        with self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=3):
            concurrent(3)
        for holder, limit in ((self.org, 40), (self.team, 20), (self.pipeline_key, 10)):
            holder.concurrent_requests_limit = limit
            holder.save()
            with self.subTest(holder=holder):
                concurrent(limit)

    def test_plain_team_member_cannot_rotate_the_key(self):
        old_hash = self.pipeline_key.key_hash
        self.client.force_login(self.plain_member)
        response = self.client.post(reverse("token_regenerate", args=[self.pipeline_key.id]))
        self.assertEqual(response.status_code, 404)
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.key_hash, old_hash)

    def test_rotating_from_the_detail_page_returns_there_with_the_new_secret(self):
        detail = reverse("token_detail", args=[self.pipeline_key.id])
        self.client.force_login(self.team_admin)
        response = self.client.post(
            reverse("token_regenerate", args=[self.pipeline_key.id]), {"next": detail}, follow=True
        )
        self.assertEqual(response.redirect_chain, [(detail, 302)])
        self.assertEqual(Token.find_by_key(secret_on(response)), self.pipeline_key)

    def test_rotating_from_the_team_page_lands_on_the_key_not_the_team(self):
        self.client.force_login(self.team_admin)
        response = self.client.post(reverse("token_regenerate", args=[self.pipeline_key.id]))
        self.assertRedirects(
            response,
            reverse("token_detail", args=[self.pipeline_key.id]),
            fetch_redirect_response=False,
        )
