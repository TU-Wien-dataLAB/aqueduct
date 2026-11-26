from datetime import timedelta

from django.urls import reverse
from django.utils import timezone

from management.models import Request
from management.tests.helpers import RoleWorld


class RequestLogTest(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.solo_ok = Request.objects.create(token=cls.solo_key, path="/solo-ok", status_code=200)
        cls.solo_429 = Request.objects.create(token=cls.solo_key, path="/solo-429", status_code=429)
        cls.member_500 = Request.objects.create(
            token=cls.member_key, path="/member-500", status_code=500
        )
        cls.team_ok = Request.objects.create(
            token=cls.pipeline_key, path="/pipeline-ok", status_code=200
        )
        cls.foreign_ok = Request.objects.create(
            token=cls.foreign_key, path="/foreign-ok", status_code=200
        )

    def paths(self, user, **params) -> set[str]:
        response = self.as_user(user).get(reverse("requests"), params)
        self.assertEqual(response.status_code, 200)
        return {request.path for request in response.context["requests"]}

    def test_everyone_sees_the_requests_of_the_keys_they_may_see(self):
        self.assertEqual(self.paths(self.solo), {"/solo-ok", "/solo-429"})
        self.assertEqual(self.paths(self.member), {"/member-500", "/pipeline-ok"})
        self.assertEqual(
            self.paths(self.org_admin), {"/solo-ok", "/solo-429", "/member-500", "/pipeline-ok"}
        )
        self.assertEqual(self.paths(self.other_org_admin), {"/foreign-ok"})

    def test_a_platform_admin_sees_the_org_they_switched_to(self):
        self.assertNotIn("/foreign-ok", self.paths(self.admin))
        self.client.post(reverse("org_switch"), {"org_id": self.other_org.id})
        self.assertEqual(self.paths(self.admin), {"/foreign-ok"})

    def test_filters_by_key_and_status(self):
        self.assertEqual(self.paths(self.solo, status="rate_limited"), {"/solo-429"})
        self.assertEqual(self.paths(self.solo, status="ok"), {"/solo-ok"})
        self.assertEqual(self.paths(self.org_admin, status="failed"), {"/member-500"})
        self.assertEqual(self.paths(self.org_admin, key=self.pipeline_key.id), {"/pipeline-ok"})

    def test_another_persons_key_filters_to_nothing(self):
        self.assertEqual(self.paths(self.solo, key=self.member_key.id), set())

    def test_a_key_that_is_not_an_id_filters_to_nothing(self):
        for key in ("not-a-number", "\u00b2", "\u0663"):
            with self.subTest(key=key):
                self.assertEqual(self.paths(self.solo, key=key), set())

    def test_fifty_per_page(self):
        Request.objects.bulk_create(
            Request(token=self.solo_key, path=f"/bulk-{n}", status_code=200) for n in range(60)
        )
        page = self.as_user(self.solo).get(reverse("requests"))
        self.assertEqual(len(page.context["requests"]), 50)
        self.assertContains(page, "1\u201350 of 62")
        self.assertContains(page, "?page=2")


class RequestLogExtrasTest(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        now = timezone.now()
        for path, model, status, minutes_ago, output in (
            ("/v1/chat/completions", "gpt-4.1-nano", 200, 5, 30),
            ("/v1/chat/completions", "gpt-4.1-nano", 429, 10, 0),
            ("/v1/embeddings", "text-embedding-ada-002", 200, 90, 10),
            ("/v1/audio/speech", "gpt-4o-mini-tts", 500, 60 * 30, 20),
        ):
            Request.objects.create(
                token=cls.solo_key,
                path=path,
                model=model,
                status_code=status,
                output_tokens=output,
                timestamp=now - timedelta(minutes=minutes_ago),
                user_agent="curl/8.7",
            )
        Request.objects.create(token=cls.member_key, path="/v1/x", model="whisper-1")

    def page(self, user=None, **params):
        response = self.as_user(user or self.solo).get(reverse("requests"), params)
        self.assertEqual(response.status_code, 200)
        return response

    def paths(self, **params) -> list[str]:
        return [request.path for request in self.page(**params).context["requests"]]

    def test_status_tabs_count_the_other_filters(self):
        counts = self.page().context["counts"]
        self.assertEqual(
            counts,
            {"all": 4, "ok": 2, "rate_limited": 1, "failed": 1}
            | {"client_error": 0, "server_error": 1},
        )
        counts = self.page(model="gpt-4.1-nano", status="ok").context["counts"]
        self.assertEqual(
            counts,
            {"all": 2, "ok": 1, "rate_limited": 1, "failed": 0}
            | {"client_error": 0, "server_error": 0},
        )
        self.assertContains(self.page(status="failed"), 'aria-current="page">Failed')

    def test_filters_by_model_and_time(self):
        self.assertEqual(self.paths(model="text-embedding-ada-002"), ["/v1/embeddings"])
        self.assertEqual(len(self.paths(span="1h")), 2)
        self.assertEqual(len(self.paths(span="1d")), 3)
        self.assertEqual(len(self.paths(span="1m")), 4)
        self.assertEqual(len(self.paths(span="forever")), 4)

    def test_colours_a_row_as_its_status_tab_counts_it(self):
        Request.objects.create(token=self.solo_key, path="/v1/moved", status_code=302)
        self.assertEqual(self.paths(status="ok", q="moved"), ["/v1/moved"])
        page = self.page(q="moved")
        self.assertContains(page, '<span class="st-ok font-medium">302</span>', html=True)

    def test_filters_by_client_and_server_errors_for_the_usage_page(self):
        self.assertEqual(self.paths(status="server_error"), ["/v1/audio/speech"])
        self.assertEqual(self.paths(status="client_error"), [])

    def test_text_search_looks_at_model_and_path_only(self):
        self.assertEqual(self.paths(q="EMBED"), ["/v1/embeddings"])
        self.assertEqual(self.paths(q="speech"), ["/v1/audio/speech"])
        self.assertEqual(self.paths(q="curl"), [])

    def test_sorts_by_allowed_columns_only(self):
        by_output = ["/v1/chat/completions", "/v1/embeddings", "/v1/audio/speech"]
        self.assertEqual(self.paths(sort="output")[:3], by_output)
        self.assertEqual(self.paths(sort="-output")[1:], by_output[::-1])
        newest_first = self.paths()
        self.assertEqual(self.paths(sort="token__user__email"), newest_first)
        self.assertEqual(self.page(sort="nope").context["sort"], "-time")
        self.assertContains(self.page(sort="model"), 'aria-sort="ascending"')

    def test_filters_never_reach_other_peoples_requests(self):
        self.assertNotIn("whisper-1", self.page().context["models"])
        response = self.page(key=self.member_key.id, model="whisper-1")
        self.assertEqual(list(response.context["requests"]), [])
        self.assertEqual(response.context["counts"]["all"], 0)
        self.assertEqual(self.paths(q="/v1/x"), [])

    def test_rows_expand_to_their_details(self):
        response = self.page(q="embeddings")
        self.assertContains(response, '<details class="group">', count=1)
        self.assertContains(response, "User agent")
        self.assertContains(response, "curl/8.7")
