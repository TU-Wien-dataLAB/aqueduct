from datetime import timedelta

from django.urls import reverse
from django.utils import timezone

from management.models import Request, Token
from management.tests.helpers import RoleWorld


class KeyListTest(RoleWorld):
    def test_lists_personal_and_team_keys_with_last_use(self):
        Request.objects.create(
            token=self.member_key, timestamp=timezone.now() - timedelta(hours=2), path="/v1/x"
        )
        page = self.as_user(self.member).get(reverse("tokens"))
        self.assertContains(page, reverse("token_detail", args=[self.member_key.id]))
        self.assertContains(page, reverse("token_detail", args=[self.pipeline_key.id]))
        self.assertContains(page, "2\xa0hours ago")
        self.assertContains(page, "Never")
        self.assertNotContains(page, reverse("token_detail", args=[self.solo_key.id]))

    def test_last_use_is_the_newest_request_of_each_key(self):
        now = timezone.now()
        for key, hours_ago in [(self.member_key, 5), (self.member_key, 2), (self.pipeline_key, 3)]:
            Request.objects.create(token=key, timestamp=now - timedelta(hours=hours_ago))
        Request.objects.create(token=self.pipeline_key, timestamp=now - timedelta(days=9))
        page = self.as_user(self.member).get(reverse("tokens"))
        self.assertEqual(page.context["tokens"][0].last_used, now - timedelta(hours=2))
        self.assertEqual(page.context["service_accounts"][0].last_used, now - timedelta(hours=3))
        page = self.as_user(self.org_admin).get(reverse("all_keys"))
        self.assertEqual(
            {key.name: key.last_used for key in page.context["keys"]},
            {
                "member-laptop": now - timedelta(hours=2),
                "pipeline": now - timedelta(hours=3),
                "solo-laptop": None,
            },
        )

    def test_team_keys_have_no_personal_actions(self):
        page = self.as_user(self.member).get(reverse("tokens"))
        self.assertContains(page, reverse("token_edit", args=[self.member_key.id]))
        self.assertNotContains(page, reverse("token_edit", args=[self.pipeline_key.id]))

    def test_without_keys_the_page_walks_through_the_first_one(self):
        page = self.as_user(self.team_admin).get(reverse("tokens"))
        self.assertContains(page, "Create your first API key")
        self.assertContains(page, f'action="{reverse("token_create")}"')
        self.assertContains(page, 'value="my-first-key"')
        self.assertContains(page, "curl http://testserver/v1/chat/completions")
        self.assertContains(page, '"model": "gpt-4.1-nano"')
        self.assertContains(page, reverse("token_detail", args=[self.pipeline_key.id]))
        self.assertNotContains(page, "Create API key")
        self.assertNotContains(self.as_user(self.solo).get(reverse("tokens")), "first API key")

    def test_keys_show_their_requests_per_day(self):
        now = timezone.now()
        for days_ago in (0, 0, 1, 8):
            Request.objects.create(token=self.member_key, timestamp=now - timedelta(days=days_ago))
        Request.objects.create(token=self.solo_key, timestamp=now)
        page = self.as_user(self.member).get(reverse("tokens"))
        member_key, pipeline_key = page.context["tokens"][0], page.context["service_accounts"][0]
        self.assertEqual([bar["height"] for bar in member_key.week], [0, 0, 0, 0, 0, 50, 100])
        self.assertEqual(member_key.week[-1]["text"], "2 requests")
        self.assertEqual([bar["height"] for bar in pipeline_key.token.week], [0] * 7)
        today = timezone.localdate()
        self.assertContains(page, f'title="{today:%a} {today.day} {today:%b}: 2 requests"')

    def test_all_keys_show_their_requests_per_day(self):
        Request.objects.create(token=self.solo_key)
        page = self.as_user(self.org_admin).get(reverse("all_keys"))
        weeks = {key.name: [bar["text"] for bar in key.week] for key in page.context["keys"]}
        self.assertEqual(weeks["solo-laptop"], ["0 requests"] * 6 + ["1 request"])
        self.assertEqual(weeks["member-laptop"], ["0 requests"] * 7)


class KeyPageTest(RoleWorld):
    def page(self, user, token: Token):
        return self.as_user(user).get(reverse("token_detail", args=[token.id]))

    def test_who_may_open_a_key(self):
        for user, token, status in (
            (self.solo, self.solo_key, 200),
            (self.member, self.pipeline_key, 200),
            (self.org_admin, self.solo_key, 200),
            (self.admin, self.foreign_key, 200),
            (self.member, self.solo_key, 404),
            (self.solo, self.pipeline_key, 404),
            (self.other_org_admin, self.solo_key, 404),
        ):
            with self.subTest(user=user.username, key=token.name):
                self.assertEqual(self.page(user, token).status_code, status)

    def test_actions_only_for_whoever_manages_the_key(self):
        rotate = reverse("token_regenerate", args=[self.solo_key.id])
        self.assertContains(self.page(self.solo, self.solo_key), rotate)
        self.assertNotContains(self.page(self.org_admin, self.solo_key), rotate)
        self.assertNotContains(
            self.page(self.member, self.pipeline_key),
            reverse("token_regenerate", args=[self.pipeline_key.id]),
        )
        self.assertContains(
            self.page(self.team_admin, self.pipeline_key),
            reverse("service_account_edit", args=[self.pipeline.id]),
        )

    def test_shows_limits_and_links_to_its_requests(self):
        page = self.page(self.solo, self.solo_key)
        self.assertContains(page, "This hour")
        self.assertContains(page, f'href="{reverse("requests")}?key={self.solo_key.pk}"')

    def test_a_new_key_lands_on_its_page_and_shows_once(self):
        self.as_user(self.solo)
        response = self.client.post(reverse("token_create"), {"name": "fresh"}, follow=True)
        token = Token.objects.get(name="fresh")
        self.assertRedirects(response, reverse("token_detail", args=[token.id]))
        self.assertContains(response, "Copy it now")
        self.assertNotContains(response, "New key:")
        self.assertNotContains(self.page(self.solo, token), "Copy it now")

    def test_rotating_shows_the_new_key_on_the_key_page(self):
        self.as_user(self.solo)
        response = self.client.post(
            reverse("token_regenerate", args=[self.solo_key.id]), follow=True
        )
        self.assertRedirects(response, reverse("token_detail", args=[self.solo_key.id]))
        self.assertContains(response, "Copy it now")

    def test_shows_requests_per_day_and_per_model_of_this_key_only(self):
        for model, tokens in (("gpt-4.1-nano", 10), ("gpt-4.1-nano", 5), ("whisper-1", 1)):
            Request.objects.create(token=self.pipeline_key, model=model, input_tokens=tokens)
        Request.objects.create(token=self.member_key, model="dall-e-2")
        Request.objects.create(
            token=self.pipeline_key, model="dall-e-2", timestamp=timezone.now() - timedelta(days=9)
        )
        page = self.page(self.member, self.pipeline_key)
        self.assertEqual(
            [(row["model"], row["requests"], row["input"]) for row in page.context["per_model"]],
            [("gpt-4.1-nano", 2, 15), ("whisper-1", 1, 1)],
        )
        _metric, _title, _top, bars = page.context["charts"]["metrics"][1]
        self.assertEqual([bar["text"] for bar in bars][-2:], ["0 requests", "3 requests"])
        self.assertContains(page, "Models, last 7 days")
        self.assertContains(page, f"key={self.pipeline_key.pk}&amp;span=1w&amp;model=whisper-1")
