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

    def test_team_keys_have_no_personal_actions(self):
        page = self.as_user(self.member).get(reverse("tokens"))
        self.assertContains(page, reverse("token_edit", args=[self.member_key.id]))
        self.assertNotContains(page, reverse("token_edit", args=[self.pipeline_key.id]))


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

    def test_shows_limits_and_latest_requests(self):
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", status_code=429)
        page = self.page(self.solo, self.solo_key)
        self.assertContains(page, "This hour")
        self.assertContains(page, "gpt-4.1-nano")
        self.assertContains(page, '<span class="st-warn font-medium">429</span>', html=True)

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
