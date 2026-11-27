from django.urls import reverse

from management.models import Request
from management.tests.helpers import RoleWorld


class AllKeysTest(RoleWorld):
    def names(self, user, **params) -> set[str]:
        response = self.as_user(user).get(reverse("all_keys"), params)
        self.assertEqual(response.status_code, 200)
        return {token.name for token in response.context["keys"]}

    def test_org_admins_see_every_key_of_their_org(self):
        expected = {"solo-laptop", "member-laptop", "pipeline"}
        self.assertEqual(self.names(self.org_admin), expected)
        self.assertEqual(self.names(self.admin), expected)
        self.assertEqual(self.names(self.other_org_admin), {"foreign-laptop"})

    def test_others_are_refused(self):
        for user in (self.solo, self.team_admin):
            with self.subTest(user=user.username):
                response = self.as_user(user).get(reverse("all_keys"))
                self.assertEqual(response.status_code, 403)

    def test_search_by_name_owner_or_team(self):
        self.assertEqual(self.names(self.org_admin, q="solo"), {"solo-laptop"})
        self.assertEqual(self.names(self.org_admin, q="member@"), {"member-laptop"})
        self.assertEqual(self.names(self.org_admin, q="nlp"), {"pipeline"})

    def test_shows_last_use_and_links_each_key(self):
        Request.objects.create(token=self.solo_key)
        page = self.as_user(self.org_admin).get(reverse("all_keys"))
        self.assertContains(page, reverse("token_detail", args=[self.pipeline_key.id]))
        self.assertContains(page, "now")
        self.assertContains(page, "Never")

    def test_the_sidebar_links_it_for_org_admins_only(self):
        link = f'href="{reverse("all_keys")}"'
        self.assertContains(self.as_user(self.org_admin).get(reverse("tokens")), link)
        self.assertNotContains(self.as_user(self.solo).get(reverse("tokens")), link)
