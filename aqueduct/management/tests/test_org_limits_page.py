from django.urls import reverse

from management.tests.helpers import RoleWorld


class OrgLimitsTest(RoleWorld):
    def url(self, org=None):
        return reverse("org_limits", args=[(org or self.org).id])

    def test_who_may_edit_an_orgs_limits(self):
        for user, org, status in (
            (self.org_admin, self.org, 200),
            (self.admin, self.other_org, 200),
            (self.org_admin, self.other_org, 404),
            (self.team_admin, self.org, 404),
            (self.solo, self.org, 404),
        ):
            with self.subTest(user=user.username, org=org.name):
                self.assertEqual(self.as_user(user).get(self.url(org)).status_code, status)

    def test_saving_sets_the_defaults(self):
        self.as_user(self.org_admin)
        response = self.client.post(
            self.url(), {"requests_per_minute": "120", "hourly_limit_multiplier": "30"}
        )
        self.assertRedirects(response, reverse("org"))
        self.org.refresh_from_db()
        self.assertEqual(self.org.requests_per_minute, 120)
        self.assertEqual(self.org.hourly_limit_multiplier, 30)
        self.assertIsNone(self.org.input_tokens_per_minute)

    def test_the_model_rules_apply(self):
        self.as_user(self.org_admin)
        response = self.client.post(self.url(), {"hourly_limit_multiplier": "61"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Must be between 1 and 60")

    def test_a_foreign_admin_cannot_post(self):
        self.as_user(self.other_org_admin)
        response = self.client.post(self.url(), {"requests_per_minute": "1"})
        self.assertEqual(response.status_code, 404)
        self.org.refresh_from_db()
        self.assertIsNone(self.org.requests_per_minute)

    def test_the_teams_page_links_it_for_org_admins(self):
        self.assertContains(self.as_user(self.org_admin).get(reverse("org")), self.url())
        self.assertNotContains(self.as_user(self.member).get(reverse("org")), self.url())
