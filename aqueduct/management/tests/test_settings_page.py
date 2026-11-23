from django.urls import reverse

from management.tests.helpers import RoleWorld


class SettingsPageTest(RoleWorld):
    def test_shows_the_account_teams_and_keys(self):
        page = self.as_user(self.team_admin).get(reverse("settings"))
        self.assertContains(page, "team-admin@example.com")
        self.assertContains(page, "lab-org")
        self.assertContains(page, reverse("team", args=[self.team.id]))
        self.assertContains(page, "Team admin")
        self.assertContains(page, "0 of 3")

    def test_a_person_without_a_team_is_told_so(self):
        page = self.as_user(self.solo).get(reverse("settings"))
        self.assertContains(page, "You are not in a team.")
        self.assertContains(page, "1 of 3")

    def test_signed_out_visitors_go_to_sign_in(self):
        response = self.client.get(reverse("settings"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("sso"), response["Location"])

    def test_the_sidebar_links_it(self):
        page = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(page, f'href="{reverse("settings")}"')
