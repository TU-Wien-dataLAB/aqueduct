from django.urls import reverse

from management.models import Team
from management.tests.helpers import RoleWorld


class TeamsListTest(RoleWorld):
    def test_org_admins_get_every_team_and_the_delete_dialog(self):
        page = self.as_user(self.org_admin).get(reverse("org"))
        for team in (self.team, self.side_team):
            self.assertContains(page, reverse("team", args=[team.id]))
        self.assertNotContains(page, reverse("team", args=[self.foreign_team.id]))
        self.assertContains(page, 'data-confirm="Delete NLP Lab?"')
        self.assertContains(page, reverse("team_create"))

    def test_members_get_their_teams_without_actions(self):
        page = self.as_user(self.member).get(reverse("org"))
        self.assertContains(page, reverse("team", args=[self.team.id]))
        self.assertNotContains(page, reverse("team", args=[self.side_team.id]))
        self.assertNotContains(page, "data-confirm=")
        self.assertNotContains(page, reverse("team_create"))

    def test_the_dialog_posts_to_the_delete_view(self):
        self.as_user(self.org_admin)
        response = self.client.post(reverse("team_delete", args=[self.side_team.id]))
        self.assertRedirects(response, reverse("org"))
        self.assertFalse(Team.objects.filter(pk=self.side_team.pk).exists())


class TeamPageTest(RoleWorld):
    def page(self, user):
        return self.as_user(user).get(reverse("team", args=[self.team.id]))

    def test_a_member_sees_service_accounts_and_members_only(self):
        page = self.page(self.member)
        self.assertContains(page, reverse("token_detail", args=[self.pipeline_key.id]))
        self.assertContains(page, "member@example.com")
        self.assertNotContains(page, reverse("service_account_edit", args=[self.pipeline.id]))
        self.assertNotContains(page, "user_profile_to_add_id")
        self.assertNotContains(page, "Make team admin")

    def test_a_team_admin_manages_service_accounts_and_members(self):
        page = self.page(self.team_admin)
        self.assertContains(page, reverse("service_account_edit", args=[self.pipeline.id]))
        self.assertContains(page, reverse("service_account_transfer", args=[self.pipeline.id]))
        self.assertContains(page, 'name="user_profile_to_add_id"')
        self.assertContains(
            page, f'<option value="{self.solo.profile.id}">solo@example.com</option>'
        )
        self.assertNotContains(page, "Make team admin")

    def test_an_org_admin_appoints_team_admins(self):
        self.assertContains(self.page(self.org_admin), "Make team admin")

    def test_adding_a_member_from_the_list(self):
        self.as_user(self.team_admin)
        self.client.post(
            reverse("team", args=[self.team.id]), {"user_profile_to_add_id": self.solo.profile.id}
        )
        self.assertTrue(self.team.member_profiles.filter(pk=self.solo.profile.pk).exists())

    def test_limits_show_the_team_and_the_org_default(self):
        self.org.requests_per_minute = 1000
        self.org.save()
        page = self.page(self.member)
        self.assertContains(page, "Organization default")
        self.assertContains(page, "1,000")


class TeamFormsTest(RoleWorld):
    def test_forms_render_in_the_form_page(self):
        self.as_user(self.team_admin)
        for url in (
            reverse("team_edit", args=[self.team.id]),
            reverse("service_account_create", args=[self.team.id]),
            reverse("service_account_edit", args=[self.pipeline.id]),
            reverse("service_account_transfer", args=[self.pipeline.id]),
        ):
            with self.subTest(url=url):
                page = self.client.get(url)
                self.assertContains(page, '<form method="post" class="card max-w-2xl bg-base-200">')
        page = self.client.get(reverse("service_account_create", args=[self.team.id]))
        self.assertContains(page, 'class="input w-full"')
        self.assertContains(page, "data-min-expiration=")
