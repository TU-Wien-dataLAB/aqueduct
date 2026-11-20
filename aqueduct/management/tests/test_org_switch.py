
from django.urls import reverse

from management.access import WholeOrg
from management.models import Team, UserGroup
from management.org_choice import SESSION_KEY
from management.tests.helpers import RoleWorld

ORG_SCOPED = ("usage", "usage_keys", "requests")


class OrgSwitchTestBase(RoleWorld):
    def setUp(self):
        super().setUp()
        self.far_admin = self.user("far-admin", UserGroup.ADMIN, self.other_org)

    def switch(self, org_id: object, next_url: str = ""):
        return self.client.post(reverse("org_switch"), {"org_id": org_id, "next": next_url})


class OrgSwitchTest(OrgSwitchTestBase):
    def viewed(self, url_name: str, **params):
        chosen = self.client.get(reverse(url_name), params).context["filters"].viewer.chosen
        return chosen.org if isinstance(chosen, WholeOrg) else None

    def test_all_organizations_lists_every_team_with_its_org(self):
        self.as_user(self.far_admin)
        self.switch("all")
        for url_name in ORG_SCOPED:
            with self.subTest(page=url_name):
                self.assertIsNone(self.viewed(url_name))
        teams = self.client.get(reverse("org"))
        self.assertContains(teams, "<th>Organization</th>", html=True)
        for team in ("Foreign", "NLP Lab", "Robotics"):
            row = teams.content.decode().split(f">{team}</a>", 1)[1].split("</tr>", 1)[0]
            with self.subTest(team=team):
                org = "other-org" if team == "Foreign" else "lab-org"
                self.assertIn(f'<td class="truncate max-lg:hidden">{org}</td>', row)
        self.assertContains(
            teams, '<span class="muted text-sm">All organizations</span>', html=True
        )

    def test_org_parameter_applies_and_stays_for_later_pages(self):
        self.as_user(self.far_admin)
        self.switch("all")
        self.assertEqual(self.viewed("usage", org=self.org.id), self.org)
        self.assertEqual(self.viewed("requests"), self.org)
        self.assertContains(self.client.get(reverse("org")), "NLP Lab")

    def test_unknown_org_parameter_shows_the_own_org_and_keeps_the_choice(self):
        self.as_user(self.far_admin)
        self.switch("all")
        self.assertEqual(self.viewed("usage", org="999999"), self.other_org)
        self.assertIsNone(self.viewed("usage"))

    def test_shell_shows_the_switch_to_admins_only(self):
        self.as_user(self.far_admin)
        self.switch(self.org.id)
        page = self.client.get(reverse("usage"))
        self.assertContains(page, 'id="org-switch"')
        self.assertContains(page, '<option value="all">All organizations</option>', html=True)
        self.assertContains(
            page, f'<option value="{self.org.id}" selected>lab-org</option>', html=True
        )
        self.assertNotContains(page, "Organization: ")
        for user in (self.org_admin, self.member):
            with self.subTest(user=user.username):
                for url_name in (*ORG_SCOPED, "org"):
                    page = self.as_user(user).get(reverse(url_name), follow=True)
                    self.assertNotContains(page, "org-switch")

    def test_non_admins_cannot_choose_another_org(self):
        for user in (self.org_admin, self.member):
            with self.subTest(user=user.username):
                self.as_user(user)
                response = self.switch(self.other_org.id)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(SESSION_KEY, self.client.session)
                self.assertEqual(self.viewed("usage", org=self.other_org.id), self.org)
                self.assertEqual(self.viewed("requests", org="all"), self.org)
                self.assertNotIn(SESSION_KEY, self.client.session)
                teams = self.client.get(reverse("org"), {"org": self.other_org.id})
                self.assertNotContains(teams, "Foreign")

    def test_non_admins_ignore_an_org_left_in_the_session(self):
        self.as_user(self.org_admin)
        session = self.client.session
        session[SESSION_KEY] = str(self.other_org.id)
        session.save()
        self.assertEqual(self.viewed("usage"), self.org)
        self.assertNotContains(self.client.get(reverse("org")), "Foreign")

    def test_switch_returns_without_the_old_orgs_filters(self):
        self.as_user(self.far_admin)
        back = f"{reverse('requests')}?org={self.org.id}&key=7&key_q=lap&span=1d&model=gpt-4&page=2"
        response = self.switch(self.other_org.id, back)
        self.assertRedirects(
            response, f"{reverse('requests')}?span=1d&model=gpt-4", fetch_redirect_response=False
        )

    def test_switch_ignores_a_foreign_next(self):
        self.as_user(self.far_admin)
        response = self.switch(self.org.id, "https://evil.example/")
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)

    def test_switch_to_an_unknown_org_is_a_404(self):
        self.as_user(self.far_admin)
        self.switch(self.org.id)
        self.assertEqual(self.switch("999999").status_code, 404)
        self.assertEqual(self.viewed("usage"), self.org)


class TeamCreateOrgTest(OrgSwitchTestBase):
    def create(self, org_id: object):
        return self.client.post(reverse("team_create"), {"name": "Vision", "org": org_id})

    def test_admin_on_all_organizations_sees_the_own_org_as_target(self):
        self.as_user(self.far_admin)
        self.switch("all")
        page = self.client.get(reverse("team_create"))
        self.assertContains(page, "The team is created in other-org.")
        self.assertContains(page, f'name="org" value="{self.other_org.id}"')

    def test_team_goes_to_the_org_the_form_was_opened_for(self):
        self.as_user(self.far_admin)
        self.switch(self.org.id)
        self.assertContains(self.client.get(reverse("team_create")), "created in lab-org.")
        self.switch(self.other_org.id)
        self.assertRedirects(
            self.create(self.org.id), reverse("org"), fetch_redirect_response=False
        )
        self.assertEqual(Team.objects.get(name="Vision").org, self.org)

    def test_a_create_without_an_org_field_lands_in_the_viewed_org(self):
        self.as_user(self.org_admin)
        response = self.client.post(reverse("team_create"), {"name": "Vision"})
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
        self.assertEqual(Team.objects.get(name="Vision").org, self.org)

    def test_org_admin_cannot_create_in_another_org(self):
        self.as_user(self.org_admin)
        for org_id in (self.other_org.id, "", "999999"):
            with self.subTest(org=org_id):
                self.assertEqual(self.create(org_id).status_code, 404)
        self.assertFalse(Team.objects.filter(name="Vision").exists())
