
from django.urls import reverse

from management.access import WholeOrg
from management.models import Team, UserGroup
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


class TeamCreateOrgTest(OrgSwitchTestBase):
    def create(self, org_id: object):
        return self.client.post(reverse("team_create"), {"name": "Vision", "org": org_id})

    def test_a_create_without_an_org_field_lands_in_the_viewed_org(self):
        self.as_user(self.org_admin)
        response = self.client.post(reverse("team_create"), {"name": "Vision"})
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
        self.assertEqual(Team.objects.get(name="Vision").org, self.org)
