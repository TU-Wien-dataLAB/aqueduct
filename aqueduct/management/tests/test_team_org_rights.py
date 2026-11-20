from typing import ClassVar

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase
from django.urls import reverse

from management.models import Org, Team, TeamMembership, UserProfile

User = get_user_model()


class TeamOrgRightsTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.org_admin = User.objects.get(username="Someone")
        self.global_admin = User.objects.get(username="Me")
        self.other_team = Team.objects.create(
            name="other-team", org=Org.objects.create(name="other-org")
        )
        bob = User.objects.create_user(username="bob", email="bob@example.com")
        bob.groups.add(Group.objects.get(name="user"))
        self.bob_profile = UserProfile.objects.create(user=bob, org=self.other_team.org)
        self.membership = TeamMembership.objects.create(
            user_profile=self.bob_profile, team=self.other_team
        )

    def test_org_admin_cannot_delete_team_of_another_org(self):
        self.client.force_login(self.org_admin)
        self.client.post(reverse("team_delete", kwargs={"id": self.other_team.id}))
        self.assertTrue(Team.objects.filter(pk=self.other_team.pk).exists())

    def test_org_admin_cannot_make_team_admin_in_another_org(self):
        self.client.force_login(self.org_admin)
        self.client.post(
            reverse("team_members", kwargs={"id": self.other_team.id}),
            {"profile_id": self.bob_profile.id, "action": "grant_admin"},
        )
        self.membership.refresh_from_db()
        self.assertFalse(self.membership.is_admin)

    def test_org_admin_can_delete_team_of_own_org(self):
        self.client.force_login(self.org_admin)
        team = Team.objects.get(name="Whale")
        self.client.post(reverse("team_delete", kwargs={"id": team.id}))
        self.assertFalse(Team.objects.filter(pk=team.pk).exists())

    def test_global_admin_can_still_delete_team_of_another_org(self):
        self.client.force_login(self.global_admin)
        self.client.post(reverse("team_delete", kwargs={"id": self.other_team.id}))
        self.assertFalse(Team.objects.filter(pk=self.other_team.pk).exists())
