"""
Tests for OAuth team name sync admin action.
"""

from django.contrib.admin import site
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.cookie import CookieStorage
from django.test import RequestFactory, TestCase, override_settings

from management.admin import TeamAdmin, sync_oauth_team_names_action
from management.models import Org, Team

User = get_user_model()


def team_names_with_prefix(group: str, groups: list[str] | None = None) -> tuple[str, str] | None:
    """
    Sample function that adds 'Team-' prefix to group names.
    """
    if group.startswith("E"):
        return (f"Team-{group}", group)
    return None


def team_names_strip_suffix(group: str, groups: list[str] | None = None) -> tuple[str, str] | None:
    """
    Sample function that strips suffix after dash.
    """
    if group.startswith("E"):
        return (group.split("-", maxsplit=1)[0], group)
    return None


def team_names_empty(group: str, groups: list[str] | None = None) -> tuple[str, str] | None:
    """
    Sample function that returns None for all groups (causes deletion).
    """
    return None


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True, OIDC_RP_SIGN_ALGO="HS256", OIDC_RP_IDP_SIGN_KEY="test-key"
)
class SyncOauthTeamNamesAdminActionTestCase(TestCase):
    """Test OAuth team name sync admin action."""

    def setUp(self):
        self.org = Org.objects.create(name="test-org")
        self.other_org = Org.objects.create(name="other-org")
        self.factory = RequestFactory()
        self.user = User.objects.create_superuser(
            username="admin", email="admin@example.com", password="admin"
        )
        self.modeladmin = TeamAdmin(Team, site)

    def _create_request(self):
        request = self.factory.get("/admin/management/team/")
        request.user = self.user
        request._messages = CookieStorage(request)
        return request

    def test_update_team_names(self):
        """Test team names are updated based on mapping function."""
        Team.objects.create(name="E123-Students", org=self.org, oauth_group_name="E123-Students")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="E123-Students")
        self.assertEqual(team.name, "E123")

    def test_skip_name_collision(self):
        """Test skipping update when new name already exists."""
        Team.objects.create(name="E123", org=self.org, oauth_group_name="E123")
        Team.objects.create(name="E123-Duplicate", org=self.org, oauth_group_name="E123-Students")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="E123-Students")
        self.assertEqual(team.name, "E123-Duplicate")

    def test_skip_manual_teams(self):
        """Test that manual teams (empty oauth_group_name) are not affected."""
        Team.objects.create(name="ManualTeam", org=self.org, oauth_group_name="")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="")
        self.assertEqual(team.name, "ManualTeam")

    def test_no_changes_when_name_unchanged(self):
        """Test skipping teams when name is already correct."""
        Team.objects.create(name="E123", org=self.org, oauth_group_name="E123")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="E123")
        self.assertEqual(team.name, "E123")

    def test_warning_on_deletion(self):
        """Test warning message when teams would be deleted."""
        Team.objects.create(name="E123", org=self.org, oauth_group_name="E123-Students")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_empty):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="E123-Students")
        self.assertIsNotNone(team)

    def test_filter_by_queryset(self):
        """Test that action only affects selected teams."""
        team1 = Team.objects.create(
            name="E123-Students", org=self.org, oauth_group_name="E123-Students"
        )
        team2 = Team.objects.create(
            name="E456-Students", org=self.org, oauth_group_name="E456-Students"
        )

        request = self._create_request()
        queryset = Team.objects.filter(pk=team1.pk)

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team1.refresh_from_db()
        team2.refresh_from_db()

        self.assertEqual(team1.name, "E123")
        self.assertEqual(team2.name, "E456-Students")

    def test_no_function_configured(self):
        """Test handling when function is not configured."""
        Team.objects.create(name="E123", org=self.org, oauth_group_name="E123")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=None):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        team = Team.objects.get(oauth_group_name="E123")
        self.assertEqual(team.name, "E123")

    def test_multiple_teams_updated(self):
        """Test updating multiple teams at once."""
        Team.objects.create(name="E123-Students", org=self.org, oauth_group_name="E123-Students")
        Team.objects.create(name="E456-Staff", org=self.org, oauth_group_name="E456-Staff")
        Team.objects.create(name="E789-Admins", org=self.org, oauth_group_name="E789-Admins")

        request = self._create_request()
        queryset = Team.objects.all()

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=team_names_strip_suffix):
            sync_oauth_team_names_action(self.modeladmin, request, queryset)

        self.assertEqual(Team.objects.get(oauth_group_name="E123-Students").name, "E123")
        self.assertEqual(Team.objects.get(oauth_group_name="E456-Staff").name, "E456")
        self.assertEqual(Team.objects.get(oauth_group_name="E789-Admins").name, "E789")
