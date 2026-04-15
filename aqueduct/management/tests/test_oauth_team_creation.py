"""
Tests for OAuth team creation and membership management.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.models import Org, Team, TeamMembership, UserProfile

User = get_user_model()


def sample_team_names_from_groups(groups: list[str]) -> list[str]:
    """
    Sample implementation that filters groups starting with 'E'
    and extracts team names (removes suffix after dash).
    Example: ['E123-Students', 'E456-Staff'] -> ['E123', 'E456']
    """
    team_names = []
    for group in groups:
        if group.startswith("E"):
            team_name = group.split("-")[0]
            team_names.append(team_name)
    return team_names


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=sample_team_names_from_groups,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class OAuthTeamCreationTestCase(TestCase):
    """Test team creation from OAuth groups."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_team_created_from_groups(self):
        """Test that teams are created from OAuth groups."""
        claims = {
            "email": "test@example.com",
            "groups": ["E123-Students", "E456-Staff", "Other-Group"],
        }

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 2)
        self.assertTrue(Team.objects.filter(name="E123", org=self.org).exists())
        self.assertTrue(Team.objects.filter(name="E456", org=self.org).exists())
        self.assertFalse(Team.objects.filter(name="Other-Group", org=self.org).exists())

        memberships = TeamMembership.objects.filter(user_profile=profile)
        self.assertEqual(memberships.count(), 2)

        team1 = Team.objects.get(name="E123", org=self.org)
        team2 = Team.objects.get(name="E456", org=self.org)
        self.assertEqual(team1.oauth_group_name, "E123")
        self.assertEqual(team2.oauth_group_name, "E456")
        self.assertTrue(team1.managed_by_oauth)
        self.assertTrue(team2.managed_by_oauth)

    def test_team_reuse_existing(self):
        """Test that existing teams are reused, not duplicated."""
        team = Team.objects.create(name="E123", org=self.org, oauth_group_name="")

        claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 1)
        self.assertEqual(TeamMembership.objects.count(), 1)
        self.assertEqual(TeamMembership.objects.first().team, team)
        self.assertEqual(team.oauth_group_name, "")
        self.assertFalse(team.managed_by_oauth)

    def test_no_teams_when_feature_disabled(self):
        """Test that no teams are created when feature is disabled."""
        with override_settings(ENABLE_OAUTH_GROUP_MANAGEMENT=False):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)
            profile = UserProfile.objects.create(user=user, org=self.org)

            backend._sync_teams(user, profile, claims["groups"])

            self.assertEqual(Team.objects.count(), 0)
            self.assertEqual(TeamMembership.objects.count(), 0)

    def test_empty_groups_list(self):
        """Test handling of empty groups list."""
        claims = {"email": "test@example.com", "groups": []}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 0)
        self.assertEqual(TeamMembership.objects.count(), 0)

    def test_none_groups(self):
        """Test handling of None groups."""
        claims = {"email": "test@example.com", "groups": None}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 0)
        self.assertEqual(TeamMembership.objects.count(), 0)


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=sample_team_names_from_groups,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class OAuthTeamMembershipTestCase(TestCase):
    """Test OAuth team membership synchronization."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_user_added_to_teams_on_create(self):
        """Test that user is added to teams when created."""
        claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        memberships = TeamMembership.objects.filter(user_profile=profile)
        self.assertEqual(memberships.count(), 2)

        team_names = {m.team.name for m in memberships}
        self.assertEqual(team_names, {"E123", "E456"})

        team1 = Team.objects.get(name="E123", org=self.org)
        team2 = Team.objects.get(name="E456", org=self.org)
        self.assertEqual(team1.oauth_group_name, "E123")
        self.assertEqual(team2.oauth_group_name, "E456")

    def test_user_removed_from_teams_when_group_removed(self):
        """Test that user is removed from teams when groups are removed."""
        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        initial_groups = ["E123-Students", "E456-Staff"]
        self.backend._sync_teams(user, profile, initial_groups)

        self.assertEqual(TeamMembership.objects.filter(user_profile=profile).count(), 2)

        updated_groups = ["E123-Students"]
        self.backend._sync_teams(user, profile, updated_groups)

        memberships = TeamMembership.objects.filter(user_profile=profile)
        self.assertEqual(memberships.count(), 1)
        self.assertEqual(memberships.first().team.name, "E123")

    def test_membership_sync_on_update(self):
        """Test membership sync on update_user()."""
        claims_initial = {"email": "test@example.com", "groups": ["E123-Students"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims_initial["groups"])
        self.assertEqual(TeamMembership.objects.filter(user_profile=profile).count(), 1)

        claims_updated = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        self.backend._sync_teams(user, profile, claims_updated["groups"])

        memberships = TeamMembership.objects.filter(user_profile=profile)
        self.assertEqual(memberships.count(), 2)
        team_names = {m.team.name for m in memberships}
        self.assertEqual(team_names, {"E123", "E456"})

    def test_org_boundaries_teams_in_different_org(self):
        """Test that users can only join teams in their own org."""
        other_org = Org.objects.create(name="other-org")

        claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        team = Team.objects.filter(name="E123").first()
        self.assertIsNotNone(team)
        self.assertEqual(team.org, self.org)

        team_in_other_org = Team.objects.filter(name="E123", org=other_org).first()
        self.assertIsNone(team_in_other_org)


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=sample_team_names_from_groups,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class OAuthTeamEdgeCasesTestCase(TestCase):
    """Test OAuth team edge cases."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.other_org = Org.objects.create(name="other-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_duplicate_team_names_different_orgs(self):
        """Test that same team name can exist in different orgs."""
        claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        user1 = User.objects.create_user(username="testuser1", email="test1@example.com")
        user1.groups.add(self.user_group)
        profile1 = UserProfile.objects.create(user=user1, org=self.org)

        user2 = User.objects.create_user(username="testuser2", email="test2@example.com")
        user2.groups.add(self.user_group)
        profile2 = UserProfile.objects.create(user=user2, org=self.other_org)

        self.backend._sync_teams(user1, profile1, claims["groups"])
        self.backend._sync_teams(user2, profile2, claims["groups"])

        self.assertEqual(Team.objects.filter(name="E123").count(), 2)

        team1 = Team.objects.get(name="E123", org=self.org)
        team2 = Team.objects.get(name="E123", org=self.other_org)

        self.assertNotEqual(team1.id, team2.id)
        self.assertEqual(team1.oauth_group_name, "E123")
        self.assertEqual(team2.oauth_group_name, "E123")

    def test_special_characters_in_group_names(self):
        """Test handling of special characters in group names."""
        claims = {
            "email": "test@example.com",
            "groups": ["E123-Students", "E456_Test", "E789.Test"],
        }

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 3)
        team_names = {t.name for t in Team.objects.all()}
        self.assertEqual(team_names, {"E123", "E456_Test", "E789.Test"})

    def test_very_long_team_names(self):
        """Test handling of very long team names."""
        long_group_name = "E" + "123" * 80 + "-Students"

        claims = {"email": "test@example.com", "groups": [long_group_name]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        team = Team.objects.first()
        self.assertIsNotNone(team)

    def test_case_sensitivity(self):
        """Test that team names preserve case from group names."""
        claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        self.assertEqual(Team.objects.count(), 2)
        team_names = {t.name for t in Team.objects.all()}
        self.assertEqual(team_names, {"E123", "E456"})


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class OAuthTeamSettingsTestCase(TestCase):
    """Test OAuth team settings behavior."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_default_function_returns_empty_list(self):
        """Test that default function returns empty list."""
        from django.conf import settings

        func = getattr(settings, "OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION", lambda groups: [])
        result = func(["E123-Students", "E456-Staff"])
        self.assertEqual(result, [])

    def test_custom_filter_function(self):
        """Test custom filter function."""

        def custom_filter(groups: list[str]) -> list[str]:
            return [g for g in groups if g in {"E123", "E456"}]

        with override_settings(OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=custom_filter):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123", "E456", "E789"]}

            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)
            profile = UserProfile.objects.create(user=user, org=self.org)

            backend._sync_teams(user, profile, claims["groups"])

            self.assertEqual(Team.objects.count(), 2)
            team_names = {t.name for t in Team.objects.all()}
            self.assertEqual(team_names, {"E123", "E456"})

    def test_enabled_flag_false(self):
        """Test that no sync happens when ENABLE_OAUTH_GROUP_MANAGEMENT=False."""
        with override_settings(ENABLE_OAUTH_GROUP_MANAGEMENT=False):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)
            profile = UserProfile.objects.create(user=user, org=self.org)

            backend._sync_teams(user, profile, claims["groups"])

            self.assertEqual(Team.objects.count(), 0)
            self.assertEqual(TeamMembership.objects.count(), 0)

    def test_creation_disabled_flag(self):
        """Test that teams are not created when ENABLE_OAUTH_GROUP_CREATION=False."""
        with override_settings(ENABLE_OAUTH_GROUP_CREATION=False):
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)
            profile = UserProfile.objects.create(user=user, org=self.org)

            self.backend._sync_teams(user, profile, claims["groups"])

            self.assertEqual(Team.objects.count(), 0)
            self.assertEqual(TeamMembership.objects.count(), 0)

    def test_manual_team_not_affected_by_oauth_sync(self):
        """Test that manually created teams without oauth_group_name are not affected."""
        manual_team = Team.objects.create(name="ManualTeam", org=self.org, oauth_group_name="")

        claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        self.backend._sync_teams(user, profile, claims["groups"])

        manual_team.refresh_from_db()
        self.assertEqual(manual_team.oauth_group_name, "")
        self.assertFalse(manual_team.managed_by_oauth)


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION=sample_team_names_from_groups,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class OAuthTeamIntegrationTestCase(TestCase):
    """Integration tests for OAuth team creation in create_user and update_user."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_create_user_syncs_teams(self):
        """Test that create_user syncs teams from OAuth groups."""
        from unittest.mock import patch

        claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        with patch.object(OIDCBackend, "_org", return_value=self.org):
            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)

            with patch(
                "mozilla_django_oidc.auth.OIDCAuthenticationBackend.create_user", return_value=user
            ):
                result_user = self.backend.create_user(claims)

                self.assertIsNotNone(result_user)

                profile = UserProfile.objects.get(user=user)
                memberships = TeamMembership.objects.filter(user_profile=profile)

        self.assertEqual(memberships.count(), 2)
        team_names = {m.team.name for m in memberships}
        self.assertEqual(team_names, {"E123", "E456"})

    def test_update_user_syncs_teams(self):
        """Test that update_user syncs teams from OAuth groups."""
        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        profile = UserProfile.objects.create(user=user, org=self.org)

        initial_claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        self.backend._sync_teams(user, profile, initial_claims["groups"])
        self.assertEqual(TeamMembership.objects.filter(user_profile=profile).count(), 1)

        updated_claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        self.backend.update_user(user, updated_claims)

        user.refresh_from_db()
        profile = UserProfile.objects.get(user=user)
        memberships = TeamMembership.objects.filter(user_profile=profile)

        self.assertEqual(memberships.count(), 2)
        team_names = {m.team.name for m in memberships}
        self.assertEqual(team_names, {"E123", "E456"})
