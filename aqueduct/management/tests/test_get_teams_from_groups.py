from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.models import Org

User = get_user_model()


def sample_team_names_from_groups(
    group: str, groups: list[str] | None = None
) -> tuple[str, str] | None:
    if group.startswith("E"):
        team_name = group.split("-", maxsplit=1)[0]
        return (team_name, group)
    return None


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FUNCTION=sample_team_names_from_groups,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class GetTeamsFromGroupsTestCase(TestCase):
    """Test _get_teams method."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_get_teams_with_valid_claims(self):
        """Test extracting teams from valid claims dict."""
        claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        result = self.backend._get_teams(claims)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("E123", "E123-Students"))
        self.assertEqual(result[1], ("E456", "E456-Staff"))

    def test_get_teams_with_empty_groups(self):
        """Test handling of empty groups list in claims."""
        claims = {"email": "test@example.com", "groups": []}

        result = self.backend._get_teams(claims)

        self.assertEqual(result, [])

    def test_get_teams_with_missing_groups_key(self):
        """Test handling of missing 'groups' key in claims."""
        claims = {"email": "test@example.com"}

        result = self.backend._get_teams(claims)

        self.assertEqual(result, [])

    def test_get_teams_with_none_groups(self):
        """Test handling of None groups value in claims."""
        claims = {"email": "test@example.com", "groups": None}

        result = self.backend._get_teams(claims)

        self.assertEqual(result, [])

    def test_get_teams_feature_disabled(self):
        """Test that empty list is returned when feature is disabled."""
        with override_settings(ENABLE_OAUTH_GROUP_MANAGEMENT=False):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            result = backend._get_teams(claims)

            self.assertEqual(result, [])

    def test_get_teams_filters_invalid_results(self):
        """Test that invalid return values from function are filtered out."""

        def bad_function(group, groups=None):
            if group == "valid":
                return ("ValidTeam", "valid")
            elif group == "wrong_type":
                return "not-a-tuple"
            elif group == "wrong_length":
                return ("only", "one", "extra")
            elif group == "empty_team":
                return ("", "empty_team")
            elif group == "none_team":
                return (None, "none_team")
            return None

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=bad_function):
            backend = OIDCBackend()
            claims = {
                "email": "test@example.com",
                "groups": ["valid", "wrong_type", "wrong_length", "empty_team", "none_team"],
            }

            result = backend._get_teams(claims)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], ("ValidTeam", "valid"))

    def test_get_teams_function_raises_exception(self):
        """Test that exceptions in function are caught and logged."""

        def raising_function(group, groups=None):
            if group == "error":
                raise ValueError("Test error")
            return ("ValidTeam", group)

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=raising_function):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["valid", "error", "valid2"]}

            result = backend._get_teams(claims)

            # Should have 2 valid results, skipping the one that raised exception
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], ("ValidTeam", "valid"))
            self.assertEqual(result[1], ("ValidTeam", "valid2"))

    def test_get_teams_strips_whitespace(self):
        """Test that team names and group names are stripped of whitespace."""

        def whitespace_function(group, groups=None):
            return ("  TeamName  ", "  GroupName  ")

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=whitespace_function):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["test-group"]}

            result = backend._get_teams(claims)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], ("TeamName", "GroupName"))

    def test_get_teams_no_function_configured(self):
        """Test handling when no function is configured."""

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=None):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            # When function is None, the default lambda returns None for all groups
            result = backend._get_teams(claims)

            self.assertEqual(result, [])
