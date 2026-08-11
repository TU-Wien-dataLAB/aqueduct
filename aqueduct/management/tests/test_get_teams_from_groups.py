from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.models import Org
from management.tests.helpers import sample_team_names

User = get_user_model()


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FUNCTION=sample_team_names,
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

    def test_get_teams_no_function_configured(self):
        """Test handling when no function is configured."""

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=None):
            backend = OIDCBackend()
            claims = {"email": "test@example.com", "groups": ["E123-Students"]}

            # When function is None, the default lambda returns None for all groups
            result = backend._get_teams(claims)

            self.assertEqual(result, [])
