from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.tests.helpers import sample_extract_teams, sample_map_team_names


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FUNCTION=sample_extract_teams,
    OAUTH_DISPLAY_TEAM_NAMES_FUNCTION=sample_map_team_names,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class GetTeamNamesTestCase(TestCase):
    """Test the _get_team_names extraction method."""

    def setUp(self):
        self.backend = OIDCBackend()

    def test_extracts_team_names_from_valid_claims(self):
        claims = {"email": "test@example.com", "groups": ["E123-Students", "E456-Staff"]}

        result = self.backend._get_team_names(claims)

        self.assertEqual(result, ["E123-Students", "E456-Staff"])

    def test_filters_non_matching_groups(self):
        claims = {"groups": ["E123-Students", "Other-Group", "E456-Staff"]}

        result = self.backend._get_team_names(claims)

        self.assertEqual(result, ["E123-Students", "E456-Staff"])

    def test_empty_groups_returns_empty(self):
        result = self.backend._get_team_names({"groups": []})
        self.assertEqual(result, [])

    def test_missing_groups_key_returns_empty(self):
        result = self.backend._get_team_names({"email": "test@example.com"})
        self.assertEqual(result, [])

    def test_none_groups_returns_empty(self):
        result = self.backend._get_team_names({"groups": None})
        self.assertEqual(result, [])

    def test_returns_empty_when_function_returns_non_list(self):
        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=lambda claims: "not-a-list"):
            backend = OIDCBackend()

            result = backend._get_team_names({"groups": ["E123-Students"]})

            self.assertEqual(result, [])

    def test_returns_empty_when_function_is_none(self):
        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=None):
            backend = OIDCBackend()

            result = backend._get_team_names({"groups": ["E123-Students"]})

            self.assertEqual(result, [])

    def test_returns_empty_when_function_raises(self):
        def broken(claims):
            raise RuntimeError("boom")

        with override_settings(OAUTH_TEAM_NAMES_FUNCTION=broken):
            backend = OIDCBackend()

            result = backend._get_team_names({"groups": ["E123-Students"]})

            self.assertEqual(result, [])

    def test_returns_empty_when_management_disabled(self):
        with override_settings(ENABLE_OAUTH_GROUP_MANAGEMENT=False):
            backend = OIDCBackend()

            result = backend._get_team_names({"groups": ["E123-Students"]})

            self.assertEqual(result, [])


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OAUTH_TEAM_NAMES_FUNCTION=sample_extract_teams,
    OAUTH_DISPLAY_TEAM_NAMES_FUNCTION=sample_map_team_names,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class GetTeamsTestCase(TestCase):
    """Test the _get_teams display-mapping method."""

    def setUp(self):
        self.backend = OIDCBackend()

    def test_map_team_names(self):
        result = self.backend._get_teams(["E123-Students", "E456-Staff"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("E123", "E123-Students"))
        self.assertEqual(result[1], ("E456", "E456-Staff"))

    def test_empty_team_names_returns_empty(self):
        result = self.backend._get_teams([])
        self.assertEqual(result, [])

    def test_returns_empty_when_function_not_configured(self):
        with override_settings(OAUTH_DISPLAY_TEAM_NAMES_FUNCTION=None):
            backend = OIDCBackend()

            result = backend._get_teams(["E123-Students"])

            self.assertEqual(result, [])

    def test_returns_empty_when_function_returns_non_list(self):
        with override_settings(OAUTH_DISPLAY_TEAM_NAMES_FUNCTION=lambda names: "nope"):
            backend = OIDCBackend()

            result = backend._get_teams(["E123-Students"])

            self.assertEqual(result, [])

    def test_skips_malformed_entries(self):
        def bad_map(names):
            return [("E123", "E123-Students"), ("bad",), (None, "x"), 42]

        with override_settings(OAUTH_DISPLAY_TEAM_NAMES_FUNCTION=bad_map):
            backend = OIDCBackend()

            result = backend._get_teams(["E123-Students"])

            self.assertEqual(result, [("E123", "E123-Students")])
