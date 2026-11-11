from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.tests.helpers import (
    SNIPPET_DISPLAY_MAP_BAD_ENTRIES,
    SNIPPET_DISPLAY_MAP_NON_LIST,
    SNIPPET_DISPLAY_MAP_NONE,
    SNIPPET_TEAM_NAMES_AND_MAP,
    SNIPPET_TEAM_NAMES_NON_LIST,
    SNIPPET_TEAM_NAMES_NONE,
    SNIPPET_TEAM_NAMES_RAISES,
    seed_active_config,
)


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class GetTeamNamesTestCase(TestCase):
    """Test the _get_team_names extraction method."""

    def setUp(self):
        seed_active_config(SNIPPET_TEAM_NAMES_AND_MAP)
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

    def test_returns_empty_when_team_names_returns_non_list(self):
        seed_active_config(SNIPPET_TEAM_NAMES_NON_LIST)

        result = OIDCBackend()._get_team_names({"groups": ["E123-Students"]})

        self.assertEqual(result, [])

    def test_returns_empty_when_team_names_returns_none(self):
        seed_active_config(SNIPPET_TEAM_NAMES_NONE)

        result = OIDCBackend()._get_team_names({"groups": ["E123-Students"]})

        self.assertEqual(result, [])

    def test_returns_empty_when_team_names_raises(self):
        seed_active_config(SNIPPET_TEAM_NAMES_RAISES)

        result = OIDCBackend()._get_team_names({"groups": ["E123-Students"]})

        self.assertEqual(result, [])

    def test_returns_empty_when_management_disabled(self):
        with override_settings(ENABLE_OAUTH_GROUP_MANAGEMENT=False):
            result = OIDCBackend()._get_team_names({"groups": ["E123-Students"]})

        self.assertEqual(result, [])


@override_settings(
    ENABLE_OAUTH_GROUP_MANAGEMENT=True,
    ENABLE_OAUTH_GROUP_CREATION=True,
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
)
class GetTeamsTestCase(TestCase):
    """Test the _get_teams display-mapping method."""

    def setUp(self):
        seed_active_config(SNIPPET_TEAM_NAMES_AND_MAP)
        self.backend = OIDCBackend()

    def test_map_team_names(self):
        result = self.backend._get_teams(["E123-Students", "E456-Staff"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("E123", "E123-Students"))
        self.assertEqual(result[1], ("E456", "E456-Staff"))

    def test_empty_team_names_returns_empty(self):
        result = self.backend._get_teams([])
        self.assertEqual(result, [])

    def test_returns_empty_when_display_mapping_returns_none(self):
        seed_active_config(SNIPPET_DISPLAY_MAP_NONE)

        result = OIDCBackend()._get_teams(["E123-Students"])

        self.assertEqual(result, [])

    def test_returns_empty_when_display_mapping_returns_non_list(self):
        seed_active_config(SNIPPET_DISPLAY_MAP_NON_LIST)

        result = OIDCBackend()._get_teams(["E123-Students"])

        self.assertEqual(result, [])

    def test_skips_malformed_entries(self):
        seed_active_config(SNIPPET_DISPLAY_MAP_BAD_ENTRIES)

        result = OIDCBackend()._get_teams(["E123-Students"])

        self.assertEqual(result, [("E123", "E123-Students")])
