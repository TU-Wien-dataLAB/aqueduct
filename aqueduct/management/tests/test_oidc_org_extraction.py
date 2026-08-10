from django.test import TestCase, override_settings

from management.auth import OIDCBackend, default_org_name, get_org_name
from management.models import Org


class DefaultOrgNameTestCase(TestCase):
    """Test the default_org_name function."""

    def test_returns_first_group(self):
        """Test that default_org_name returns the first group."""
        claims = {"groups": ["org-alpha", "org-beta", "org-gamma"]}
        result = default_org_name(claims)
        self.assertEqual(result, "org-alpha")

    def test_returns_none_when_no_groups(self):
        """Test that default_org_name returns None when groups key is missing."""
        claims = {"email": "test@example.com", "name": "Test User"}
        result = default_org_name(claims)
        self.assertIsNone(result)

    def test_returns_none_when_groups_is_empty_list(self):
        """Test that default_org_name returns None when groups is empty."""
        claims = {"groups": []}
        result = default_org_name(claims)
        self.assertIsNone(result)

    def test_returns_none_when_groups_is_none(self):
        """Test that default_org_name returns None when groups is None."""
        claims = {"groups": None}
        result = default_org_name(claims)
        self.assertIsNone(result)


class GetOrgNameTestCase(TestCase):
    """Test the get_org_name function with custom configurations."""

    def test_uses_default_implementation_when_setting_is_none(self):
        """Test that get_org_name uses default_org_name when setting returns None."""
        claims = {"groups": ["test-org"]}
        result = get_org_name(claims)
        self.assertEqual(result, "default")

    @override_settings(ORG_NAME_FROM_OIDC_FUNCTION=lambda claims: claims.get("custom_org"))
    def test_uses_custom_function_when_configured(self):
        """Test that get_org_name uses custom function when ORG_NAME_FROM_OIDC_FUNCTION is set."""
        claims = {"groups": ["default-org"], "custom_org": "custom-org"}
        result = get_org_name(claims)
        self.assertEqual(result, "custom-org")

    @override_settings(ORG_NAME_FROM_OIDC_FUNCTION=lambda claims: claims.get("groups", [""])[-1])
    def test_custom_function_can_extract_last_group(self):
        """Test custom function that extracts the last group instead of first."""
        claims = {"groups": ["org-first", "org-middle", "org-last"]}
        result = get_org_name(claims)
        self.assertEqual(result, "org-last")

    @override_settings(ORG_NAME_FROM_OIDC_FUNCTION=lambda claims: None)
    def test_custom_function_can_return_none(self):
        """Test that custom function can return None to indicate no org."""
        claims = {"groups": ["some-org"]}
        result = get_org_name(claims)
        self.assertIsNone(result)


@override_settings(
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
    ORG_NAME_FROM_OIDC_FUNCTION=default_org_name,
)
class OIDCBackendOrgTestCase(TestCase):
    """Test the OIDCBackend._org() method."""

    def setUp(self):
        self.backend = OIDCBackend()

    def test_creates_org_if_not_exists(self):
        """Test that _org creates a new Org if it doesn't exist."""
        claims = {"groups": ["new-org"]}
        self.assertEqual(Org.objects.count(), 0)
        org = self.backend._org(claims)
        self.assertEqual(Org.objects.count(), 1)
        self.assertEqual(org.name, "new-org")
        self.assertIsInstance(org, Org)

    def test_reuses_existing_org(self):
        """Test that _org reuses an existing Org."""
        Org.objects.create(name="existing-org")
        self.assertEqual(Org.objects.count(), 1)

        claims = {"groups": ["existing-org"]}
        org = self.backend._org(claims)

        self.assertEqual(Org.objects.count(), 1)
        self.assertEqual(org.name, "existing-org")

    def test_returns_none_when_no_org_name(self):
        """Test that _org returns None when no org name can be extracted."""
        claims = {"email": "test@example.com"}
        org = self.backend._org(claims)
        self.assertIsNone(org)

    def test_returns_none_when_groups_empty(self):
        """Test that _org returns None when groups list is empty."""
        claims = {"groups": []}
        org = self.backend._org(claims)
        self.assertIsNone(org)

    def test_returns_none_when_groups_none(self):
        """Test that _org returns None when groups is None."""
        claims = {"groups": None}
        org = self.backend._org(claims)
        self.assertIsNone(org)

    def test_first_group_determines_org(self):
        """Test that the first group determines the organization."""
        claims = {"groups": ["primary-org", "secondary-org", "tertiary-org"]}
        org = self.backend._org(claims)
        self.assertEqual(org.name, "primary-org")


@override_settings(
    OIDC_RP_SIGN_ALGO="HS256",
    OIDC_RP_IDP_SIGN_KEY="test-key",
    ORG_NAME_FROM_OIDC_FUNCTION=default_org_name,
)
class OIDCBackendOrgIntegrationTestCase(TestCase):
    """Integration tests for org creation in authentication flow."""

    def setUp(self):
        self.backend = OIDCBackend()

    def test_multiple_users_same_org_reuse_org(self):
        """Test that multiple users from same org reuse the same Org object."""
        claims1 = {"groups": ["shared-org"], "email": "user1@example.com"}
        claims2 = {"groups": ["shared-org"], "email": "user2@example.com"}

        org1 = self.backend._org(claims1)
        org2 = self.backend._org(claims2)

        self.assertEqual(Org.objects.count(), 1)
        self.assertEqual(org1.id, org2.id)
        self.assertEqual(org1.name, "shared-org")
        self.assertEqual(org2.name, "shared-org")

    def test_multiple_users_different_orgs_create_separate_orgs(self):
        """Test that users from different orgs create separate Org objects."""
        claims1 = {"groups": ["org-a"], "email": "user1@example.com"}
        claims2 = {"groups": ["org-b"], "email": "user2@example.com"}

        org1 = self.backend._org(claims1)
        org2 = self.backend._org(claims2)

        self.assertEqual(Org.objects.count(), 2)
        self.assertNotEqual(org1.id, org2.id)
        self.assertEqual(org1.name, "org-a")
        self.assertEqual(org2.name, "org-b")
