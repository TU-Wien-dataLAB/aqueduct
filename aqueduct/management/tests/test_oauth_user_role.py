"""Tests for OAuth user role assignment (_update_user_role and role flows)."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.models import Org, UserProfile
from management.tests.helpers import sample_extract_teams, sample_map_team_names

User = get_user_model()

ROLE_SETTINGS = {
    "ENABLE_OAUTH_GROUP_MANAGEMENT": True,
    "ENABLE_OAUTH_GROUP_CREATION": True,
    "OAUTH_TEAM_NAMES_FUNCTION": sample_extract_teams,
    "OAUTH_DISPLAY_TEAM_NAMES_FUNCTION": sample_map_team_names,
    "OIDC_RP_SIGN_ALGO": "HS256",
    "OIDC_RP_IDP_SIGN_KEY": "test-key",
}


@override_settings(**ROLE_SETTINGS)
class UpdateUserRoleTestCase(TestCase):
    """Test _update_user_role directly."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")

    def _make_user(self):
        user = User.objects.create_user(username="testuser", email="test@example.com")
        profile = UserProfile.objects.create(user=user, org=self.org)
        return user, profile

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_admin_when_admin_group_in_team_names(self):
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["E123-Students", "E456-Staff"])

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_user_when_admin_group_not_in_team_names(self):
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["E456-Staff"])

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_user_when_team_names_empty(self):
        """An admin is demoted when no team names remain."""
        user, profile = self._make_user()
        user.is_staff = True
        user.is_superuser = True
        profile.group = "admin"
        user.save()
        profile.save()

        self.backend._update_user_role(user, profile, [])

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    @override_settings(ADMIN_GROUP=None)
    def test_user_when_admin_group_not_configured(self):
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["E123-Students"])

        profile.refresh_from_db()
        self.assertEqual(profile.group, "user")


@override_settings(**ROLE_SETTINGS)
class UserRoleIntegrationTestCase(TestCase):
    """Test role assignment through create_user and update_user."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_create_user_assigns_admin_role(self):
        claims = {"email": "test@example.com", "groups": ["E123-Students"]}

        with patch.object(OIDCBackend, "_org", return_value=self.org):
            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)

            with patch(
                "mozilla_django_oidc.auth.OIDCAuthenticationBackend.create_user", return_value=user
            ):
                result_user = self.backend.create_user(claims)

            self.assertIsNotNone(result_user)
            profile = UserProfile.objects.get(user=user)
            user.refresh_from_db()
            self.assertEqual(profile.group, "admin")
            self.assertTrue(user.is_staff)
            self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_create_user_assigns_regular_role_when_not_admin(self):
        claims = {"email": "test@example.com", "groups": ["E456-Staff"]}

        with patch.object(OIDCBackend, "_org", return_value=self.org):
            user = User.objects.create_user(username="testuser", email="test@example.com")
            user.groups.add(self.user_group)

            with patch(
                "mozilla_django_oidc.auth.OIDCAuthenticationBackend.create_user", return_value=user
            ):
                result_user = self.backend.create_user(claims)

            self.assertIsNotNone(result_user)
            profile = UserProfile.objects.get(user=user)
            user.refresh_from_db()
            self.assertEqual(profile.group, "user")
            self.assertFalse(user.is_staff)
            self.assertFalse(user.is_superuser)

    @override_settings(ADMIN_GROUP="E123-Students")
    def test_update_user_demotes_admin_when_admin_group_removed(self):
        user = User.objects.create_user(username="testuser", email="test@example.com")
        user.groups.add(self.user_group)
        user.is_staff = True
        user.is_superuser = True
        user.save()
        profile = UserProfile.objects.create(user=user, org=self.org)
        profile.group = "admin"
        profile.save()

        # Admin group removed; only a non-admin team remains
        updated_claims = {"email": "test@example.com", "groups": ["E456-Staff"]}

        with patch.object(OIDCBackend, "_org", return_value=self.org):
            self.backend.update_user(user, updated_claims)

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
