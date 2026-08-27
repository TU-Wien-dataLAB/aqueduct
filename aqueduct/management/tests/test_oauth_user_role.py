"""Tests for OAuth user role assignment (_update_user_role and role flows)."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings

from management.auth import OIDCBackend
from management.models import Org, UserProfile
from management.tests.helpers import (
    SNIPPET_USER_GROUP_BASIC,
    SNIPPET_USER_GROUP_INVALID,
    SNIPPET_USER_GROUP_RAISES,
    seed_active_config,
)

User = get_user_model()

ROLE_SETTINGS = {
    "ENABLE_OAUTH_GROUP_MANAGEMENT": True,
    "ENABLE_OAUTH_GROUP_CREATION": True,
    "OIDC_RP_SIGN_ALGO": "HS256",
    "OIDC_RP_IDP_SIGN_KEY": "test-key",
}


@override_settings(**ROLE_SETTINGS)
class UpdateUserRoleTestCase(TestCase):
    """Test _update_user_role directly (claims-driven via user_group())."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")

    def _make_user(self, email="test@example.com"):
        user = User.objects.create_user(username="testuser", email=email)
        profile = UserProfile.objects.create(user=user, org=self.org)
        return user, profile

    def test_admin_when_user_group_returns_admin(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()

        self.backend._update_user_role(
            user, profile, ["E123-Students"], {"groups": ["E123-Students"]}
        )

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    def test_org_admin_when_user_group_returns_org_admin(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()

        self.backend._update_user_role(
            user, profile, ["E456-OrgAdmins"], {"groups": ["E456-OrgAdmins"]}
        )

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "org-admin")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_user_when_user_group_returns_user(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["E789-Staff"], {"groups": ["E789-Staff"]})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_admin_demoted_when_user_group_no_longer_admin(self):
        """An admin is demoted when the claims no longer map to 'admin'."""
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()
        user.is_staff = True
        user.is_superuser = True
        profile.group = "admin"
        user.save()
        profile.save()

        self.backend._update_user_role(user, profile, ["E789-Staff"], {"groups": ["E789-Staff"]})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_default_user_group_when_no_snippet(self):
        """With no active snippet the base class user_group() returns 'user'."""
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, [], {"groups": ["E123-Students"]})

        profile.refresh_from_db()
        self.assertEqual(profile.group, "user")

    def test_user_when_user_group_returns_invalid_group(self):
        seed_active_config(SNIPPET_USER_GROUP_INVALID)
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, [], {})

        profile.refresh_from_db()
        self.assertEqual(profile.group, "user")

    def test_user_when_user_group_raises(self):
        seed_active_config(SNIPPET_USER_GROUP_RAISES)
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, [], {})

        profile.refresh_from_db()
        self.assertEqual(profile.group, "user")

    @override_settings(ADMIN_GROUP="admins")
    def test_admin_group_promotes_to_admin_on_login(self):
        """A user in the configured ADMIN_GROUP becomes admin regardless of the snippet."""
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()

        # Snippet would map E789-Staff -> "user", but ADMIN_GROUP overrides it.
        self.backend._update_user_role(
            user, profile, ["E789-Staff", "admins"], {"groups": ["E789-Staff", "admins"]}
        )

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_GROUP="admins")
    def test_admin_group_overrides_admin_snippet_demotion(self):
        """ADMIN_GROUP keeps a user admin even if the snippet maps them to user."""
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()
        user.is_staff = False
        user.is_superuser = False
        profile.group = "user"
        user.save()
        profile.save()

        self.backend._update_user_role(user, profile, ["admins"], {"groups": ["admins"]})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_GROUP="admins")
    def test_admin_group_not_matched_stays_user(self):
        """User outside ADMIN_GROUP falls back to the snippet's user_group()."""
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["E789-Staff"], {"groups": ["E789-Staff"]})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    @override_settings(ADMIN_GROUP="admins")
    def test_admin_group_promotion_without_snippet(self):
        """With no active snippet, ADMIN_GROUP alone still promotes to admin."""
        user, profile = self._make_user()

        self.backend._update_user_role(user, profile, ["admins"], {"groups": ["admins"]})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_SUPERUSER_EMAILS=["you@example.com"])
    def test_admin_superuser_email_forced_superuser_regardless_of_group(self):
        """ADMIN_SUPERUSER_EMAILS emails become superuser even when their group is not admin."""
        # No snippet / groups -> group stays 'user', but the allowlisted email is superuser.
        user, profile = self._make_user(email="you@example.com")

        self.backend._update_user_role(user, profile, [], {})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_SUPERUSER_EMAILS=["you@example.com"])
    def test_admin_superuser_email_ignores_case(self):
        """ADMIN_SUPERUSER_EMAILS matching is case-insensitive."""
        # Note: _make_user sets the email as given on the user; OIDC delivers a
        # mixed-case email but the user is the same account.
        user, profile = self._make_user(email="YOU@EXAMPLE.com")

        self.backend._update_user_role(user, profile, [], {})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertTrue(user.is_superuser)

    @override_settings(ADMIN_SUPERUSER_EMAILS=["someone-else@example.com"])
    def test_admin_superuser_email_mismatch_stays_user(self):
        """An email not in ADMIN_SUPERUSER_EMAILS is not made superuser (regardless of group)."""
        user, profile = self._make_user(email="you@example.com")

        self.backend._update_user_role(user, profile, [], {})

        profile.refresh_from_db()
        user.refresh_from_db()
        self.assertEqual(profile.group, "user")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)


@override_settings(**ROLE_SETTINGS)
class UserRoleIntegrationTestCase(TestCase):
    """Test role assignment through create_user and update_user."""

    def setUp(self):
        self.backend = OIDCBackend()
        self.org = Org.objects.create(name="test-org")
        self.user_group, _ = Group.objects.get_or_create(name="user")

    def test_create_user_assigns_admin_role(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
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

    def test_create_user_assigns_regular_role_when_not_admin(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
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

    def test_update_user_demotes_admin_when_user_group_removed(self):
        seed_active_config(SNIPPET_USER_GROUP_BASIC)
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
