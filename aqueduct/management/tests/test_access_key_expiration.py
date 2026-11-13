from datetime import timedelta
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from management.models import Org, ServiceAccount, Team, Token, UserGroup, UserProfile

User = get_user_model()


class AccessKeyExpirationViewTests(TestCase):
    def setUp(self):
        org = Org.objects.create(name="expiry-org")
        self.team = Team.objects.create(name="expiry-team", org=org)
        self.user = User.objects.create_user(username="expiry", email="expiry@example.com")
        UserProfile.objects.create(user=self.user, org=org, group=UserGroup.ORG_ADMIN.value)
        self.client.force_login(self.user)

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_user_key_create_and_edit_cap_missing_or_late_expiration(self):
        before = timezone.now()
        response = self.client.post(reverse("token_create"), {"name": "key", "expires_at": ""})
        self.assertEqual(response.status_code, 302)
        token = Token.objects.get(user=self.user)
        self.assertGreaterEqual(token.expires_at, before + timedelta(days=7))
        self.assertLessEqual(token.expires_at, timezone.now() + timedelta(days=7))

        far_future = (timezone.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        response = self.client.post(
            reverse("token_edit", kwargs={"id": token.id}),
            {"name": "key", "expires_at": far_future},
        )
        self.assertEqual(response.status_code, 302)
        token.refresh_from_db()
        self.assertLessEqual(token.expires_at, timezone.now() + timedelta(days=7))

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_service_account_create_and_edit_cap_expiration(self):
        response = self.client.post(
            reverse("service_account_create", kwargs={"id": self.team.id}),
            {"name": "service", "description": "", "token_expires_at": ""},
        )
        self.assertEqual(response.status_code, 302)
        account = ServiceAccount.objects.get(name="service")
        token = account.token
        self.assertIsNotNone(token.expires_at)
        self.assertLessEqual(token.expires_at, timezone.now() + timedelta(days=7))
        response = self.client.post(
            reverse("service_account_edit", kwargs={"service_account_id": account.id}),
            {"name": "service", "description": "", "token_expires_at": ""},
        )
        self.assertEqual(response.status_code, 302)
        token.refresh_from_db()
        self.assertIsNotNone(token.expires_at)
        self.assertLessEqual(token.expires_at, timezone.now() + timedelta(days=7))

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=-1)
    def test_unlimited_user_key_can_have_no_expiration(self):
        response = self.client.post(reverse("token_create"), {"name": "key", "expires_at": ""})
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(Token.objects.get(user=self.user).expires_at)


class AccessKeyExpirationMigrationTests(TransactionTestCase):
    def test_migrate_shortens_only_keys_beyond_new_deadline(self):
        user = User.objects.create_user(username="migration")
        now = timezone.now()
        tokens = []
        for name, expiry in [
            ("never", None),
            ("later", now + timedelta(days=60)),
            ("earlier", now + timedelta(days=2)),
            ("expired", now - timedelta(days=1)),
        ]:
            token = Token(name=name, user=user, expires_at=expiry)
            token._set_new_key()
            token.save()
            tokens.append(token)

        with override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=30):
            call_command("migrate", verbosity=0, stdout=StringIO())
        for token in tokens:
            token.refresh_from_db()
        self.assertGreaterEqual(tokens[0].expires_at, now + timedelta(days=30))
        self.assertEqual(tokens[0].expires_at, tokens[1].expires_at)
        self.assertEqual(tokens[2].expires_at, now + timedelta(days=2))
        self.assertEqual(tokens[3].expires_at, now - timedelta(days=1))

        with override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7):
            call_command("migrate", verbosity=0, stdout=StringIO())
        for token in tokens:
            token.refresh_from_db()
        self.assertGreaterEqual(tokens[0].expires_at, now + timedelta(days=7))
        self.assertLess(tokens[0].expires_at, now + timedelta(days=8))
        self.assertEqual(tokens[0].expires_at, tokens[1].expires_at)
        self.assertEqual(tokens[2].expires_at, now + timedelta(days=2))
        self.assertEqual(tokens[3].expires_at, now - timedelta(days=1))

        with override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=-1):
            call_command("migrate", verbosity=0, stdout=StringIO())
        tokens[0].refresh_from_db()
        self.assertLess(tokens[0].expires_at, now + timedelta(days=8))
