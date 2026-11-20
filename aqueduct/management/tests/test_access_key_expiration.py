from datetime import timedelta
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from management.models import Org, ServiceAccount, Team, Token, UserGroup, UserProfile

User = get_user_model()

_RENDER_HTML = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)


@_RENDER_HTML
class AccessKeyExpirationViewTests(TestCase):
    def setUp(self):
        org = Org.objects.create(name="expiry-org")
        self.team = Team.objects.create(name="expiry-team", org=org)
        self.user = User.objects.create_user(username="expiry", email="expiry@example.com")
        UserProfile.objects.create(user=self.user, org=org, group=UserGroup.ORG_ADMIN.value)
        self.client.force_login(self.user)

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_user_key_create_without_expiration_uses_maximum(self):
        before = timezone.now()
        response = self.client.post(reverse("token_create"), {"name": "key", "expires_at": ""})
        self.assertEqual(response.status_code, 302)
        token = Token.objects.get(user=self.user)
        self.assertGreaterEqual(token.expires_at, before + timedelta(days=7))
        self.assertLessEqual(token.expires_at, timezone.now() + timedelta(days=7))

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_user_key_create_rejects_expiration_beyond_maximum(self):
        far_future = (timezone.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        response = self.client.post(
            reverse("token_create"), {"name": "late", "expires_at": far_future}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Expiration date cannot be later than")
        self.assertContains(response, "the maximum of 7 days from now")
        self.assertFalse(Token.objects.filter(user=self.user, name="late").exists())

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_user_key_edit_rejects_expiration_beyond_maximum(self):
        token = Token(name="key", user=self.user)
        token._set_new_key()
        token.save()
        original_expiration = token.expires_at

        far_future = (timezone.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        response = self.client.post(
            reverse("token_edit", kwargs={"id": token.id}),
            {"name": "key", "expires_at": far_future},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Expiration date cannot be later than")
        token.refresh_from_db()
        self.assertEqual(token.expires_at, original_expiration)

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_user_key_accepts_expiration_within_maximum(self):
        expiration = (timezone.now() + timedelta(days=2)).replace(second=0, microsecond=0)
        response = self.client.post(
            reverse("token_create"),
            {"name": "key", "expires_at": expiration.strftime("%Y-%m-%dT%H:%M")},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Token.objects.get(user=self.user).expires_at, expiration)

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_token_create_form_shows_maximum_expiration(self):
        response = self.client.get(reverse("token_create"))
        self.assertContains(response, "A key can last at most 7 days.")
        self.assertContains(response, ' max="')

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=-1)
    def test_token_create_form_omits_maximum_expiration_when_unlimited(self):
        response = self.client.get(reverse("token_create"))
        self.assertNotContains(response, "A key can last at most")
        self.assertNotContains(response, ' max="')

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

    @override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=7)
    def test_service_account_create_rejects_expiration_beyond_maximum(self):
        far_future = (timezone.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        response = self.client.post(
            reverse("service_account_create", kwargs={"id": self.team.id}),
            {"name": "late-service", "description": "", "token_expires_at": far_future},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Expiration date cannot be later than")
        self.assertFalse(ServiceAccount.objects.filter(name="late-service").exists())

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
