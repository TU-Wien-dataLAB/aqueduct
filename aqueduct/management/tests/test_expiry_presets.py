import re
from datetime import UTC, datetime, timedelta
from unittest import mock

from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from management.models import ServiceAccount, Token
from management.tests.helpers import RoleWorld, make_token

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


def fixed_clock():
    return mock.patch("management.expiry_presets.now", return_value=NOW)


def options(response) -> list[tuple[str, bool]]:
    """The expiry radios of a form page, in order, with whether each is checked."""
    radios = re.findall(
        r'<input type="radio" name="expiry_preset"[^>]*aria-label="([^"]*)"([^>]*)>',
        response.content.decode(),
    )
    return [(label, " checked" in rest) for label, rest in radios]


class KeyExpiryTest(RoleWorld):
    def create(self, **data):
        self.client.force_login(self.solo)
        with fixed_clock():
            self.client.post(reverse("token_create"), {"name": "fresh", **data})
        return Token.objects.filter(name="fresh").first()

    def test_each_preset_stores_its_lifetime_from_now(self):
        for preset, days in (("30d", 30), ("90d", 90), ("1y", 365)):
            with self.subTest(preset=preset):
                Token.objects.filter(name="fresh").delete()
                key = self.create(expiry_preset=preset)
                self.assertEqual(key.expires_at, NOW + timedelta(days=days))

    def test_never_stores_no_expiry_even_next_to_a_typed_date(self):
        key = self.create(expiry_preset="never", expires_at="2031-05-05T10:00")
        self.assertIsNone(key.expires_at)

    def test_pick_a_date_stores_the_typed_date(self):
        key = self.create(expiry_preset="date", expires_at="2031-05-05T10:00")
        self.assertEqual(key.expires_at, datetime(2031, 5, 5, 10, 0, tzinfo=UTC))

    def test_pick_a_date_without_a_date_is_refused(self):
        self.assertIsNone(self.create(expiry_preset="date", expires_at=""))
        response = self.client.post(
            reverse("token_create"), {"name": "fresh", "expiry_preset": "date", "expires_at": ""}
        )
        self.assertContains(response, "Pick a date, or choose another option.")

    def test_a_plain_date_without_presets_still_works(self):
        key = self.create(expires_at="2031-05-05T10:00")
        self.assertEqual(key.expires_at, datetime(2031, 5, 5, 10, 0, tzinfo=UTC))
        self.assertIsNone(Token.objects.get(name="solo-laptop").expires_at)

    def test_a_past_date_is_refused(self):
        self.client.force_login(self.solo)
        response = self.client.post(
            reverse("token_create"),
            {"name": "fresh", "expiry_preset": "date", "expires_at": "2001-01-01T00:00"},
        )
        self.assertContains(response, "Expiration date must be in the future.")
        self.assertFalse(Token.objects.filter(name="fresh").exists())

    def test_an_unknown_preset_is_refused(self):
        self.client.force_login(self.solo)
        response = self.client.post(
            reverse("token_create"), {"name": "fresh", "expiry_preset": "forever"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Token.objects.filter(name="fresh").exists())

    def test_the_form_offers_the_presets_with_never_chosen_and_says_what_follows(self):
        self.client.force_login(self.solo)
        with fixed_clock():
            page = self.client.get(reverse("token_create"))
        self.assertEqual(
            options(page),
            [
                ("30 days", False),
                ("90 days", False),
                ("1 year", False),
                ("Never", True),
                ("Pick a date", False),
            ],
        )
        self.assertContains(page, 'data-sentence="Expires 31 Jan 2030, 12:00 UTC"')
        self.assertContains(page, 'data-sentence="Expires 1 Apr 2030, 12:00 UTC"')
        self.assertContains(page, 'data-sentence="Expires 1 Jan 2031, 12:00 UTC"')
        self.assertContains(page, 'data-sentence="Never expires"')
        self.assertContains(page, 'name="expires_at"')

    def test_editing_a_key_with_an_expiry_shows_its_date_chosen(self):
        key = make_token(self.solo, "dated", expires_at=NOW + timedelta(days=400))
        self.client.force_login(self.solo)
        page = self.client.get(reverse("token_edit", args=[key.id]))
        self.assertEqual(options(page)[-1], ("Pick a date", True))
        self.assertContains(page, 'value="2031-02-05T12:00"')
        self.assertContains(page, "Expires 5 Feb 2031, 12:00 UTC")

    def test_editing_can_switch_a_key_to_a_lifetime_or_to_never(self):
        key = make_token(self.solo, "dated", expires_at=NOW + timedelta(days=400))
        self.client.force_login(self.solo)
        url = reverse("token_edit", args=[key.id])
        with fixed_clock():
            self.client.post(url, {"name": "dated", "expiry_preset": "90d"})
        key.refresh_from_db()
        self.assertEqual(key.expires_at, NOW + timedelta(days=90))
        self.client.post(url, {"name": "dated", "expiry_preset": "never"})
        key.refresh_from_db()
        self.assertIsNone(key.expires_at)


@override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=60)
class MaximumLifetimeTest(RoleWorld):
    def setUp(self):
        self.client.force_login(self.solo)

    def post(self, **data):
        return self.client.post(reverse("token_create"), {"name": "fresh", **data})

    def test_the_form_offers_only_lifetimes_within_the_maximum_and_names_it(self):
        with fixed_clock():
            page = self.client.get(reverse("token_create"))
        self.assertEqual(
            options(page), [("30 days", False), ("60 days", True), ("Pick a date", False)]
        )
        self.assertContains(page, "A key can last at most 60 days.")
        self.assertContains(page, 'data-sentence="Expires 2 Mar 2030, 12:00 UTC"')
        self.assertRegex(page.content.decode(), r'name="expires_at"[^>]* max="\d{4}-\d\d-\d\dT')

    def test_a_maximum_equal_to_a_preset_is_offered_once(self):
        with override_settings(MAX_ACCESS_KEY_EXPIRATION_DAYS=30):
            page = self.client.get(reverse("token_create"))
        self.assertEqual(options(page), [("30 days", True), ("Pick a date", False)])

    def test_the_longest_option_and_never_both_store_the_maximum(self):
        for preset in ("max", "never"):
            with self.subTest(preset=preset):
                Token.objects.filter(name="fresh").delete()
                before = timezone.now()
                self.post(expiry_preset=preset)
                expires_at = Token.objects.get(name="fresh").expires_at
                self.assertGreaterEqual(expires_at, before + timedelta(days=60))
                self.assertLessEqual(expires_at, timezone.now() + timedelta(days=60))

    def test_a_preset_beyond_the_maximum_is_refused_and_the_message_shows(self):
        response = self.post(expiry_preset="90d")
        self.assertContains(response, "Expiration date cannot be later than")
        self.assertFalse(Token.objects.filter(name="fresh").exists())
        self.assertEqual(
            options(response), [("30 days", False), ("60 days", False), ("Pick a date", True)]
        )


class ServiceAccountExpiryTest(RoleWorld):
    def create(self, **data):
        self.client.force_login(self.team_admin)
        with fixed_clock():
            self.client.post(
                reverse("service_account_create", args=[self.team.id]),
                {"name": "rag", "description": "", **data},
            )
        return ServiceAccount.objects.get(name="rag").token

    def test_a_plain_date_still_works(self):
        key = self.create(token_expires_at="2031-05-05T10:00")
        self.assertEqual(key.expires_at, datetime(2031, 5, 5, 10, 0, tzinfo=UTC))

    def test_editing_to_never_clears_the_expiry(self):
        self.pipeline_key.expires_at = NOW + timedelta(days=400)
        self.pipeline_key.save()
        self.client.force_login(self.team_admin)
        self.client.post(
            reverse("service_account_edit", args=[self.pipeline.id]),
            {"name": "pipeline", "description": "", "expiry_preset": "never"},
        )
        self.pipeline_key.refresh_from_db()
        self.assertIsNone(self.pipeline_key.expires_at)
