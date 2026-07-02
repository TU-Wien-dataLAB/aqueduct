"""Tests for the ``LimitMixin`` hourly/daily multiplier fields and validation.

Covers §3.3/§4.2 of the cache-RPM plan:
- ``LimitSet.from_objects`` resolves the two new fields (specific -> org -> None).
- ``LimitSet.windows()`` applies settings defaults and scales per-minute limits.
- Admin form fields for Org/Team/UserProfile accept and save the new fields.
- ``LimitMixin.clean()`` bounds validation surfaces via ``full_clean()`` for all
  three limit-bearing models (Org/Team inherit ``LimitMixin.clean()`` directly;
  ``UserProfile.clean()`` calls ``super().clean()``).
"""

from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings

from management.models import LimitSet, Org, Team, UserProfile

User = get_user_model()


class LimitSetResolutionTest(TestCase):
    def test_from_objects_resolves_new_fields(self):
        org = Org.objects.create(name="TestOrg")
        org.hourly_limit_multiplier = 6
        org.daily_limit_multiplier = 1440
        org.save(update_fields=["hourly_limit_multiplier", "daily_limit_multiplier"])

        team = Team.objects.create(name="TestTeam", org=org)
        team.hourly_limit_multiplier = 3  # overrides org
        team.daily_limit_multiplier = None  # falls back to org
        team.save(update_fields=["hourly_limit_multiplier", "daily_limit_multiplier"])

        limits = LimitSet.from_objects(team, org)
        self.assertEqual(limits.hourly_limit_multiplier, 3)  # team
        self.assertEqual(limits.daily_limit_multiplier, 1440)  # org fallback

    @override_settings(AQUEDUCT_HOURLY_LIMIT_MULTIPLIER=60, AQUEDUCT_DAILY_LIMIT_MULTIPLIER=1440)
    def test_windows_applies_settings_defaults_and_scales(self):
        org = Org.objects.create(name="TestOrg2", requests_per_minute=10)
        limits = LimitSet.from_objects(org, None)
        windows = limits.windows()
        self.assertEqual(windows[0], ("min", 60, 10, None, None))
        self.assertEqual(windows[1], ("hour", 3600, 600, None, None))  # 10 x 60
        self.assertEqual(windows[2], ("day", 86400, 14400, None, None))  # 10 x 1440

    def test_windows_uses_object_multiplier_over_settings(self):
        org = Org.objects.create(
            name="TestOrg3",
            requests_per_minute=10,
            hourly_limit_multiplier=6,
            daily_limit_multiplier=120,
        )
        limits = LimitSet.from_objects(org, None)
        windows = limits.windows()
        self.assertEqual(windows[1][2], 60)  # 10 x 6 (object value, not settings default)
        self.assertEqual(windows[2][2], 1200)  # 10 x 120


class LimitValidationTest(TestCase):
    def test_valid_multiplier_values_pass(self):
        for h, d in [(1, 1), (60, 1440), (None, None), (10, 240), (60, None), (None, 1440)]:
            org = Org(name=f"Org-{h}-{d}", hourly_limit_multiplier=h, daily_limit_multiplier=d)
            org.clean()  # should not raise

    def test_hourly_multiplier_out_of_range_raises(self):
        for bad in (0, 61):
            org = Org(name=f"OrgBadH{bad}", hourly_limit_multiplier=bad)
            with self.assertRaises(ValidationError) as ctx:
                org.clean()
            self.assertIn("hourly_limit_multiplier", ctx.exception.error_dict)

    def test_daily_multiplier_out_of_range_raises(self):
        for bad in (0, 1441):
            org = Org(name=f"OrgBadD{bad}", daily_limit_multiplier=bad)
            with self.assertRaises(ValidationError) as ctx:
                org.clean()
            self.assertIn("daily_limit_multiplier", ctx.exception.error_dict)

    def test_daily_exceeds_24x_hourly_raises(self):
        # hourly=10 -> 24*10=240; daily=241 is unreachable -> invalid
        org = Org(name="OrgInconsistent", hourly_limit_multiplier=10, daily_limit_multiplier=241)
        with self.assertRaises(ValidationError) as ctx:
            org.clean()
        self.assertIn("daily_limit_multiplier", ctx.exception.error_dict)

    def test_daily_equal_24x_hourly_passes(self):
        org = Org(name="OrgBoundary", hourly_limit_multiplier=10, daily_limit_multiplier=240)
        org.clean()  # should not raise

    def test_team_validation_runs(self):
        org = Org.objects.create(name="OrgForTeam")
        team = Team(name="TeamBad", org=org, hourly_limit_multiplier=61)
        with self.assertRaises(ValidationError) as ctx:
            team.clean()
        self.assertIn("hourly_limit_multiplier", ctx.exception.error_dict)

    def test_userprofile_validation_runs_despite_own_clean(self):
        """UserProfile.clean() calls super().clean(), so LimitMixin bounds apply."""
        org = Org.objects.create(name="OrgForProfile")
        user = User.objects.create_user(username="ValUser", email="val@example.com")
        profile = UserProfile(user=user, org=org, daily_limit_multiplier=1441)
        with self.assertRaises(ValidationError) as ctx:
            profile.clean()
        self.assertIn("daily_limit_multiplier", ctx.exception.error_dict)

    @override_settings(
        LITELLM_ROUTER_CONFIG_FILE_PATH=Path(__file__).resolve().parents[3]
        / "example_router_config.yaml"
    )
    def test_full_clean_surfaces_multiplier_error_via_admin_form(self):
        """The admin ModelForm runs full_clean(), surfacing the clean() bounds.

        The form supplies the JSON exclusion fields, so full_clean() reaches
        clean() (a bare model.full_clean() would trip on the blank default []).
        """
        from management.admin import OrgAdminForm

        form = OrgAdminForm(
            data={
                "name": "OrgViaForm",
                "requests_per_minute": "",
                "input_tokens_per_minute": "",
                "output_tokens_per_minute": "",
                "hourly_limit_multiplier": "61",  # out of range
                "daily_limit_multiplier": "",
                "excluded_models": "[]",
                "merge_exclusion_lists": "on",
                "excluded_mcp_servers": "[]",
                "merge_mcp_server_exclusion_lists": "on",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("hourly_limit_multiplier", form.errors)


class AdminFormFieldsTest(TestCase):
    """The new fields are editable in the Django admin forms (lightweight check)."""

    def test_fields_present_in_admin_forms(self):
        from management.admin import OrgAdminForm, TeamAdminForm, UserProfileAdminForm

        for form_class in (OrgAdminForm, TeamAdminForm, UserProfileAdminForm):
            fields = form_class.Meta.fields
            self.assertIn("hourly_limit_multiplier", fields)
            self.assertIn("daily_limit_multiplier", fields)
