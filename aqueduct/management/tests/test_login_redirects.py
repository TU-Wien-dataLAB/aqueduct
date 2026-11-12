from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from management.admin import set_user_group
from management.models import Org, UserProfile

User = get_user_model()


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class SSOLoginRedirectTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="login-user")
        UserProfile.objects.create(user=self.user, org=Org.objects.create(name="login-org"))
        set_user_group(User.objects.filter(pk=self.user.pk), "user")
        self.client.force_login(self.user)

    def test_non_admins_leave_admin_redirects_for_the_org_page(self):
        admin_index = reverse("admin:index")
        targets = (
            admin_index,
            admin_index.rstrip("/"),
            reverse("admin:auth_user_changelist"),
            f"http://testserver{admin_index}?view=all",
        )
        for group in ("user", "org-admin"):
            set_user_group(User.objects.filter(pk=self.user.pk), group)
            for login in ("sso", "admin_sso"):
                for target in targets:
                    with self.subTest(group=group, login=login, target=target):
                        response = self.client.get(reverse(login), {"next": target}, follow=True)
                        self.assertRedirects(response, reverse("org"))

    def test_make_user_action_does_not_trap_the_demoted_admin(self):
        set_user_group(User.objects.filter(pk=self.user.pk), "admin")
        response = self.client.post(
            reverse("admin:auth_user_changelist"),
            {"action": "make_user", "_selected_action": [str(self.user.pk)], "index": "0"},
            follow=True,
        )
        self.user.refresh_from_db()
        self.assertFalse(self.user.is_staff)
        self.assertFalse(self.user.is_superuser)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.redirect_chain[-1][0], reverse("org"))

    def test_staff_can_return_to_the_admin(self):
        self.user.is_staff = True
        self.user.save(update_fields=["is_staff"])
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                response = self.client.get(reverse(login), {"next": reverse("admin:index")})
                self.assertRedirects(response, reverse("admin:index"))

    def test_superuser_without_staff_access_leaves_the_admin(self):
        self.user.is_superuser = True
        self.user.save(update_fields=["is_superuser"])
        response = self.client.get(reverse("admin_sso"), {"next": reverse("admin:index")})
        self.assertRedirects(response, reverse("org"))

    def test_management_next_is_preserved(self):
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                target = reverse("tokens") + "?page=2"
                response = self.client.get(reverse(login), {"next": target})
                self.assertRedirects(response, target)

    def test_admin_prefix_lookalike_is_not_treated_as_the_admin(self):
        target = reverse("admin:index").rstrip("/") + "-help/"
        response = self.client.get(reverse("sso"), {"next": target})
        self.assertRedirects(response, target, fetch_redirect_response=False)

    def test_missing_next_uses_the_org_page(self):
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                self.assertRedirects(self.client.get(reverse(login)), reverse("org"))

    def test_anonymous_visitors_still_see_the_sso_page(self):
        self.client.logout()
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                response = self.client.get(reverse(login), {"next": reverse("admin:index")})
                self.assertEqual(response.status_code, 200)
                self.assertTemplateUsed(response, "management/login.html")
