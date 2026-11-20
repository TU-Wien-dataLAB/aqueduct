from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.admin import set_user_group
from management.models import Org, UserProfile

User = get_user_model()


@override_settings(
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
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

    def test_staff_can_return_to_the_admin(self):
        self.user.is_staff = True
        self.user.save(update_fields=["is_staff"])
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                response = self.client.get(reverse(login), {"next": reverse("admin:index")})
                self.assertRedirects(response, reverse("admin:index"))

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

    def test_anonymous_visitors_still_see_the_sso_page(self):
        self.client.logout()
        for login in ("sso", "admin_sso"):
            with self.subTest(login=login):
                response = self.client.get(reverse(login), {"next": reverse("admin:index")})
                self.assertEqual(response.status_code, 200)
                self.assertTemplateUsed(response, "management/login.html")
