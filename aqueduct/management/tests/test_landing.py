import re
from unittest import mock
from urllib.parse import parse_qs, urlparse

from django.test import override_settings
from django.urls import reverse

from management.tests.helpers import RoleWorld

OIDC = override_settings(
    OIDC_RP_CLIENT_ID="aqueduct", OIDC_OP_AUTHORIZATION_ENDPOINT="https://sso.example.com/authorize"
)

UNSAFE_NEXT = (
    "https://evil.example",
    "//evil.example",
    "/\\evil.example",
    "http:evil.example",
    "javascript:alert(1)",
    "",
)


@OIDC
class LandingTest(RoleWorld):
    def roles(self):
        return {
            "admin": self.admin,
            "org admin": self.org_admin,
            "team admin": self.team_admin,
            "team member": self.member,
            "user": self.solo,
        }

    def sign_in(self, user, **params):
        self.client.logout()
        login = self.client.get(reverse("sso"), params)
        init = re.search(r'href="([^"]*)" class="btn btn-primary', login.content.decode())
        assert init is not None, "no login button"
        provider = self.client.get(init.group(1).replace("&amp;", "&"))
        state = parse_qs(urlparse(provider["Location"]).query)["state"][0]
        user.backend = "django.contrib.auth.backends.ModelBackend"
        with mock.patch("mozilla_django_oidc.views.auth.authenticate", return_value=user):
            return self.client.get(
                reverse("oidc_authentication_callback"), {"code": "c", "state": state}
            )

    def test_signed_in_visit_to_the_login_page_follows_a_next_on_this_site(self):
        self.client.force_login(self.solo)
        page = f"{reverse('requests')}?span=1d&status=500"
        self.assertRedirects(
            self.client.get(reverse("sso"), {"next": page}), page, fetch_redirect_response=False
        )

    def test_site_root_opens_api_keys(self):
        self.client.force_login(self.solo)
        self.assertRedirects(self.client.get("/"), reverse("tokens"), status_code=301)
