import re
from datetime import UTC, datetime

from django.test import TestCase, modify_settings, override_settings
from django.urls import reverse
from tos.models import TermsOfService

from aqueduct.urls import terms_of_service
from aqueduct.urls import urlpatterns as aqueduct_urlpatterns
from management.tests.helpers import RoleWorld

urlpatterns = [terms_of_service, *aqueduct_urlpatterns]


class SignInPageTest(TestCase):
    def test_says_what_aqueduct_is_and_offers_sign_in(self):
        page = self.client.get(reverse("sso"))
        self.assertContains(page, ">One API key for your models.</h1>")
        self.assertContains(page, "through one OpenAI-compatible API")
        self.assertContains(page, ">Sign in with SSO</a>")
        self.assertContains(page, "<title>Sign in · Aqueduct AI Gateway</title>")

    def test_hands_a_next_on_this_site_to_the_sign_in_button(self):
        init = reverse("oidc_authentication_init")
        page = self.client.get(reverse("sso"), {"next": "/aqueduct/management/requests/?span=1d"})
        self.assertContains(page, f'href="{init}?next=/aqueduct/management/requests/%3Fspan%3D1d"')

    def test_drops_a_next_that_leaves_the_site(self):
        init = reverse("oidc_authentication_init")
        for unsafe in ("https://evil.example", "//evil.example", "/\\evil.example"):
            with self.subTest(next=unsafe):
                page = self.client.get(reverse("sso"), {"next": unsafe})
                self.assertContains(page, f'href="{init}"')
                self.assertNotContains(page, "evil.example")


@override_settings(ROOT_URLCONF=__name__)
@modify_settings(MIDDLEWARE={"append": "management.middleware.UserAgreementMiddleware"})
class TermsPromptTest(RoleWorld):
    def setUp(self):
        self.terms = TermsOfService.objects.create(active=True, content="<p>Be kind.</p>")
        TermsOfService.objects.filter(pk=self.terms.pk).update(
            created=datetime(2026, 10, 17, 9, tzinfo=UTC)
        )
        self.client.force_login(self.solo)

    def accept(self, next_url: str):
        self.client.get(reverse("tokens"))
        return self.client.post(reverse("tos_check_tos"), {"accept": "accept", "next": next_url})

    def test_prompt_has_the_logo_and_footer_but_no_navigation(self):
        page = self.client.get(reverse("tos_check_tos"), {"next": reverse("settings")})
        self.assertContains(page, ">Accept the terms to continue</h1>")
        self.assertContains(page, "Version of <time")
        self.assertContains(page, ">17 Oct 2026</time>")
        self.assertContains(page, "Be kind.")
        self.assertContains(
            page, 'name="accept" value="accept" class="btn btn-primary">Accept</button>'
        )
        self.assertContains(
            page, f'formaction="{reverse("oidc_logout")}">Decline and sign out</button>'
        )
        self.assertContains(page, "Aqueduct</span>")
        self.assertContains(page, "<footer")
        self.assertNotContains(page, "drawer-side")

    def test_accepting_opens_the_page_that_asked(self):
        page = f"{reverse('requests')}?span=1d"
        self.assertRedirects(self.accept(page), page, fetch_redirect_response=False)
        self.assertEqual(self.client.get(reverse("settings")).status_code, 200)

    def test_accepting_never_leaves_the_site(self):
        for target in (
            "https://evil.example",
            "//evil.example",
            "/\\evil.example",
            "http:evil.example",
            "",
        ):
            with self.subTest(target=target):
                self.assertRedirects(
                    self.accept(target), reverse("tokens"), fetch_redirect_response=False
                )

    def test_accepting_with_a_javascript_url_lands_on_api_keys(self):
        self.assertRedirects(
            self.accept("javascript:alert(1)"), reverse("tokens"), fetch_redirect_response=False
        )

    def test_a_first_sign_in_goes_through_the_terms_to_the_first_visit_on_api_keys(self):
        self.client.force_login(self.user("newcomer"))
        prompt = self.client.get(reverse("tokens"), follow=True)
        self.assertContains(prompt, ">Accept the terms to continue</h1>")
        landing = self.client.post(
            reverse("tos_check_tos"), {"accept": "accept", "next": reverse("tokens")}, follow=True
        )
        self.assertEqual(landing.redirect_chain[-1][0], reverse("tokens"))
        self.assertContains(landing, "Create your first API key")

    def test_declining_signs_out(self):
        page = self.client.get(reverse("tos_check_tos"), {"next": reverse("settings")})
        decline = re.search(r'formaction="([^"]+)">Decline and sign out', page.content.decode())
        self.client.post(decline.group(1), {"next": reverse("settings")})
        response = self.client.get(reverse("settings"))
        self.assertRedirects(
            response, f"{reverse('sso')}?next={reverse('settings')}", fetch_redirect_response=False
        )

    def test_terms_page_shows_the_version_in_the_app_date_format(self):
        self.client.logout()
        page = self.client.get(reverse("tos"))
        self.assertContains(page, ">Terms of service</h1>")
        self.assertContains(page, ">17 Oct 2026</time>")
