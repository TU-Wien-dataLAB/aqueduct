from django.conf import settings
from django.test import TestCase, modify_settings, override_settings
from django.urls import reverse
from tos.models import TermsOfService

from aqueduct.urls import terms_of_service
from aqueduct.urls import urlpatterns as aqueduct_urlpatterns
from management.models import Org
from management.tests.helpers import make_user

urlpatterns = [terms_of_service, *aqueduct_urlpatterns]

UNSAFE_NEXT = (
    "https://evil.example",
    "//evil.example",
    "/\\evil.example",
    "http:evil.example",
    "javascript:alert(1)",
)


class SignInPageTest(TestCase):
    def setUp(self):
        self.client.force_login(make_user("ana", Org.objects.create(name="lab-org")))

    def test_a_signed_in_visit_follows_a_next_on_this_site(self):
        target = f"http://testserver{reverse('tokens')}?page=2"
        response = self.client.get(reverse("sso"), {"next": target})
        self.assertRedirects(response, target, fetch_redirect_response=False)

    def test_a_signed_in_visit_drops_a_next_that_leaves_the_site(self):
        for login in ("sso", "admin_sso"):
            for unsafe in UNSAFE_NEXT:
                with self.subTest(login=login, next=unsafe):
                    response = self.client.get(reverse(login), {"next": unsafe})
                    self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)


@override_settings(ROOT_URLCONF=__name__)
@modify_settings(MIDDLEWARE={"append": "management.middleware.UserAgreementMiddleware"})
class TermsPromptTest(TestCase):
    def setUp(self):
        TermsOfService.objects.create(active=True, content="<p>Be kind.</p>")
        self.client.force_login(make_user("ana", Org.objects.create(name="lab-org")))

    def accept(self, next_url: str):
        self.client.get(reverse("tokens"))
        return self.client.post(reverse("tos_check_tos"), {"accept": "accept", "next": next_url})

    def test_accepting_opens_the_page_that_asked(self):
        page = f"{reverse('tokens')}?page=2"
        self.assertRedirects(self.accept(page), page, fetch_redirect_response=False)

    def test_accepting_never_leaves_the_site(self):
        for target in (*UNSAFE_NEXT, ""):
            with self.subTest(target=target):
                self.assertRedirects(
                    self.accept(target), settings.LOGIN_REDIRECT_URL, fetch_redirect_response=False
                )
