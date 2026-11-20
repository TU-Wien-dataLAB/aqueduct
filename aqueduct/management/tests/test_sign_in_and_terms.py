
from django.test import TestCase
from django.urls import reverse

from aqueduct.urls import terms_of_service
from aqueduct.urls import urlpatterns as aqueduct_urlpatterns

urlpatterns = [terms_of_service, *aqueduct_urlpatterns]


class SignInPageTest(TestCase):

    def test_drops_a_next_that_leaves_the_site(self):
        init = reverse("oidc_authentication_init")
        for unsafe in ("https://evil.example", "//evil.example", "/\\evil.example"):
            with self.subTest(next=unsafe):
                page = self.client.get(reverse("sso"), {"next": unsafe})
                self.assertContains(page, f'href="{init}"')
                self.assertNotContains(page, "evil.example")
