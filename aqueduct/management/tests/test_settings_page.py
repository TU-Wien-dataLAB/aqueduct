from django.test import TestCase, override_settings
from django.urls import reverse
from tos.models import TermsOfService, UserAgreement

from aqueduct.urls import terms_of_service
from aqueduct.urls import urlpatterns as aqueduct_urlpatterns
from management.models import Org, Team, TeamMembership, UserGroup
from management.tests.helpers import make_token, make_user

urlpatterns = [terms_of_service, *aqueduct_urlpatterns]


class SettingsPageTest(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="Informatics", requests_per_minute=600)
        self.nlp = Team.objects.create(name="NLP Lab", org=self.org)
        self.teaching = Team.objects.create(name="Teaching", org=self.org)
        self.member = self.user("member")
        TeamMembership.objects.create(
            user_profile=self.member.profile, team=self.nlp, is_admin=True
        )
        TeamMembership.objects.create(user_profile=self.member.profile, team=self.teaching)
        self.other_org = Org.objects.create(name="Elsewhere")
        self.outsider = self.user("outsider", org=self.other_org)
        TeamMembership.objects.create(
            user_profile=self.outsider.profile,
            team=Team.objects.create(name="Foreign", org=self.other_org),
        )
        make_token(self.outsider, "outsider-key")

    def user(self, name: str, group: UserGroup = UserGroup.USER, org: Org | None = None):
        return make_user(name, org or self.org, group)

    def page(self, user):
        self.client.force_login(user)
        return self.client.get(reverse("settings"))

    def test_shows_the_viewers_account(self):
        response = self.page(self.user("boss", UserGroup.ORG_ADMIN))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "boss@example.com")
        self.assertContains(response, "<dd>Informatics</dd>", html=True)
        self.assertContains(response, "Org Admin</span>")
        self.assertContains(
            response, '<span class="muted">You are not in a team.</span>', html=True
        )
        self.assertContains(response, "Set by your sign-in provider.")
        for gone in ("Sign-in</dt>", "Appearance", "e-mail"):
            self.assertNotContains(response, gone)

    def test_admin_viewing_another_org_sees_their_own(self):
        self.client.force_login(self.user("root", UserGroup.ADMIN))
        response = self.client.get(reverse("settings"), {"org": self.other_org.id})
        self.assertContains(response, "<dd>Informatics</dd>", html=True)
        self.assertNotContains(response, "<dd>Elsewhere</dd>", html=True)

    def test_lists_only_the_viewers_teams_with_role_and_link(self):
        response = self.page(self.member)
        nlp, teaching = (
            reverse("team", args=[self.nlp.id]),
            reverse("team", args=[self.teaching.id]),
        )
        self.assertContains(response, f'<a class="link-plain" href="{nlp}">NLP Lab</a>', html=True)
        self.assertContains(response, f'href="{teaching}"')
        self.assertRegex(
            response.content.decode(), r"NLP Lab</a>\s*<span class=\"muted\">Team admin"
        )
        self.assertRegex(response.content.decode(), r"Teaching</a>\s*<span class=\"muted\">Member")
        self.assertNotContains(response, "Foreign")
        self.assertNotContains(response, "outsider@example.com")

    def test_counts_only_the_viewers_personal_keys(self):
        make_token(self.member, "laptop")
        with self.settings(MAX_USER_TOKENS=5):
            response = self.page(self.member)
        self.assertContains(response, "<dd>1 of 5</dd>", html=True)

    def test_terms_card_hidden_without_the_terms_route(self):
        self.assertNotContains(self.page(self.member), "Terms of service</h2>")

    @override_settings(ROOT_URLCONF=__name__)
    def test_terms_card_shows_when_the_viewer_accepted_the_current_terms(self):
        old = TermsOfService.objects.create(active=True, content="old")
        current = TermsOfService.objects.create(active=True, content="new")
        response = self.page(self.member)
        self.assertContains(response, "Terms of service</h2>")
        self.assertContains(response, ">Not accepted</span>")

        UserAgreement.objects.create(terms_of_service=old, user=self.member)
        self.assertContains(self.page(self.member), ">Not accepted</span>")

        agreement = UserAgreement.objects.create(terms_of_service=current, user=self.member)
        accepted = agreement.created
        response = self.page(self.member)
        self.assertContains(
            response,
            "Accepted "
            f'<time datetime="{accepted.isoformat()}" title="{accepted:%Y-%m-%d %H:%M:%S} UTC">'
            f"{accepted.day} {accepted:%b %Y}</time></span>",
        )
        self.assertContains(response, f'href="{reverse("tos")}"')
