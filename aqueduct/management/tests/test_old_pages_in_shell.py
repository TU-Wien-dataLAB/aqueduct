from http import HTTPStatus

from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.models import Org, ServiceAccount, Team, UserGroup
from management.tests.helpers import join, make_token, make_user

ROLES = ("platform admin", "org admin", "team admin", "member")

M = "/aqueduct/management/"
TOKENS = M + "tokens/"
ORG = M + "org/"
TEAM = M + "team/{team}"
TEAM_CREATE = M + "team/create/"
TEAM_EDIT = TEAM + "/edit/"
TEAM_DELETE = TEAM + "/delete/"
TEAM_MEMBERS = TEAM + "/members/"
SA_ADD = TEAM + "/service-accounts/add/"
SA_EDIT = M + "service-accounts/{service_account}/edit/"
SA_DELETE = M + "service-accounts/{service_account}/delete/"
SA_TRANSFER = M + "service-accounts/{service_account}/transfer/"

PAGES = [
    TOKENS,
    TOKENS + "create/",
    TOKENS + "{token}/edit/",
    TOKENS + "{token}/delete/",
    ORG,
    TEAM_CREATE,
    TEAM,
    TEAM_EDIT,
    TEAM_DELETE,
    TEAM_MEMBERS,
    SA_ADD,
    SA_EDIT,
    SA_DELETE,
    SA_TRANSFER,
    M + "models/",
    M + "usage/",
    M + "usage/?token={token}",
    M + "files/",
    M + "batches/",
    M + "mcp-servers/",
    M + "vector-stores/",
    M + "vector-stores/vs_proto/",
]

# A page missing here answers 200 without a redirect.
OTHER_ANSWERS: dict[tuple[str, str], tuple[int, str]] = {
    **{(role, TEAM_MEMBERS): (302, TEAM) for role in ROLES},
    ("team admin", TEAM_CREATE): (404, ""),
    ("team admin", TEAM_DELETE): (404, ""),
    ("member", TEAM_CREATE): (404, ""),
    ("member", TEAM_EDIT): (404, ""),
    ("member", TEAM_DELETE): (404, ""),
    ("member", SA_ADD): (404, ""),
    ("member", SA_EDIT): (404, ""),
    ("member", SA_DELETE): (404, ""),
    ("member", SA_TRANSFER): (404, ""),
}


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class OldPagesInShellTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        org = Org.objects.create(name="E060")
        cls.team = Team.objects.create(name="Whale", org=org)
        cls.service_account = ServiceAccount.objects.create(name="TUna", team=cls.team)
        cls.users = {
            "platform admin": make_user("padmin", org, UserGroup.ADMIN),
            "org admin": make_user("oadmin", org, UserGroup.ORG_ADMIN),
            "team admin": make_user("tadmin", org),
            "member": make_user("member", org),
        }
        for role, user in cls.users.items():
            join(user, cls.team, is_admin=role == "team admin")
        cls.tokens = {role: make_token(user, f"{role} key") for role, user in cls.users.items()}
        make_token(cls.users["platform admin"], "TUna key", service_account=cls.service_account)

    def url(self, role: str, template: str) -> str:
        return template.format(
            token=self.tokens[role].pk, team=self.team.pk, service_account=self.service_account.pk
        )

    def answers(self):
        for role in ROLES:
            self.client.force_login(self.users[role])
            for template in PAGES:
                yield role, template, self.client.get(self.url(role, template))

    def test_every_page_gives_the_recorded_answer(self):
        for role, template, response in self.answers():
            expected_status, location = OTHER_ANSWERS.get((role, template), (200, ""))
            with self.subTest(role=role, page=template):
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(
                    response.get("Location", ""), self.url(role, location) if location else ""
                )

    def test_every_page_that_renders_sits_in_the_shell(self):
        for role, template, response in self.answers():
            if response.status_code == HTTPStatus.OK:
                with self.subTest(role=role, page=template):
                    self.assertTemplateUsed(response, "management/_sidebar.html")
                    self.assertContains(response, 'href="{}"'.format(reverse("tokens")))
