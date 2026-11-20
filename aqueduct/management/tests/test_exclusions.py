from itertools import product

from django.test import TestCase, override_settings

from management.models import Org, ServiceAccount, Team, Token
from management.models.exclusions import Exclusion, excluded
from management.tests.helpers import make_token, make_user

FIELDS = {
    Exclusion.MODELS: ("excluded_models", "merge_exclusion_lists"),
    Exclusion.MCP_SERVERS: ("excluded_mcp_servers", "merge_mcp_server_exclusion_lists"),
}


@override_settings(
    AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST=["default"],
    AQUEDUCT_DEFAULT_MCP_SERVER_EXCLUSION_LIST=["default"],
)
class ExclusionChainTest(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="org")
        self.team = Team.objects.create(name="team", org=self.org)
        self.member = make_user("member", self.org)
        self.profile = self.member.profile
        self.personal_key = make_token(self.member, "laptop")
        bot = ServiceAccount.objects.create(name="bot", team=self.team)
        self.team_key = make_token(self.member, "bot", service_account=bot)

    def configure(self, kind: Exclusion, org_merges: bool, holder_merges: bool) -> None:
        names, merges = FIELDS[kind]
        for obj, name, merging in (
            (self.org, "org", org_merges),
            (self.team, "team", holder_merges),
            (self.profile, "profile", holder_merges),
        ):
            setattr(obj, names, [name])
            setattr(obj, merges, merging)
            obj.save()

    def key_exclusions(self, token: Token, kind: Exclusion) -> frozenset[str]:
        fresh = Token.objects.get(pk=token.pk)
        if kind is Exclusion.MODELS:
            return fresh.model_exclusion_list()
        return fresh.mcp_server_exclusion_list()

    def test_each_level_adds_its_parent_only_while_it_merges(self):
        for kind, org_merges, holder_merges in product(Exclusion, (False, True), (False, True)):
            with self.subTest(kind=kind.name, org_merges=org_merges, holder_merges=holder_merges):
                self.configure(kind, org_merges, holder_merges)
                org_level = {"org", "default"} if org_merges else {"org"}
                inherited = org_level if holder_merges else set()
                self.assertEqual(excluded(self.org, kind), org_level)
                self.assertEqual(excluded(self.team, kind), {"team", *inherited})
                self.assertEqual(excluded(self.profile, kind), {"profile", *inherited})
                self.assertEqual(self.key_exclusions(self.team_key, kind), {"team", *inherited})
                self.assertEqual(
                    self.key_exclusions(self.personal_key, kind), {"profile", *inherited}
                )
