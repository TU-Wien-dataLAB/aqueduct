from django.contrib.auth import get_user_model
from django.test import TestCase

from management.models import FileObject, Org, ServiceAccount, Team, Token
from management.ownership import RELATED, key_org, of_orgs, org_id
from management.tests.helpers import make_token, make_user

User = get_user_model()


class KeyOrgTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.north = Org.objects.create(name="north")
        cls.south = Org.objects.create(name="south")
        cls.north_team = Team.objects.create(name="north-a", org=cls.north)
        cls.south_team = Team.objects.create(name="south-a", org=cls.south)
        cls.member = make_user("member", cls.north)
        cls.southerner = make_user("southerner", cls.south)
        cls.moved = make_user("moved", cls.north)
        cls.orphan = User.objects.create_user(username="orphan")
        cls.member_key = make_token(cls.member, "member-laptop")
        cls.southern_key = make_token(cls.southerner, "southern-laptop")
        cls.orphan_key = make_token(cls.orphan, "orphan-laptop")
        cls.north_bot = ServiceAccount.objects.create(name="north-bot", team=cls.north_team)
        cls.north_key = make_token(cls.member, "north-bot", service_account=cls.north_bot)
        cls.south_bot = ServiceAccount.objects.create(name="south-bot", team=cls.south_team)
        cls.south_key = make_token(cls.moved, "south-bot", service_account=cls.south_bot)

    def test_the_coalesce_the_filter_and_the_python_form_name_the_same_org(self):
        tokens = list(Token.objects.select_related(*RELATED))
        expected = {
            self.member_key.pk: self.north.pk,
            self.southern_key.pk: self.south.pk,
            self.orphan_key.pk: None,
            self.north_key.pk: self.north.pk,
            self.south_key.pk: self.south.pk,
        }
        annotated = dict(Token.objects.annotate(o=key_org()).values_list("id", "o"))
        self.assertEqual(annotated, expected)
        with self.assertNumQueries(0):
            self.assertEqual({token.pk: org_id(token) for token in tokens}, expected)
        for ids in ({self.north.pk}, {self.south.pk}, {self.north.pk, self.south.pk}):
            with self.subTest(orgs=ids):
                in_sql = set(Token.objects.filter(of_orgs(ids)).values_list("id", flat=True))
                self.assertEqual(in_sql, {pk for pk, org in expected.items() if org in ids})


class ReachTest(TestCase):
    def test_a_personal_key_reaches_only_its_own_and_a_team_key_its_teams(self):
        org = Org.objects.create(name="org")
        team = Team.objects.create(name="team", org=org)
        other_team = Team.objects.create(name="other", org=org)
        alice, bob = make_user("alice", org), make_user("bob", org)
        keys = {
            "alice": make_token(alice, "laptop"),
            "bob": make_token(bob, "laptop"),
            "alice_bot": make_token(
                alice, "bot", service_account=ServiceAccount.objects.create(name="a", team=team)
            ),
            "bob_bot": make_token(
                bob, "bot", service_account=ServiceAccount.objects.create(name="b", team=team)
            ),
            "other_bot": make_token(
                bob, "bot", service_account=ServiceAccount.objects.create(name="c", team=other_team)
            ),
        }
        for name, key in keys.items():
            FileObject.objects.create(
                id=name, bytes=1, created_at=0, filename=name, purpose="batch", token=key
            )

        def reached(name: str) -> set[str]:
            return set(FileObject.objects.filter(keys[name].reach()).values_list("id", flat=True))

        self.assertEqual(reached("alice"), {"alice"})
        self.assertEqual(reached("bob"), {"bob"})
        self.assertEqual(reached("alice_bot"), {"alice_bot", "bob_bot"})
        self.assertEqual(reached("other_bot"), {"other_bot"})
