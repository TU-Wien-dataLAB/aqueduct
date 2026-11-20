from datetime import timedelta
from itertools import product
from unittest import mock

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.access import EveryOrg, KeyAct, OrgWide, TeamAct, Viewer, WholeOrg, requests_of
from management.models import (
    Org,
    Request,
    ServiceAccount,
    Team,
    TeamMembership,
    Token,
    UserGroup,
    UserProfile,
)
from management.ownership import RELATED, org_id
from management.tests.helpers import CHAT, join, make_token, make_user

User = get_user_model()


class Matrix(TestCase):
    """north has teams north-a and north-b, south has south-a. Every user has a personal key
    and every team a service-account key. `stale_admin` and `stale_member` keep a membership
    in south-a, the state an org move leaves behind; `moved` created south-a's key in south
    and now lives in north; `orphan` has no profile."""

    @classmethod
    def setUpTestData(cls):
        cls.north = Org.objects.create(name="north")
        cls.south = Org.objects.create(name="south")
        cls.north_a = Team.objects.create(name="north-a", org=cls.north)
        cls.north_b = Team.objects.create(name="north-b", org=cls.north)
        cls.south_a = Team.objects.create(name="south-a", org=cls.south)

        cls.admin = make_user("admin", cls.north, UserGroup.ADMIN)
        cls.org_admin = make_user("org-admin", cls.north, UserGroup.ORG_ADMIN)
        cls.stale_admin = make_user("stale-admin", cls.north, UserGroup.ORG_ADMIN)
        cls.member = make_user("member", cls.north)
        cls.stale_member = make_user("stale-member", cls.north)
        cls.moved = make_user("moved", cls.north)
        cls.southerner = make_user("southerner", cls.south)
        cls.orphan = User.objects.create_user(username="orphan", email="orphan@example.com")
        join(cls.admin, cls.north_a)
        join(cls.org_admin, cls.north_a)
        join(cls.stale_admin, cls.south_a, is_admin=True)
        join(cls.member, cls.north_a)
        join(cls.member, cls.north_b, is_admin=True)
        join(cls.stale_member, cls.north_a)
        join(cls.stale_member, cls.south_a)
        join(cls.southerner, cls.south_a)

        for user in User.objects.all():
            make_token(user, f"{user.username}-laptop")
        for team, creator in (
            (cls.north_a, cls.member),
            (cls.north_b, cls.member),
            (cls.south_a, cls.moved),
        ):
            account = ServiceAccount.objects.create(name=f"{team.name}-bot", team=team)
            make_token(creator, f"{team.name}-bot", service_account=account)
        cls.south_bot = Token.objects.get(name="south-a-bot")

    def viewers(self) -> list[Viewer]:
        profiles = UserProfile.objects.select_related("org", "user").order_by("user__username")
        found = [Viewer.of(profile) for profile in profiles]
        admin = UserProfile.objects.select_related("org", "user").get(user=self.admin)
        choices: tuple[OrgWide, ...] = (WholeOrg(self.south), EveryOrg())
        return found + [Viewer.of(admin, lambda _own, c=chosen: c) for chosen in choices]

    def tokens(self) -> list[Token]:
        return list(Token.objects.select_related(*RELATED))

    def names(self, tokens) -> set[str]:
        return set(tokens.values_list("name", flat=True))


class KeyScopeAgreementTest(Matrix):
    def test_every_key_act_answers_the_same_in_sql_and_in_python(self):
        tokens = self.tokens()
        for viewer, act in product(self.viewers(), KeyAct):
            with self.subTest(viewer=viewer.profile.user.username, chosen=viewer.chosen, act=act):
                in_sql = set(viewer.keys(act).values_list("id", flat=True))
                with self.assertNumQueries(0):
                    in_python = {token.id for token in tokens if viewer.may_key(act, token)}
                self.assertEqual(in_sql, in_python)

    def test_team_and_org_acts_agree_too(self):
        teams, orgs = list(Team.objects.all()), list(Org.objects.all())
        for viewer in self.viewers():
            for act in TeamAct:
                with self.subTest(viewer=viewer.profile.user.username, act=act):
                    in_sql = set(viewer.teams(act).values_list("id", flat=True))
                    self.assertEqual(in_sql, {t.id for t in teams if viewer.may_team(act, t)})
            in_sql = set(viewer.orgs().values_list("id", flat=True))
            self.assertEqual(in_sql, {o.id for o in orgs if viewer.administers(o)})

    def test_the_acts_nest(self):
        nested = (
            (KeyAct.SEE, KeyAct.OPEN),
            (KeyAct.OWN, KeyAct.MINE),
            (KeyAct.MINE, KeyAct.OPEN),
            (KeyAct.OWN, KeyAct.MANAGE),
            (KeyAct.MANAGE, KeyAct.OPEN),
        )
        for viewer, (inner, outer) in product(self.viewers(), nested):
            with self.subTest(viewer=viewer.profile.user.username, inner=inner, outer=outer):
                self.assertLessEqual(self.names(viewer.keys(inner)), self.names(viewer.keys(outer)))

    def test_a_listed_org_is_exactly_that_orgs_keys(self):
        tokens = self.tokens()
        for viewer in self.viewers():
            shown, listed = viewer.shown, set(viewer.keys(KeyAct.SEE).values_list("id", flat=True))
            with self.subTest(viewer=viewer.profile.user.username, shown=shown):
                if isinstance(shown, WholeOrg):
                    self.assertEqual(listed, {t.id for t in tokens if org_id(t) == shown.org.pk})
                elif isinstance(shown, EveryOrg):
                    self.assertEqual(listed, {t.id for t in tokens})

    def test_a_stale_membership_grants_nothing(self):
        for user in (self.stale_admin, self.stale_member):
            viewer = Viewer.of(user.profile)
            with self.subTest(user=user.username):
                for act in KeyAct:
                    self.assertNotIn("south-a-bot", self.names(viewer.keys(act)))
                for act in TeamAct:
                    self.assertNotIn(self.south_a, viewer.teams(act))

    def test_each_rank_administers_its_orgs(self):
        for user, expected in (
            (self.admin, {self.north, self.south}),
            (self.org_admin, {self.north}),
            (self.stale_admin, {self.north}),
            (self.member, set()),
        ):
            viewer = Viewer.of(user.profile)
            with self.subTest(user=user.username):
                self.assertEqual(set(viewer.orgs()), expected)
                self.assertEqual(
                    {o for o in (self.north, self.south) if viewer.administers(o)}, expected
                )

    def test_each_rank_reaches_its_keys(self):
        everything = self.names(Token.objects.all())
        north = everything - {"southerner-laptop", "south-a-bot", "orphan-laptop"}
        for user, act, expected in (
            (self.admin, KeyAct.OPEN, everything),
            (self.admin, KeyAct.SEE, north),
            (self.org_admin, KeyAct.OPEN, north),
            (self.org_admin, KeyAct.MANAGE, {"org-admin-laptop", "north-a-bot", "north-b-bot"}),
            (self.member, KeyAct.OPEN, {"member-laptop", "north-a-bot", "north-b-bot"}),
            (self.member, KeyAct.MANAGE, {"member-laptop", "north-b-bot"}),
            (self.member, KeyAct.OWN, {"member-laptop"}),
            (self.stale_member, KeyAct.OPEN, {"stale-member-laptop", "north-a-bot"}),
            (self.moved, KeyAct.OPEN, {"moved-laptop"}),
        ):
            with self.subTest(user=user.username, act=act):
                self.assertEqual(self.names(Viewer.of(user.profile).keys(act)), expected)


class OrgSwitchTest(Matrix):
    def test_a_switched_platform_admin_keeps_their_teams_but_a_leftover_row_grants_nothing(self):
        profile = UserProfile.objects.select_related("org", "user").get(user=self.admin)
        admin = Viewer.of(profile, lambda _own: WholeOrg(self.south))
        self.assertEqual(admin.shown, WholeOrg(self.south))
        self.assertEqual(admin.member_of, {self.north_a.pk})
        self.assertEqual(self.names(admin.keys(KeyAct.MINE)), {"admin-laptop", "north-a-bot"})

        stale = Viewer.of(
            UserProfile.objects.select_related("org", "user").get(user=self.stale_member)
        )
        self.assertNotIn("south-a-bot", self.names(stale.keys(KeyAct.MINE)))
        self.assertEqual(stale.member_of, {self.north_a.pk})


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class CrossOrgRuleTest(Matrix):
    def setUp(self):
        super().setUp()
        sent = timezone.now() - timedelta(hours=1)
        Request.objects.bulk_create(
            Request(token=token, path=f"/{token.name}", model=CHAT, timestamp=sent, status_code=200)
            for token in Token.objects.all()
        )

    def listed(self, page: str, **params) -> set[str]:
        response = self.client.get(reverse(page), params)
        self.assertEqual(response.status_code, 200)
        if page == "requests":
            return {row.request.token.name for row in response.context["log"].rows}
        if page == "usage_keys":
            return {key.name for key in response.context["keys"]}
        return {row.token.name for row in response.context["keys"].rows}

    def test_a_stale_membership_opens_nothing_and_lists_nothing(self):
        bot = self.south_bot
        for user in (self.stale_admin, self.stale_member):
            self.client.force_login(user)
            with self.subTest(user=user.username):
                for page in ("requests", "usage_keys", "tokens"):
                    self.assertNotIn(bot.name, self.listed(page))
                for url in (
                    reverse("token_detail", args=[bot.id]),
                    reverse("team", args=[self.south_a.id]),
                    f"{reverse('requests')}?key={bot.id}",
                    f"{reverse('usage')}?key={bot.id}",
                ):
                    self.assertEqual(self.client.get(url).status_code, 404, url)
                before = Token.objects.values().get(id=bot.id)
                response = self.client.post(reverse("token_regenerate", args=[bot.id]))
                self.assertEqual(response.status_code, 404)
                self.assertEqual(Token.objects.values().get(id=bot.id), before)

    def test_the_key_page_and_the_log_agree_for_every_viewer(self):
        for user in (
            self.org_admin,
            self.stale_admin,
            self.member,
            self.stale_member,
            self.moved,
            self.southerner,
        ):
            self.client.force_login(user)
            listed = self.listed("requests")
            with self.subTest(user=user.username):
                for token in Token.objects.all():
                    status = self.client.get(reverse("token_detail", args=[token.id])).status_code
                    self.assertEqual(status, 200 if token.name in listed else 404, token.name)


class RequestScopeTest(Matrix):
    def test_both_branches_of_the_id_list_lever_return_the_same_requests(self):
        sent = timezone.now() - timedelta(hours=1)
        Request.objects.bulk_create(
            Request(token=token, path=f"/{token.name}", model=CHAT, timestamp=sent)
            for token in Token.objects.all()
        )
        for viewer in self.viewers():
            expected = set(
                Request.objects.filter(token__in=viewer.keys(KeyAct.SEE)).values_list(
                    "id", flat=True
                )
            )
            with self.subTest(viewer=viewer.profile.user.username, shown=viewer.shown):
                named = requests_of(viewer.shown)
                self.assertEqual(set(named.values_list("id", flat=True)), expected)
                with mock.patch("management.access._LISTED_KEYS", 0):
                    many = requests_of(viewer.shown)
                self.assertEqual(set(many.values_list("id", flat=True)), expected)
                if not isinstance(viewer.shown, EveryOrg):
                    self.assertNotIn("management_token", str(named.query))
                    self.assertIn("management_token", str(many.query))


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class OneResolutionPerRequestTest(Matrix):
    def test_a_platform_admins_page_resolves_the_viewer_once(self):
        self.client.force_login(self.admin)
        with CaptureQueriesContext(connection) as captured:
            response = self.client.get(reverse("usage"), {"org": self.south.id})
        self.assertEqual(response.status_code, 200)
        memberships = TeamMembership._meta.db_table
        chosen = f'"{Org._meta.db_table}"."id" = {self.south.id}'
        sql = [query["sql"] for query in captured]
        self.assertEqual(sum(1 for q in sql if f'FROM "{memberships}"' in q and "is_admin" in q), 1)
        self.assertEqual(sum(1 for q in sql if chosen in q and "LIMIT 1" in q), 1)


class RankTest(TestCase):
    def setUp(self):
        self.profile = make_user("ana", Org.objects.create(name="north")).profile

    def test_the_highest_group_decides(self):
        self.profile.user.groups.set(Group.objects.filter(name__in=list(UserGroup.values)))
        self.assertEqual(self.profile.rank, UserGroup.ADMIN)

    def test_a_user_without_a_group_ranks_as_a_user(self):
        self.profile.user.groups.clear()
        self.assertEqual(self.profile.rank, UserGroup.USER)
