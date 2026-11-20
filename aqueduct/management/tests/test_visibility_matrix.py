from datetime import timedelta

from django.urls import reverse
from django.utils import timezone

from management.models import Request, ServiceAccount, Token
from management.tests.helpers import CHAT, EMBEDDING, RoleWorld, join, make_token

LAB = {
    "/solo-laptop",
    "/member-laptop",
    "/member-embed",
    "/sent-as-solo",
    "/member-typo",
    "/pipeline",
    "/side-sa",
    "/org-admin-laptop",
}
FOREIGN = {"/foreign-laptop", "/foreign-sa", "/foreign-typo"}
MEMBER = {"/member-laptop", "/member-embed", "/sent-as-solo", "/member-typo"}
KEY_OF = {
    "/member-embed": "member-laptop",
    "/sent-as-solo": "member-laptop",
    "/member-typo": "member-laptop",
    "/foreign-typo": "foreign-laptop",
}
NOT_SERVED = "gpt-5"
NOT_FOUND = 404

ROWS = (
    ("solo", {}, {"/solo-laptop"}),
    ("solo", {"scope": "mine"}, {"/solo-laptop"}),
    ("solo", {"org": "all"}, {"/solo-laptop"}),
    ("solo", {"org": "other_org"}, {"/solo-laptop"}),
    ("solo", {"model": NOT_SERVED}, set()),
    ("solo", {"key": "solo-laptop"}, {"/solo-laptop"}),
    ("solo", {"key": "member-laptop"}, NOT_FOUND),
    ("solo", {"key": "pipeline"}, NOT_FOUND),
    ("solo", {"key": "foreign-laptop"}, NOT_FOUND),
    ("solo", {"key": "abc"}, NOT_FOUND),
    ("member", {}, MEMBER | {"/pipeline"}),
    ("member", {"model": EMBEDDING}, {"/member-embed"}),
    ("member", {"model": NOT_SERVED}, {"/member-typo"}),
    ("member", {"key": "pipeline"}, {"/pipeline"}),
    ("member", {"key": "side-sa"}, NOT_FOUND),
    ("member", {"key": "solo-laptop"}, NOT_FOUND),
    ("team_admin", {}, {"/pipeline"}),
    ("team_admin", {"key": "side-sa"}, NOT_FOUND),
    ("org_admin", {}, LAB),
    ("org_admin", {"scope": "mine"}, {"/org-admin-laptop"}),
    ("org_admin", {"model": NOT_SERVED}, {"/member-typo"}),
    ("org_admin", {"key": "member-laptop"}, MEMBER),
    ("org_admin", {"key": "side-sa", "model": CHAT}, {"/side-sa"}),
    ("org_admin", {"org": "all"}, LAB),
    ("org_admin", {"key": "foreign-laptop"}, NOT_FOUND),
    ("admin", {}, LAB),
    ("admin", {"scope": "mine"}, set()),
    ("admin", {"key": "foreign-sa"}, NOT_FOUND),
    ("admin", {"org": "all"}, LAB | FOREIGN),
    ("admin", {"org": "all", "key": "foreign-sa"}, {"/foreign-sa"}),
    ("admin", {"org": "other_org"}, FOREIGN),
    ("admin", {"org": "other_org", "key": "solo-laptop"}, NOT_FOUND),
    ("other_org_admin", {}, FOREIGN),
    ("other_org_admin", {"model": NOT_SERVED}, {"/foreign-typo"}),
    ("other_org_admin", {"key": "foreign-sa"}, {"/foreign-sa"}),
    ("other_org_admin", {"org": "all"}, FOREIGN),
    ("other_org_admin", {"org": "org"}, FOREIGN),
    ("other_org_admin", {"key": "solo-laptop"}, NOT_FOUND),
    ("other_org_admin", {"key": "pipeline"}, NOT_FOUND),
)

ALL_KEYS_REDIRECT_TO = {"solo": "solo-laptop", "team_admin": "pipeline"}

LAB_KEYS = {"solo-laptop", "member-laptop", "pipeline", "side-sa", "org-admin-laptop"}
FOREIGN_KEYS = {"foreign-laptop", "foreign-sa"}
OPENS = (
    ("solo", {"solo-laptop"}),
    ("member", {"member-laptop", "pipeline"}),
    ("team_admin", {"pipeline"}),
    ("org_admin", LAB_KEYS),
    ("admin", LAB_KEYS | FOREIGN_KEYS),
    ("other_org_admin", FOREIGN_KEYS),
)
VIEWED_ORGS = {"org_admin": ("all",), "admin": ("other_org", "all")}
# API keys lists a person's own keys and their teams' service accounts, whatever their rank.
OWN_KEYS = (
    ("solo", {"solo-laptop"}),
    ("member", {"member-laptop", "pipeline"}),
    ("team_admin", {"pipeline"}),
    ("org_admin", {"org-admin-laptop"}),
    ("admin", set()),
    ("other_org_admin", set()),
)


class VisibilityMatrixTest(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        side_account = ServiceAccount.objects.create(name="side-sa", team=cls.side_team)
        make_token(cls.team_admin, "side-sa", service_account=side_account)
        make_token(cls.org_admin, "org-admin-laptop")
        foreign_account = ServiceAccount.objects.create(name="foreign-sa", team=cls.foreign_team)
        make_token(cls.foreigner, "foreign-sa", service_account=foreign_account)
        sent = {"timestamp": timezone.now() - timedelta(hours=1), "status_code": 200}
        rows = [
            Request(token=token, path=f"/{token.name}", model=CHAT, **sent)
            for token in Token.objects.all()
        ]
        rows.append(Request(token=cls.member_key, path="/member-embed", model=EMBEDDING, **sent))
        rows.append(
            Request(
                token=cls.member_key,
                path="/sent-as-solo",
                model=CHAT,
                user_id="solo@example.com",
                **sent,
            )
        )
        for owner, path in ((cls.member_key, "/member-typo"), (cls.foreign_key, "/foreign-typo")):
            rows.append(Request(token=owner, path=path, model=NOT_SERVED, **sent))
        Request.objects.bulk_create(rows)

    def query(self, params: dict[str, str]) -> dict[str, object]:
        resolved: dict[str, object] = dict(params)
        if token := Token.objects.filter(name=params.get("key", "")).first():
            resolved["key"] = token.id
        if params.get("org") in ("org", "other_org"):
            resolved["org"] = getattr(self, params["org"]).id
        return resolved

    def get(self, role: str, name: str, params: dict[str, object]):
        self.client.force_login(getattr(self, role))
        return self.client.get(reverse(name), params)

    def test_overview_and_requests_show_the_same_requests(self):
        for role, params, expected in ROWS:
            query = self.query(params)
            with self.subTest(role=role, **params):
                overview = self.get(role, "usage", query)
                log = self.get(role, "requests", query)
                if expected == NOT_FOUND:
                    self.assertEqual((overview.status_code, log.status_code), (404, 404))
                    continue
                filters = overview.context["filters"]
                shown = filters.period.of(filters.requests)
                self.assertEqual(set(shown.values_list("path", flat=True)), expected)
                kpis = {kpi.label: kpi.value for kpi in overview.context["kpis"]}
                self.assertEqual(kpis["Requests"], str(len(expected)))
                rows = log.context["log"].rows
                self.assertEqual({row.request.path for row in rows}, expected)
                self.assertEqual(log.context["log"].tabs[0].count, len(expected))
                self.assertEqual(log.context["filters"].can_pick_org, role == "admin")

    def test_all_keys_lists_the_keys_behind_those_requests(self):
        for role, params, expected in ROWS:
            if expected == NOT_FOUND or "key" in params:
                continue
            with self.subTest(role=role, **params):
                response = self.get(role, "usage_keys", self.query(params))
                if role in ALL_KEYS_REDIRECT_TO:
                    token = Token.objects.get(name=ALL_KEYS_REDIRECT_TO[role])
                    self.assertRedirects(
                        response,
                        reverse("token_detail", args=[token.id]),
                        fetch_redirect_response=False,
                    )
                    continue
                keys = {KEY_OF.get(path, path.removeprefix("/")) for path in expected}
                listed = response.context["keys"]
                self.assertEqual({key.name for key in listed}, keys)
                self.assertEqual(
                    {key.href for key in listed},
                    {
                        reverse("token_detail", args=[Token.objects.get(name=name).id])
                        for name in keys
                    },
                )

    def test_the_key_page_opens_exactly_the_keys_each_role_may_open(self):
        for role, opens in OPENS:
            for viewed_org in (None, *VIEWED_ORGS.get(role, ())):
                self.client.force_login(getattr(self, role))
                if viewed_org is not None:
                    self.client.get(reverse("tokens"), self.query({"org": viewed_org}))
                for token in Token.objects.all():
                    with self.subTest(role=role, viewed_org=viewed_org, key=token.name):
                        response = self.client.get(reverse("token_detail", args=[token.id]))
                        self.assertEqual(response.status_code, 200 if token.name in opens else 404)

    def test_api_keys_lists_only_the_persons_keys(self):
        names = set(Token.objects.values_list("name", flat=True))
        for role, keys in OWN_KEYS:
            for viewed_org in (None, *VIEWED_ORGS.get(role, ())):
                with self.subTest(role=role, viewed_org=viewed_org):
                    params = self.query({"org": viewed_org}) if viewed_org else {}
                    self.client.force_login(getattr(self, role))
                    self.client.get(reverse("usage"), params)
                    page = self.client.get(reverse("tokens")).content.decode()
                    shown = {name for name in names if f">{name}</a>" in page}
                    self.assertEqual(shown, keys)

    def test_edit_and_delete_pages_open_for_the_owner_only(self):
        owned = {
            "solo": {"solo-laptop"},
            "member": {"member-laptop"},
            "org_admin": {"org-admin-laptop"},
        }
        for role, _opens in OPENS:
            self.client.force_login(getattr(self, role))
            for token in Token.objects.all():
                for page in ("token_edit", "token_delete"):
                    with self.subTest(role=role, key=token.name, page=page):
                        response = self.client.get(reverse(page, args=[token.id]))
                        expected = 200 if token.name in owned.get(role, set()) else 404
                        self.assertEqual(response.status_code, expected)

    def test_rotating_works_on_own_keys_and_managed_service_accounts_only(self):
        managed = {
            "solo": {"solo-laptop"},
            "member": {"member-laptop"},
            "team_admin": {"pipeline"},
            "org_admin": {"org-admin-laptop", "pipeline", "side-sa"},
            "admin": {"pipeline", "side-sa", "foreign-sa"},
            "other_org_admin": {"foreign-sa"},
        }
        for role, _opens in OPENS:
            self.client.force_login(getattr(self, role))
            for token in Token.objects.all():
                with self.subTest(role=role, key=token.name):
                    before = token.key_hash
                    response = self.client.post(reverse("token_regenerate", args=[token.id]))
                    token.refresh_from_db()
                    if token.name in managed[role]:
                        self.assertEqual(response.status_code, 302)
                        self.assertNotEqual(token.key_hash, before)
                    else:
                        self.assertEqual(response.status_code, 404)
                        self.assertEqual(token.key_hash, before)

    def test_org_admin_in_another_orgs_team_does_not_see_its_log(self):
        join(self.org_admin, self.foreign_team)
        foreign_sa = Token.objects.get(name="foreign-sa")
        self.client.force_login(self.org_admin)
        log = self.client.get(reverse("requests"))
        self.assertEqual({row.request.path for row in log.context["log"].rows}, LAB)
        self.assertEqual(
            self.client.get(reverse("requests"), {"key": foreign_sa.id}).status_code, 404
        )
        self.assertEqual(self.client.get(reverse("usage"), {"key": foreign_sa.id}).status_code, 404)
        self.assertEqual(
            self.client.get(reverse("token_detail", args=[foreign_sa.id])).status_code, 404
        )
