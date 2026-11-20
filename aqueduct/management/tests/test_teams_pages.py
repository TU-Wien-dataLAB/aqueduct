import re
from datetime import timedelta
from typing import ClassVar
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from management.models import Request, ServiceAccount, Team, TeamMembership, Token, UserGroup
from management.tests.helpers import RoleWorld, configured_limits, join, make_token, secret_on

User = get_user_model()


class TeamListTest(RoleWorld):
    def test_member_sees_only_their_teams_and_team_admins_see_a_badge(self):
        badge = '<span class="badge badge-sm whitespace-nowrap bg-base-100">Team admin</span>'
        response = self.as_user(self.member).get(reverse("org"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{reverse("team", args=[self.team.id])}"')
        self.assertContains(response, "Your role")
        self.assertNotContains(response, badge, html=True)
        self.assertNotContains(response, "Robotics")
        self.assertNotContains(response, reverse("team_create"))
        self.assertContains(self.as_user(self.team_admin).get(reverse("org")), badge, html=True)

    def test_user_outside_every_team_is_told_whom_to_ask(self):
        response = self.as_user(self.solo).get(reverse("org"))
        self.assertContains(response, "You're not in a team. Ask your org admin to add you.")
        self.assertNotContains(response, "NLP Lab")
        self.assertNotContains(response, 'name="q"')
        self.assertNotContains(response, "<table")

    def test_org_admin_sees_every_team_of_their_org_and_may_create_one(self):
        response = self.as_user(self.org_admin).get(reverse("org"))
        self.assertContains(response, "NLP Lab")
        self.assertContains(response, "Robotics")
        self.assertNotContains(response, "Foreign")
        self.assertContains(response, f'href="{reverse("team_create")}"')
        self.assertNotContains(response, "Your role")

    def test_search_filters_by_team_name(self):
        response = self.as_user(self.org_admin).get(reverse("org"), {"q": "robo"})
        self.assertContains(response, "Robotics")
        self.assertNotContains(response, "NLP Lab")
        response = self.as_user(self.org_admin).get(reverse("org"), {"q": "nothing-like-it"})
        self.assertContains(response, "No team matches.")

    def test_row_shows_counts_limit_and_service_account_tokens_of_seven_days(self):
        self.team.requests_per_minute = 1200
        self.team.save()
        now = timezone.now()
        Request.objects.create(
            token=self.pipeline_key,
            model="gpt-4.1-nano",
            timestamp=now,
            input_tokens=2000,
            output_tokens=500,
        )
        Request.objects.create(
            token=self.pipeline_key,
            model="gpt-4.1-nano",
            timestamp=now - timedelta(days=9),
            input_tokens=9_000_000,
        )
        ServiceAccount.objects.create(name="crawler", team=self.team)
        personal = make_token(self.member, "laptop")
        Request.objects.create(
            token=personal, model="gpt-4.1-nano", timestamp=now, input_tokens=7_000_000
        )
        response = self.as_user(self.member).get(reverse("org"))
        row = response.content.decode().split("Language tech", 1)[1].split("</tr>", 1)[0]
        members_and_service_accounts = re.findall(
            r'<td class="text-right max-lg:hidden">(\d+)</td>', row
        )
        self.assertEqual(members_and_service_accounts, ["2", "2"])
        self.assertIn("1,200", row)
        self.assertIn(">2.5K<", row)
        self.assertIn("2 members · 2 service accounts · 2.5K tokens", row)
        self.assertContains(response, "Requests/min")
        self.assertContains(response, "Tokens, 7 days")

    def test_teams_come_fifty_a_page_and_keep_the_search(self):
        for n in range(55):
            Team.objects.create(name=f"Squad {n:02}", org=self.org)
        client = self.as_user(self.org_admin)
        first = client.get(reverse("org"), {"q": "squad"}).content.decode()
        self.assertIn("Squad 49", first)
        self.assertNotIn("Squad 50", first)
        self.assertIn("1\u201350 of 55", first)
        self.assertIn('href="?q=squad&amp;page=2"', first)
        second = client.get(reverse("org"), {"q": "squad", "page": 2}).content.decode()
        self.assertIn("Squad 54", second)
        self.assertNotIn("Squad 49", second)


class TeamDetailTest(RoleWorld):
    def url(self, team: Team | None = None) -> str:
        return reverse("team", args=[(team or self.team).id])

    def test_member_sees_the_team_without_management_controls(self):
        response = self.as_user(self.member).get(self.url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "pipeline")
        self.assertContains(response, "team-admin@example.com")
        for name, args in [
            ("team_edit", [self.team.id]),
            ("service_account_create", [self.team.id]),
            ("service_account_edit", [self.pipeline.id]),
            ("service_account_transfer", [self.pipeline.id]),
            ("service_account_delete", [self.pipeline.id]),
            ("token_regenerate", [self.pipeline_key.id]),
            ("team_delete", [self.team.id]),
            ("team_members", [self.team.id]),
        ]:
            self.assertNotContains(response, reverse(name, args=args))
        self.assertNotContains(response, 'name="email"')
        self.assertNotContains(response, 'name="profile_id"')

    def test_team_admin_manages_members_and_service_accounts(self):
        response = self.as_user(self.team_admin).get(self.url())
        for name, args in [
            ("team_edit", [self.team.id]),
            ("service_account_create", [self.team.id]),
            ("service_account_edit", [self.pipeline.id]),
            ("service_account_transfer", [self.pipeline.id]),
            ("service_account_delete", [self.pipeline.id]),
            ("token_regenerate", [self.pipeline_key.id]),
        ]:
            self.assertContains(response, reverse(name, args=args))
        self.assertNotContains(response, reverse("team_delete", args=[self.team.id]))
        self.assertNotContains(response, 'value="grant_admin"')
        self.assertContains(response, 'name="email"')
        self.assertNotContains(response, "solo@example.com")
        self.assertNotContains(response, "<option")

    def test_the_edit_link_shows_exactly_for_those_the_edit_page_admits(self):
        edit = reverse("team_edit", args=[self.team.id])
        for user, admitted in (
            (self.member, False),
            (self.team_admin, True),
            (self.org_admin, True),
        ):
            with self.subTest(user=user.username):
                client = self.as_user(user)
                page = client.get(self.url())
                self.assertEqual(f'href="{edit}"' in page.content.decode(), admitted)
                self.assertEqual(client.get(edit).status_code, 200 if admitted else 404)
        page = self.as_user(self.team_admin).get(self.url())
        self.assertNotContains(page, 'aria-label="Edit limits"')

    def test_owner_of_a_service_account_is_pointed_to_the_transfer(self):
        response = self.as_user(self.team_admin).get(self.url())
        page = response.content.decode().split('id="members-title"', 1)[1]
        owner_row = page.split("team-admin@example.com</span>", 1)[1].split("</tr>", 1)[0]
        owner_row = owner_row.rsplit("<td", 1)[1].split(">", 1)[1].rsplit("</td>", 1)[0]
        transfer = reverse("service_account_transfer", args=[self.pipeline.id])
        self.assertInHTML(
            '<div class="muted flex flex-wrap items-baseline gap-x-1 text-sm md:justify-end">'
            '<span class="break-all">'
            'Owns <span class="font-mono">pipeline</span></span>'
            '<span class="whitespace-nowrap">· '
            f'<a class="link-plain" href="{transfer}">Transfer</a></span>'
            "</div>",
            owner_row,
        )
        self.assertNotIn('value="remove"', owner_row)
        member_row = page.split("member@example.com</span>", 1)[1].split("</tr>", 1)[0]
        self.assertIn(f'value="{self.member.profile.id}"', member_row)

    def test_org_admin_may_appoint_team_admins_and_finds_delete_on_the_edit_page(self):
        response = self.as_user(self.org_admin).get(self.url())
        self.assertNotContains(response, reverse("team_delete", args=[self.team.id]))
        self.assertContains(response, 'name="action" value="grant_admin"')
        self.assertContains(response, ">Make admin</button>")
        self.assertContains(response, 'name="action" value="revoke_admin"')
        self.assertContains(response, ">Remove admin</button>")

    def test_only_admins_see_the_service_account_cap(self):
        self.assertContains(
            self.as_user(self.member).get(self.url()), "<span>1 service account</span>"
        )
        self.assertContains(
            self.as_user(self.team_admin).get(self.url()), "<span>1 of 10 service accounts</span>"
        )

    def test_add_member_opens_one_email_field(self):
        response = self.as_user(self.team_admin).get(self.url())
        self.assertContains(response, 'placeholder="name@example.org"')
        self.assertNotContains(
            response,
            '<details class="group ml-auto flex flex-col items-end open:max-sm:w-full" open>',
        )
        response = self.as_user(self.team_admin).post(
            reverse("team_members", args=[self.team.id]),
            {"action": "add", "email": "nobody@example.com"},
        )
        self.assertContains(
            response,
            '<details class="group ml-auto flex flex-col items-end open:max-sm:w-full" open>',
        )

    def test_service_account_row_counts_seven_days(self):
        Request.objects.create(
            token=self.pipeline_key,
            model="gpt-4.1-nano",
            timestamp=timezone.now(),
            input_tokens=3000,
            cached_input_tokens=1500,
            output_tokens=1000,
            status_code=200,
        )
        response = self.as_user(self.member).get(self.url())
        self.assertContains(response, self.pipeline_key.key_preview)
        self.assertContains(response, '<td class="text-right tabular-nums max-lg:hidden">4K</td>')
        self.assertNotContains(response, "Service accounts, last 7 days")
        self.assertNotContains(response, 'class="tbar"')

    def test_unavailable_models_show_only_to_org_admins_and_only_when_some_are(self):
        response = self.as_user(self.org_admin).get(self.url())
        self.assertContains(response, "Blocked for service accounts")
        self.assertNotContains(response, '<dt class="muted">Models</dt>', html=True)
        self.team.excluded_models = ["gpt-image-1"]
        self.team.save()
        self.assertNotContains(self.as_user(self.member).get(self.url()), "gpt-image-1")
        response = self.as_user(self.org_admin).get(self.url())
        self.assertContains(response, "gpt-image-1")
        self.assertContains(response, '<dt class="muted">Models</dt>', html=True)
        self.assertNotContains(response, '<dt class="muted">MCP servers</dt>', html=True)

    def test_without_rate_limits_the_card_says_so_and_names_the_concurrent_limit(self):
        edit = f'href="{reverse("team_edit", args=[self.team.id])}" aria-label="Edit limits"'
        for user, edits in [(self.member, False), (self.team_admin, False), (self.org_admin, True)]:
            with (
                self.subTest(user=user.username),
                self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=3),
            ):
                response = self.as_user(user).get(self.url())
                self.assertContains(response, "No limit on requests or tokens.")
                self.assertEqual(configured_limits(response)["Concurrent requests"], ["3"])
                self.assertContains(
                    response, "Each service account key gets these limits unless it has its own."
                )
                self.assertEqual(edit in response.content.decode(), edits)

    def test_concurrent_limit_is_the_teams_then_the_orgs(self):
        for holder, limit in ((self.org, 40), (self.team, 20)):
            holder.concurrent_requests_limit = limit
            holder.save()
            with self.subTest(holder=holder):
                response = self.as_user(self.member).get(self.url())
                self.assertEqual(configured_limits(response)["Concurrent requests"], [str(limit)])

    def test_limits_are_listed_by_window_and_metric(self):
        self.team.requests_per_minute = 10
        self.team.input_tokens_per_minute = 5000
        self.team.hourly_limit_multiplier = 6
        self.team.save()
        response = self.as_user(self.member).get(self.url())
        limits = configured_limits(response)
        self.assertEqual(limits["Per minute"], ["10", "5K", "No limit"])
        self.assertEqual(limits["Per hour"], ["60", "30K", "No limit"])
        self.assertContains(
            response, "Each service account key gets these limits unless it has its own."
        )

    def role_of(self, response, email: str) -> str:
        page = response.content.decode()
        row = page.split(f'<span class="font-medium">{email}</span>', 1)[1].split("</tr>", 1)[0]
        return row.split("<span>", 1)[1].split("</span>", 1)[0]

    def test_roles_follow_groups_org_and_team_admin_flag(self):
        admin = self.user("site-admin", UserGroup.ADMIN, self.other_org)
        for user in (admin, self.org_admin, self.other_org_admin):
            join(user, self.team)
        response = self.as_user(self.member).get(self.url())
        for email, role in [
            ("site-admin@example.com", "Platform admin"),
            ("org-admin@example.com", "Org admin"),
            ("other-org-admin@example.com", "Member"),
            ("team-admin@example.com", "Team admin"),
            ("member@example.com", "Member"),
        ]:
            with self.subTest(email=email):
                self.assertEqual(self.role_of(response, email), role)

    def test_member_roles_cost_the_same_queries_for_any_team_size(self):
        client = self.as_user(self.member)
        with CaptureQueriesContext(connection) as small:
            client.get(self.url())
        for n in range(10):
            join(
                self.user(f"extra-{n}", UserGroup.ORG_ADMIN if n % 2 else UserGroup.USER), self.team
            )
        with CaptureQueriesContext(connection) as large:
            client.get(self.url())
        self.assertEqual(len(large), len(small))

    def test_members_come_fifty_a_page_team_admins_first_and_can_be_searched(self):
        for n in range(55):
            join(self.user(f"a{n:02}"), self.team)
        client = self.as_user(self.member)
        first = client.get(self.url()).content.decode()
        emails = re.findall(r'<span class="font-medium">([^<]+)</span>', first)
        self.assertEqual(len(emails), 50)
        self.assertEqual(emails[:2], ["team-admin@example.com", "a00@example.com"])
        self.assertIn("57 members", first)
        self.assertIn("1\u201350 of 57", first)
        second = client.get(self.url(), {"page": 2}).content.decode()
        self.assertEqual(
            re.findall(r'<span class="font-medium">([^<]+)</span>', second),
            [
                "a49@example.com",
                "a50@example.com",
                "a51@example.com",
                "a52@example.com",
                "a53@example.com",
                "a54@example.com",
                "member@example.com",
            ],
        )
        found = client.get(self.url(), {"q": "A1"}).content.decode()
        self.assertEqual(
            re.findall(r'<span class="font-medium">([^<]+)</span>', found),
            [f"a1{n}@example.com" for n in range(10)],
        )
        self.assertIn("57 members", found)
        self.assertIn("No member matches.", client.get(self.url(), {"q": "zzz"}).content.decode())

    def test_outsiders_get_404_without_the_team_name(self):
        for user, team in [
            (self.solo, self.team),
            (self.member, self.side_team),
            (self.other_org_admin, self.team),
        ]:
            with self.subTest(user=user.username, team=team.name):
                response = self.as_user(user).get(self.url(team))
                self.assertEqual(response.status_code, 404)
                self.assertNotContains(response, team.name, status_code=404)


class TeamMembershipTest(RoleWorld):
    def url(self) -> str:
        return reverse("team", args=[self.team.id])

    def change(self, client, action: str, **data):
        return client.post(reverse("team_members", args=[self.team.id]), {"action": action, **data})

    def members(self) -> set[str]:
        return set(self.team.member_profiles.values_list("user__username", flat=True))

    def test_team_admin_adds_by_email_ignoring_case_and_removes_a_member(self):
        client = self.as_user(self.team_admin)
        response = self.change(client, "add", email="Solo@Example.com")
        self.assertRedirects(response, self.url(), fetch_redirect_response=False)
        self.assertIn("solo", self.members())
        self.change(client, "remove", profile_id=self.solo.profile.id)
        self.assertNotIn("solo", self.members())

    def test_team_admin_who_leaves_lands_on_the_teams_page(self):
        TeamMembership.objects.filter(user_profile=self.member.profile).update(is_admin=True)
        response = self.as_user(self.member).post(
            reverse("team_members", args=[self.team.id]),
            {"action": "remove", "profile_id": self.member.profile.id},
            follow=True,
        )
        self.assertEqual(response.redirect_chain, [(reverse("org"), 302)])
        self.assertContains(response, "member@example.com removed.")
        self.assertNotIn("member", self.members())

    def test_org_admin_who_leaves_stays_on_the_team(self):
        join(self.org_admin, self.team)
        response = self.change(
            self.as_user(self.org_admin), "remove", profile_id=self.org_admin.profile.id
        )
        self.assertRedirects(response, self.url(), fetch_redirect_response=False)
        self.assertNotIn("org-admin", self.members())

    def test_adding_an_email_nobody_in_the_org_uses_explains_why(self):
        client = self.as_user(self.team_admin)
        before = self.members()
        for email in ("other-org-admin@example.com", "nobody@example.com"):
            with self.subTest(email=email):
                response = self.change(client, "add", email=email)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, f"No one in lab-org uses {email}.")
                self.assertContains(response, f'value="{email}"')
                self.assertEqual(self.members(), before)

    def test_adding_a_member_again_explains_why(self):
        response = self.change(self.as_user(self.team_admin), "add", email="member@example.com")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "member@example.com is already in this team.")
        self.assertEqual(TeamMembership.objects.filter(team=self.team).count(), 2)

    def test_member_and_outsider_cannot_change_membership(self):
        before = self.members()
        for user in (self.member, self.solo, self.other_org_admin):
            for data in (
                {"action": "remove", "profile_id": self.member.profile.id},
                {"action": "add", "email": "solo@example.com"},
            ):
                with self.subTest(user=user.username, data=data):
                    url = reverse("team_members", args=[self.team.id])
                    self.assertEqual(self.as_user(user).post(url, data).status_code, 404)
                    self.assertEqual(self.members(), before)

    def test_the_pages_links_after_a_failed_add_lead_back_to_the_team_page(self):
        client = self.as_user(self.team_admin)
        members = reverse("team_members", args=[self.team.id])
        self.assertRedirects(
            client.get(members, {"q": "solo", "page": "2"}),
            f"{self.url()}?q=solo&page=2",
            fetch_redirect_response=False,
        )

    def test_an_unknown_action_is_a_404(self):
        for data in ({}, {"action": "promote", "profile_id": self.member.profile.id}):
            with self.subTest(data=data):
                url = reverse("team_members", args=[self.team.id])
                self.assertEqual(self.as_user(self.org_admin).post(url, data).status_code, 404)


class TeamFormsTest(RoleWorld):
    def test_org_admin_creates_a_team(self):
        client = self.as_user(self.org_admin)
        page = client.get(reverse("team_create"))
        self.assertContains(page, "Create a team")
        self.assertNotContains(page, "The team is created in")
        response = client.post(
            reverse("team_create"), {"name": "Vision", "description": "", "org": self.org.id}
        )
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
        self.assertTrue(Team.objects.filter(name="Vision", org=self.org).exists())

    def test_users_cannot_create_a_team(self):
        for user in (self.team_admin, self.member):
            with self.subTest(user=user.username):
                response = self.as_user(user).post(reverse("team_create"), {"name": "Vision"})
                self.assertEqual(response.status_code, 404)
        self.assertFalse(Team.objects.filter(name="Vision").exists())

    def test_duplicate_team_name_rerenders_the_form(self):
        response = self.as_user(self.org_admin).post(
            reverse("team_create"), {"name": "nlp lab", "org": self.org.id}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "A team with this name already exists in your organization.")

    def test_team_admin_edits_the_team(self):
        url = reverse("team_edit", args=[self.team.id])
        client = self.as_user(self.team_admin)
        self.assertContains(client.get(url), 'value="NLP Lab"')
        response = client.post(url, {"name": "NLP Group", "description": "new"})
        self.assertRedirects(
            response, reverse("team", args=[self.team.id]), fetch_redirect_response=False
        )
        self.team.refresh_from_db()
        self.assertEqual(self.team.name, "NLP Group")

    def test_member_and_outsiders_cannot_edit_the_team(self):
        url = reverse("team_edit", args=[self.team.id])
        for user in (self.member, self.solo, self.other_org_admin):
            with self.subTest(user=user.username):
                response = self.as_user(user).post(url, {"name": "Hijacked"})
                self.assertEqual(response.status_code, 404)
        self.team.refresh_from_db()
        self.assertEqual(self.team.name, "NLP Lab")


class TeamLimitsEditTest(RoleWorld):
    LIMITS: ClassVar[dict] = {
        "requests_per_minute": "120",
        "input_tokens_per_minute": "50000",
        "output_tokens_per_minute": "",
        "concurrent_requests_limit": "8",
    }

    def url(self) -> str:
        return reverse("team_edit", args=[self.team.id])

    def post(self, user, **limits):
        data = {"name": "NLP Lab", "description": "Language tech", **self.LIMITS, **limits}
        return self.as_user(user).post(self.url(), data)

    def limits(self) -> tuple:
        self.team.refresh_from_db()
        return (
            self.team.requests_per_minute,
            self.team.input_tokens_per_minute,
            self.team.output_tokens_per_minute,
            self.team.concurrent_requests_limit,
        )

    def test_org_admin_and_admin_set_the_limits(self):
        admin = self.user("site-admin", UserGroup.ADMIN, self.other_org)
        for user in (self.org_admin, admin):
            with self.subTest(user=user.username):
                self.team.requests_per_minute = None
                self.team.save()
                page = self.as_user(user).get(self.url())
                self.assertContains(page, 'name="requests_per_minute"')
                response = self.post(user)
                self.assertRedirects(
                    response, reverse("team", args=[self.team.id]), fetch_redirect_response=False
                )
                self.assertEqual(self.limits(), (120, 50000, None, 8))

    def test_help_names_the_org_limit_an_empty_field_falls_back_to(self):
        self.org.requests_per_minute = 600
        self.org.save()
        page = self.as_user(self.org_admin).get(self.url())
        self.assertContains(page, "Leave it empty for the organization&#x27;s limit, 600.")
        self.assertContains(page, "Leave it empty for no limit.")

    def test_help_names_what_an_empty_concurrent_limit_falls_back_to(self):
        with self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=3):
            page = self.as_user(self.org_admin).get(self.url())
        self.assertContains(page, "Leave it empty for the server default, 3.")
        self.org.concurrent_requests_limit = 40
        self.org.save()
        page = self.as_user(self.org_admin).get(self.url())
        self.assertContains(page, "Leave it empty for the organization&#x27;s limit, 40.")

    def test_negative_limit_rerenders_the_form(self):
        response = self.post(self.org_admin, requests_per_minute="-1")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ensure this value is greater than or equal to 0.")
        self.assertEqual(self.limits(), (None, None, None, None))

    def test_team_admin_edits_the_team_but_not_its_limits(self):
        self.assertNotContains(
            self.as_user(self.team_admin).get(self.url()), 'name="requests_per_minute"'
        )
        response = self.post(self.team_admin, name="NLP Group")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.limits(), (None, None, None, None))
        self.team.refresh_from_db()
        self.assertEqual(self.team.name, "NLP Group")

    def test_member_and_other_org_admin_change_nothing(self):
        for user in (self.member, self.other_org_admin, self.solo):
            with self.subTest(user=user.username):
                self.assertEqual(self.post(user).status_code, 404)
        self.assertEqual(self.limits(), (None, None, None, None))

    def test_only_org_admins_find_delete_on_the_edit_page(self):
        delete = f'href="{reverse("team_delete", args=[self.team.id])}"'
        self.assertContains(self.as_user(self.org_admin).get(self.url()), delete)
        self.assertNotContains(self.as_user(self.team_admin).get(self.url()), delete)


class ServiceAccountFormsTest(RoleWorld):
    def test_team_admin_creates_a_service_account_and_sees_its_key_once(self):
        client = self.as_user(self.team_admin)
        url = reverse("service_account_create", args=[self.team.id])
        page = client.get(url)
        self.assertContains(page, "Add a service account")
        self.assertContains(page, "You'll own its key. It's shown once, after you create it.")
        self.assertContains(page, "Pick a date")
        response = client.post(
            url, {"name": "rag", "description": "", "token_expires_at": ""}, follow=True
        )
        sa = ServiceAccount.objects.get(name="rag", team=self.team)
        detail = reverse("token_detail", args=[sa.token.id])
        self.assertEqual(response.redirect_chain, [(detail, 302)])
        self.assertContains(response, 'class="break-all font-mono">rag</span> is ready')
        self.assertContains(
            response, "Store it as a secret, for example OPENAI_API_KEY in your CI."
        )
        self.assertNotContains(response, "created.")
        self.assertNotContains(response, 'class="alert ')
        secret = secret_on(response)
        self.assertEqual(Token.find_by_key(secret), sa.token)
        self.assertContains(response, "OPENAI_BASE_URL=http://testserver/v1")
        self.assertContains(response, f"OPENAI_API_KEY={secret}")
        self.assertNotContains(response, "chat/completions")
        self.assertNotContains(client.get(detail), secret)

    def test_rotating_a_service_account_key_shows_the_secret_on_its_own_page(self):
        detail = reverse("token_detail", args=[self.pipeline_key.id])
        response = self.as_user(self.team_admin).post(
            reverse("token_regenerate", args=[self.pipeline_key.id]), follow=True
        )
        self.assertEqual(response.redirect_chain, [(detail, 302)])
        self.assertContains(response, 'class="break-all font-mono">pipeline</span> is ready')
        self.assertContains(
            response, "Store it as a secret, for example OPENAI_API_KEY in your CI."
        )

    def test_the_new_key_is_named_after_its_service_account(self):
        self.as_user(self.team_admin).post(
            reverse("service_account_create", args=[self.team.id]),
            {"name": "rag", "description": "", "token_expires_at": ""},
        )
        self.assertEqual(ServiceAccount.objects.get(name="rag", team=self.team).token.name, "rag")

    def test_editing_a_service_account_leaves_the_name_of_its_key_alone(self):
        Token.objects.filter(pk=self.pipeline_key.pk).update(name="Initial token for pipeline")
        self.as_user(self.team_admin).post(
            reverse("service_account_edit", args=[self.pipeline.id]),
            {"name": "pipeline-2", "description": "", "token_expires_at": ""},
        )
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.name, "Initial token for pipeline")

    def test_a_failed_key_leaves_no_service_account_behind(self):
        url = reverse("service_account_create", args=[self.team.id])
        client = self.as_user(self.team_admin)
        with (
            patch.object(Token, "_set_new_key", side_effect=RuntimeError("no entropy")),
            self.assertRaises(RuntimeError),
        ):
            client.post(url, {"name": "rag", "description": "", "token_expires_at": ""})
        self.assertFalse(ServiceAccount.objects.filter(name="rag").exists())

    def test_member_and_outsiders_cannot_create_a_service_account(self):
        url = reverse("service_account_create", args=[self.team.id])
        for user in (self.member, self.solo, self.other_org_admin):
            with self.subTest(user=user.username):
                response = self.as_user(user).post(url, {"name": "rag", "token_expires_at": ""})
                self.assertEqual(response.status_code, 404)
        self.assertFalse(ServiceAccount.objects.filter(name="rag").exists())

    def test_team_admin_edits_a_service_account(self):
        url = reverse("service_account_edit", args=[self.pipeline.id])
        client = self.as_user(self.team_admin)
        page = client.get(url)
        self.assertContains(page, "Edit service account")
        self.assertContains(page, "Save changes")
        client.post(url, {"name": "pipeline-2", "description": "", "token_expires_at": ""})
        self.pipeline.refresh_from_db()
        self.assertEqual(self.pipeline.name, "pipeline-2")

    def test_past_expiry_rerenders_the_form_with_an_error(self):
        url = reverse("service_account_edit", args=[self.pipeline.id])
        past = (timezone.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
        response = self.as_user(self.team_admin).post(
            url, {"name": "pipeline", "description": "", "token_expires_at": past}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Expiration date must be in the future.")
        self.assertContains(response, 'class="input w-full max-w-xs input-error"')
        self.assertContains(response, 'class="input w-full max-w-md"')
        self.pipeline_key.refresh_from_db()
        self.assertIsNone(self.pipeline_key.expires_at)

    def test_member_and_outsiders_cannot_edit_a_service_account(self):
        url = reverse("service_account_edit", args=[self.pipeline.id])
        for user in (self.member, self.solo, self.other_org_admin):
            with self.subTest(user=user.username):
                response = self.as_user(user).post(
                    url, {"name": "hijacked", "token_expires_at": ""}
                )
                self.assertEqual(response.status_code, 404)
        self.pipeline.refresh_from_db()
        self.assertEqual(self.pipeline.name, "pipeline")

    def test_team_admin_picks_the_new_owner_from_the_members(self):
        url = reverse("service_account_transfer", args=[self.pipeline.id])
        client = self.as_user(self.team_admin)
        page = client.get(url)
        self.assertContains(page, '<select name="email" id="new-owner"')
        self.assertContains(
            page, '<option value="member@example.com">member@example.com</option>', html=True
        )
        self.assertNotContains(page, '<option value="team-admin@example.com"')
        self.assertContains(page, "The key keeps working. Only the owner changes.")
        response = client.post(url, {"email": "member@example.com"})
        self.assertRedirects(
            response, reverse("team", args=[self.team.id]), fetch_redirect_response=False
        )
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.user, self.member)

    def test_team_of_more_than_fifty_names_the_new_owner_by_email(self):
        for n in range(48):
            join(self.user(f"a{n:02}"), self.team)
        url = reverse("service_account_transfer", args=[self.pipeline.id])
        client = self.as_user(self.team_admin)
        self.assertContains(client.get(url), "<select")
        join(self.user("a48"), self.team)
        page = client.get(url)
        self.assertNotContains(page, "<select")
        self.assertContains(page, '<input type="email" name="email" id="new-owner"')
        response = client.post(url, {"email": "MEMBER@example.com"})
        self.assertRedirects(
            response, reverse("team", args=[self.team.id]), fetch_redirect_response=False
        )
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.user, self.member)

    def test_transfer_to_someone_outside_the_team_or_the_owner_explains_why(self):
        url = reverse("service_account_transfer", args=[self.pipeline.id])
        client = self.as_user(self.team_admin)
        for email, error in [
            ("solo@example.com", "solo@example.com is not a member of NLP Lab."),
            (
                "other-org-admin@example.com",
                "other-org-admin@example.com is not a member of NLP Lab.",
            ),
            ("nobody@example.com", "nobody@example.com is not a member of NLP Lab."),
            ("team-admin@example.com", "team-admin@example.com already owns this key."),
        ]:
            with self.subTest(email=email):
                response = client.post(url, {"email": email})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, error)
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.user, self.team_admin)

    def test_a_member_whose_profile_is_in_another_org_cannot_receive_the_key(self):
        moved = self.user("moved")
        join(moved, self.team)
        type(moved.profile).objects.filter(pk=moved.profile.pk).update(org=self.other_org)
        url = reverse("service_account_transfer", args=[self.pipeline.id])
        client = self.as_user(self.team_admin)
        self.assertNotIn("moved@example.com", client.get(url).context["owner_choices"])
        response = client.post(url, {"email": "moved@example.com"})
        self.assertContains(response, "moved@example.com is not a member of NLP Lab.")
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.user, self.team_admin)

    def test_member_and_outsiders_cannot_transfer_the_key(self):
        url = reverse("service_account_transfer", args=[self.pipeline.id])
        for user in (self.member, self.solo, self.other_org_admin):
            with self.subTest(user=user.username):
                self.as_user(user).post(url, {"email": "member@example.com"})
        self.pipeline_key.refresh_from_db()
        self.assertEqual(self.pipeline_key.user, self.team_admin)


class TeamScopeTest(RoleWorld):
    def requests(self) -> list[tuple[str, str, dict]]:
        team, sa = [self.team.id], [self.pipeline.id]
        return [
            ("get", reverse("team", args=team), {}),
            (
                "post",
                reverse("team_members", args=team),
                {"action": "add", "email": "solo@example.com"},
            ),
            (
                "post",
                reverse("team_members", args=team),
                {"action": "remove", "profile_id": self.member.profile.id},
            ),
            ("get", reverse("team_edit", args=team), {}),
            ("post", reverse("team_edit", args=team), {"name": "Hijacked"}),
            ("get", reverse("team_delete", args=team), {}),
            ("post", reverse("team_delete", args=team), {}),
            (
                "post",
                reverse("team_members", args=team),
                {"profile_id": self.member.profile.id, "action": "grant_admin"},
            ),
            ("get", reverse("service_account_create", args=team), {}),
            (
                "post",
                reverse("service_account_create", args=team),
                {"name": "rag", "token_expires_at": ""},
            ),
            ("get", reverse("service_account_edit", args=sa), {}),
            (
                "post",
                reverse("service_account_edit", args=sa),
                {"name": "hijacked", "token_expires_at": ""},
            ),
            ("get", reverse("service_account_delete", args=sa), {}),
            ("post", reverse("service_account_delete", args=sa), {}),
            ("get", reverse("service_account_transfer", args=sa), {}),
            ("post", reverse("service_account_transfer", args=sa), {"email": "member@example.com"}),
            ("post", reverse("token_regenerate", args=[self.pipeline_key.id]), {}),
        ]

    def state(self) -> tuple:
        self.pipeline_key.refresh_from_db()
        return (
            list(Team.objects.values_list("name", flat=True).order_by("name")),
            list(TeamMembership.objects.values_list("user_profile_id", "is_admin").order_by("pk")),
            list(ServiceAccount.objects.values_list("name", flat=True).order_by("name")),
            self.pipeline_key.user_id,
            self.pipeline_key.key_hash,
        )

    def test_admin_of_another_org_and_outsiders_get_404_on_every_url(self):
        before = self.state()
        for user in (self.other_org_admin, self.solo):
            for method, url, data in self.requests():
                with self.subTest(user=user.username, method=method, url=url, data=data):
                    response = getattr(self.as_user(user), method)(url, data)
                    self.assertEqual(response.status_code, 404)
                    self.assertNotContains(response, "NLP Lab", status_code=404)
                    self.assertNotContains(response, "pipeline", status_code=404)
        self.assertEqual(self.state(), before)

    def test_each_role_opens_exactly_the_pages_it_may_use(self):
        team, sa = [self.team.id], [self.pipeline.id]
        everyone = {"admin", "org_admin", "team_admin", "member", "solo", "other_org_admin"}
        managers = {"admin", "org_admin", "team_admin"}
        pages = [
            ("team", team, {"admin", "org_admin", "team_admin", "member"}),
            ("team_create", [], {"admin", "org_admin", "other_org_admin"}),
            ("team_edit", team, managers),
            ("team_delete", team, {"admin", "org_admin"}),
            ("service_account_create", team, managers),
            ("service_account_edit", sa, managers),
            ("service_account_delete", sa, managers),
            ("service_account_transfer", sa, managers),
        ]
        for role in sorted(everyone):
            client = self.as_user(getattr(self, role))
            for name, args, opens in pages:
                with self.subTest(role=role, page=name):
                    status = client.get(reverse(name, args=args)).status_code
                    self.assertEqual(status, 200 if role in opens else 404)
            with self.subTest(role=role, page="team_members"):
                self.assertRedirects(
                    client.get(reverse("team_members", args=team)),
                    reverse("team", args=team),
                    fetch_redirect_response=False,
                )

    def test_platform_admin_reaches_the_teams_of_every_org(self):
        client = self.as_user(self.admin)
        for name in ("team", "team_edit", "team_delete", "service_account_create"):
            with self.subTest(page=name):
                url = reverse(name, args=[self.foreign_team.id])
                self.assertEqual(client.get(url).status_code, 200)

    def test_member_gets_404_on_every_url_but_the_team_page(self):
        before = self.state()
        for method, url, data in self.requests():
            if method == "get" and url == reverse("team", args=[self.team.id]):
                continue
            with self.subTest(method=method, url=url, data=data):
                response = getattr(self.as_user(self.member), method)(url, data)
                self.assertEqual(response.status_code, 404)
        self.assertEqual(self.state(), before)

    def test_team_admin_may_not_delete_the_team_or_appoint_admins(self):
        client = self.as_user(self.team_admin)
        for url, data in [
            (reverse("team_delete", args=[self.team.id]), {}),
            (
                reverse("team_members", args=[self.team.id]),
                {"profile_id": self.member.profile.id, "action": "grant_admin"},
            ),
        ]:
            with self.subTest(url=url):
                self.assertEqual(client.post(url, data).status_code, 404)
        self.assertTrue(Team.objects.filter(id=self.team.id).exists())
        self.assertFalse(TeamMembership.objects.get(user_profile=self.member.profile).is_admin)

    def test_org_admin_deletes_the_team(self):
        response = self.as_user(self.org_admin).post(
            reverse("team_delete", args=[self.team.id]), follow=True
        )
        self.assertRedirects(response, reverse("org"))
        self.assertContains(response, "Team NLP Lab deleted.")
        self.assertFalse(Team.objects.filter(id=self.team.id).exists())


class TeamMessagesTest(RoleWorld):
    def toasts(self, user, url: str, data: dict) -> list[str]:
        response = self.as_user(user).post(url, data, follow=True)
        return [str(message) for message in response.context["messages"]]

    def test_membership_changes_say_what_happened_or_what_to_do_first(self):
        members = reverse("team_members", args=[self.team.id])
        for user, url, data, toast in [
            (
                self.team_admin,
                members,
                {"action": "add", "email": "solo@example.com"},
                "solo@example.com added.",
            ),
            (
                self.team_admin,
                members,
                {"action": "remove", "profile_id": self.solo.profile.id},
                "solo@example.com removed.",
            ),
            (
                self.org_admin,
                members,
                {"action": "remove", "profile_id": self.team_admin.profile.id},
                "Transfer or delete team-admin@example.com's service accounts first.",
            ),
            (
                self.org_admin,
                members,
                {"profile_id": self.org_admin.profile.id, "action": "grant_admin"},
                "Org admins already have team admin rights.",
            ),
            (
                self.org_admin,
                members,
                {"profile_id": self.member.profile.id, "action": "grant_admin"},
                "member@example.com is now a team admin.",
            ),
        ]:
            with self.subTest(toast=toast):
                self.assertEqual(self.toasts(user, url, data), [toast])
