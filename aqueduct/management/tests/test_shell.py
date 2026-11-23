from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse

from management.tests.helpers import RoleWorld

User = get_user_model()


class SidebarTest(RoleWorld):
    def nav(self, user, url_name: str = "tokens") -> str:
        response = self.as_user(user).get(reverse(url_name))
        self.assertEqual(response.status_code, 200)
        return response.content.decode().split('<nav aria-label="Main">', 1)[1].split("</nav>")[0]

    def test_everyone_gets_the_personal_and_resource_pages(self):
        nav = self.nav(self.solo)
        for url_name in ("tokens", "usage", "models", "mcp_servers", "files", "batches"):
            with self.subTest(page=url_name):
                self.assertIn(f'href="{reverse(url_name)}"', nav)

    def test_teams_page_and_admin_links_need_their_rights(self):
        for user, teams_page, admin_panel in (
            (self.solo, False, False),
            (self.team_admin, False, False),
            (self.org_admin, True, False),
            (self.admin, True, False),
        ):
            with self.subTest(user=user.username):
                nav = self.nav(user)
                self.assertEqual(f'href="{reverse("org")}"' in nav, teams_page)
                self.assertEqual(f'href="{reverse("admin:index")}"' in nav, admin_panel)
        self.admin.is_staff = True
        self.admin.save()
        self.assertIn(f'href="{reverse("admin:index")}"', self.nav(self.admin))

    def test_a_member_reaches_their_teams_from_the_sidebar(self):
        nav = self.nav(self.member)
        self.assertIn(f'href="{reverse("team", args=[self.team.id])}"', nav)
        self.assertNotIn(f'href="{reverse("team", args=[self.side_team.id])}"', nav)
        self.assertNotIn("Teams</li>", self.nav(self.solo))

    def test_the_current_page_is_marked(self):
        nav = self.nav(self.solo, "usage")
        self.assertIn(f'<a href="{reverse("usage")}" class="menu-active" aria-current="page"', nav)
        self.assertEqual(nav.count('aria-current="page"'), 1)

    def test_only_platform_admins_get_the_org_switch(self):
        self.assertNotContains(self.as_user(self.org_admin).get(reverse("tokens")), "org-switch")
        page = self.as_user(self.admin).get(reverse("tokens"))
        self.assertContains(page, f'<option value="{self.other_org.id}">other-org</option>')
        self.assertContains(page, f'<option value="{self.org.id}" selected>lab-org</option>')

    def test_the_switch_returns_to_the_page_it_was_on(self):
        self.as_user(self.admin)
        response = self.client.post(
            reverse("org_switch"), {"org_id": self.other_org.id, "next": reverse("usage")}
        )
        self.assertRedirects(response, reverse("usage"))
        response = self.client.post(
            reverse("org_switch"), {"org_id": self.org.id, "next": "https://evil.example/"}
        )
        self.assertRedirects(response, reverse("org"))

    def test_the_footer_names_the_gateway(self):
        page = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(page, 'data-copy="http://testserver/v1"')

    @override_settings(
        EXTRA_NAV_LINKS={
            "Chat with us": "https://matrix.to/#/#aqueduct:matrix.org",
            "Source": "https://github.com/TU-Wien-dataLAB/aqueduct",
            "Docs": "https://example.org/docs",
        }
    )
    def test_footer_links_carry_an_icon_for_their_site(self):
        page = self.as_user(self.solo).get(reverse("tokens")).content.decode()
        items = page[page.index("<footer") :].split("<li")
        for label, icon in (
            ("Chat with us", "M7 4H4v16h3"),
            ("Source", "M15 22v-4"),
            ("Docs", "M10 14a4 4 0"),
        ):
            with self.subTest(label):
                (item,) = [item for item in items if f"<span>{label}</span>" in item]
                self.assertIn(icon, item)


class ErrorPageTest(RoleWorld):
    def test_404_renders_in_the_shell(self):
        with override_settings(DEBUG=False):
            response = self.as_user(self.solo).get(
                reverse("token_edit", args=[self.foreign_key.id])
            )
        self.assertEqual(response.status_code, 404)
        self.assertContains(response, "Page not found", status_code=404)
        self.assertContains(response, 'class="aq-sidebar', status_code=404)


class AdminHeaderTest(RoleWorld):
    def test_the_admin_shares_the_header(self):
        self.admin.is_staff = True
        self.admin.is_superuser = True
        self.admin.save()
        page = self.as_user(self.admin).get(reverse("admin:index"))
        self.assertContains(page, "Back to Aqueduct")
