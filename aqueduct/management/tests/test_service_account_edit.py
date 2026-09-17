"""Regression test for the Service Account edit page.

The edit view (ServiceAccountUpdateView) inherited from UpdateView without
setting `form_class`/`template_name`, so every GET/POST to
/service-accounts/<id>/edit/ raised ImproperlyConfigured ("ModelFormMixin
without the 'fields' attribute") -> 500, even though the create view worked.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase
from django.urls import reverse

from management.models import Org, ServiceAccount, Team, Token, UserGroup, UserProfile

User = get_user_model()


class ServiceAccountEditViewTests(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="repro-org")
        self.team = Team.objects.create(name="repro-team", org=self.org)
        self.user = User.objects.create_user(username="adminuser", email="adminuser@example.com")
        profile = UserProfile.objects.create(user=self.user, org=self.org)
        Group.objects.get_or_create(name=UserGroup.ORG_ADMIN.value)
        profile.group = UserGroup.ORG_ADMIN.value
        self.sa = ServiceAccount.objects.create(name="repro-sa", team=self.team)
        token = Token(name="initial", user=self.user, service_account=self.sa)
        token._set_new_key()
        token.save()
        self.client.force_login(self.user)

    def _url(self):
        return reverse("service_account_edit", kwargs={"service_account_id": self.sa.id})

    def test_edit_get_returns_200(self):
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 200)

    def test_edit_get_without_token_returns_200(self):
        Token.objects.filter(service_account=self.sa).delete()
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 200)

    def test_edit_post_updates_service_account(self):
        resp = self.client.post(
            self._url(), {"name": "renamed-sa", "description": "updated", "token_expires_at": ""}
        )
        self.assertRedirects(resp, reverse("team", kwargs={"id": self.team.id}))
        self.sa.refresh_from_db()
        self.assertEqual(self.sa.name, "renamed-sa")
        self.assertEqual(self.sa.description, "updated")
