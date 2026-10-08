"""Regression test for the token edit page ownership check.

TokenEditView (an UpdateView on Token) had no get_queryset(), so any
logged-in user could open /tokens/<id>/edit/ for any token — including
service-account keys of teams they are not in — and save a new name and
expiry date. The delete view scoped by owner; the edit view did not.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from management.models import Org, ServiceAccount, Team, Token, UserProfile

User = get_user_model()


def _make_token(user: UserProfile, **kwargs) -> Token:
    token = Token(name=kwargs.pop("name", "token"), user=user.user, **kwargs)
    token._set_new_key()
    token.save()
    return token


class TokenEditViewOwnershipTests(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="edit-org")
        self.team = Team.objects.create(name="edit-team", org=self.org)

        self.owner = User.objects.create_user(username="owner", email="owner@example.com")
        self.owner_profile = UserProfile.objects.create(user=self.owner, org=self.org)

        self.other = User.objects.create_user(username="other", email="other@example.com")
        UserProfile.objects.create(user=self.other, org=self.org)

        self.client.force_login(self.other)

    def _url(self, token_id: int) -> str:
        return reverse("token_edit", kwargs={"id": token_id})

    def test_other_user_cannot_edit_someone_elses_token(self):
        token = _make_token(self.owner_profile)
        resp = self.client.get(self._url(token.id))
        self.assertEqual(resp.status_code, 404)

        resp = self.client.post(self._url(token.id), {"name": "hacked", "expires_at": ""})
        self.assertEqual(resp.status_code, 404)
        token.refresh_from_db()
        self.assertEqual(token.name, "token")

    def test_other_user_cannot_edit_service_account_token(self):
        sa = ServiceAccount.objects.create(name="edit-sa", team=self.team)
        token = _make_token(self.owner_profile, service_account=sa)
        resp = self.client.get(self._url(token.id))
        self.assertEqual(resp.status_code, 404)

    def test_owner_can_edit_own_token(self):
        self.client.force_login(self.owner)
        token = _make_token(self.owner_profile)
        resp = self.client.post(self._url(token.id), {"name": "renamed", "expires_at": ""})
        self.assertRedirects(resp, reverse("tokens"))
        token.refresh_from_db()
        self.assertEqual(token.name, "renamed")
