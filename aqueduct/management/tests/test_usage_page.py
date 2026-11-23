from django.urls import reverse

from management.models import Request
from management.tests.helpers import RoleWorld


class UsagePageTest(RoleWorld):
    def test_totals_count_the_viewers_requests(self):
        Request.objects.create(
            token=self.solo_key, model="gpt-4.1-nano", status_code=200, input_tokens=1200
        )
        Request.objects.create(token=self.solo_key, model="gpt-4.1-nano", status_code=500)
        Request.objects.create(token=self.member_key, model="gpt-4.1-nano", status_code=200)
        page = self.as_user(self.solo).get(reverse("usage"))
        self.assertEqual(page.context["total_requests"], 2)
        self.assertContains(page, '<span class="text-2xl font-semibold tabular-nums">1,200</span>')
        self.assertContains(page, "1 failed")

    def test_renders_for_every_role(self):
        for user in (self.solo, self.member, self.org_admin, self.admin):
            with self.subTest(user=user.username):
                self.assertEqual(self.as_user(user).get(reverse("usage")).status_code, 200)
