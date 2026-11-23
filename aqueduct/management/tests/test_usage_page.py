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

    def test_usage_by_model_sums_tokens(self):
        Request.objects.create(
            token=self.solo_key, model="gpt-4.1-nano", input_tokens=1200, output_tokens=30
        )
        Request.objects.create(
            token=self.solo_key, model="gpt-4.1-nano", input_tokens=800, output_tokens=0
        )
        Request.objects.create(token=self.member_key, model="gpt-4.1-nano", input_tokens=5)
        page = self.as_user(self.solo).get(reverse("usage"))
        (row,) = [row for row in page.context["per_model"] if row["count"]]
        self.assertEqual(
            row, {"model": "gpt-4.1-nano", "count": 2, "input_sum": 2000, "output_sum": 30}
        )
        self.assertContains(page, "Usage by model")
        self.assertContains(page, '<td class="text-right tabular-nums">2,000</td>', html=True)
