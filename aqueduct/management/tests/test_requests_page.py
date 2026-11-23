from django.urls import reverse

from management.models import Request
from management.tests.helpers import RoleWorld


class RequestLogTest(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.solo_ok = Request.objects.create(token=cls.solo_key, path="/solo-ok", status_code=200)
        cls.solo_429 = Request.objects.create(token=cls.solo_key, path="/solo-429", status_code=429)
        cls.member_500 = Request.objects.create(
            token=cls.member_key, path="/member-500", status_code=500
        )
        cls.team_ok = Request.objects.create(
            token=cls.pipeline_key, path="/pipeline-ok", status_code=200
        )
        cls.foreign_ok = Request.objects.create(
            token=cls.foreign_key, path="/foreign-ok", status_code=200
        )

    def paths(self, user, **params) -> set[str]:
        response = self.as_user(user).get(reverse("requests"), params)
        self.assertEqual(response.status_code, 200)
        return {request.path for request in response.context["requests"]}

    def test_everyone_sees_the_requests_of_the_keys_they_may_see(self):
        self.assertEqual(self.paths(self.solo), {"/solo-ok", "/solo-429"})
        self.assertEqual(self.paths(self.member), {"/member-500", "/pipeline-ok"})
        self.assertEqual(
            self.paths(self.org_admin), {"/solo-ok", "/solo-429", "/member-500", "/pipeline-ok"}
        )
        self.assertEqual(self.paths(self.other_org_admin), {"/foreign-ok"})

    def test_a_platform_admin_sees_the_org_they_switched_to(self):
        self.assertNotIn("/foreign-ok", self.paths(self.admin))
        self.client.post(reverse("org_switch"), {"org_id": self.other_org.id})
        self.assertEqual(self.paths(self.admin), {"/foreign-ok"})

    def test_filters_by_key_and_status(self):
        self.assertEqual(self.paths(self.solo, status="rate_limited"), {"/solo-429"})
        self.assertEqual(self.paths(self.solo, status="ok"), {"/solo-ok"})
        self.assertEqual(self.paths(self.org_admin, status="failed"), {"/member-500"})
        self.assertEqual(self.paths(self.org_admin, key=self.pipeline_key.id), {"/pipeline-ok"})

    def test_another_persons_key_filters_to_nothing(self):
        self.assertEqual(self.paths(self.solo, key=self.member_key.id), set())
        self.assertEqual(self.paths(self.solo, key="not-a-number"), {"/solo-ok", "/solo-429"})

    def test_fifty_per_page(self):
        Request.objects.bulk_create(
            Request(token=self.solo_key, path=f"/bulk-{n}", status_code=200) for n in range(60)
        )
        page = self.as_user(self.solo).get(reverse("requests"))
        self.assertEqual(len(page.context["requests"]), 50)
        self.assertContains(page, "1\u201350 of 62")
        self.assertContains(page, "?page=2")
