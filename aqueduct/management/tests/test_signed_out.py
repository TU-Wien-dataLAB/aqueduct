import time

from django.apps import apps
from django.urls import URLPattern, reverse

from management import urls
from management.models import Batch, FileObject, VectorStore
from management.tests.helpers import RoleWorld

POST_ONLY = {"token_regenerate"}
FORM = {"name": "taken-over", "email": "solo@example.com", "action": "remove", "profile_id": "1"}


class SignedOutTest(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        now = int(time.time())
        file = FileObject.objects.create(
            id="file-x", token=cls.solo_key, filename="x.jsonl", bytes=1, created_at=now
        )
        Batch.objects.create(
            id="batch-x",
            token=cls.solo_key,
            input_file=file,
            completion_window="24h",
            created_at=now,
            endpoint="/v1/chat/completions",
            status="in_progress",
        )
        VectorStore.objects.create(
            id="vs_x", token=cls.solo_key, name="x", status="completed", created_at=now
        )

    def route_url(self, pattern: URLPattern) -> str:
        ids = {"id": self.team.id, "service_account_id": self.pipeline.id}
        if pattern.name.startswith("token"):
            ids["id"] = self.solo_key.id
        elif pattern.name.startswith("file"):
            ids["id"] = "file-x"
        elif pattern.name.startswith("batch"):
            ids["id"] = "batch-x"
        elif pattern.name.startswith("vector_store"):
            ids["id"] = "vs_x"
        return reverse(pattern.name, kwargs={k: ids[k] for k in pattern.pattern.converters})

    def stored(self) -> dict[str, list[dict]]:
        return {
            model._meta.label: list(model.objects.order_by("pk").values())
            for model in apps.get_models()
        }

    def test_every_page_sends_a_signed_out_visitor_to_sign_in_and_changes_nothing(self):
        self.client.raise_request_exception = False
        before = self.stored()
        routes = [p for p in urls.urlpatterns if isinstance(p, URLPattern) and p.name]
        self.assertGreater(len(routes), 25)
        self.assertIn("tokens", {pattern.name for pattern in routes})
        for pattern in routes:
            url = self.route_url(pattern)
            for method, response in (
                ("get", self.client.get(url)),
                ("post", self.client.post(url, FORM)),
            ):
                with self.subTest(route=pattern.name, method=method):
                    if method == "get" and pattern.name in POST_ONLY:
                        self.assertEqual(response.status_code, 405)
                        continue
                    self.assertRedirects(
                        response, f"{reverse('sso')}?next={url}", fetch_redirect_response=False
                    )
        self.assertEqual(self.stored(), before)
