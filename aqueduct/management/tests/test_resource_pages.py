import time
from pathlib import Path

from django.conf import settings
from django.test import override_settings
from django.urls import reverse

from gateway.tests.utils.test_runner import get_shared_mock_server
from management.models import Batch, FileObject, Token, VectorStore
from management.tests.helpers import RoleWorld, join, make_user
from mock_api.mock_configs import MockConfig

NOW = int(time.time())


class ResourcePagesTestBase(RoleWorld):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.teammate = make_user("teammate", cls.org)
        join(cls.teammate, cls.team)

    def setUp(self):
        super().setUp()
        self.mock_server = get_shared_mock_server()
        files_api = override_settings(AQUEDUCT_FILES_API_URL=self.mock_server.base_url)
        files_api.enable()
        self.addCleanup(files_api.disable)

    def file(self, token: Token, filename: str) -> FileObject:
        return FileObject.objects.create(
            id=f"file-{filename}",
            token=token,
            filename=filename,
            bytes=2048,
            created_at=NOW,
            purpose="batch",
        )

    def batch(self, token: Token, batch_id: str, status: str) -> Batch:
        return Batch.objects.create(
            id=batch_id,
            token=token,
            input_file=self.file(token, f"{batch_id}.jsonl"),
            completion_window="24h",
            created_at=NOW,
            endpoint="/v1/chat/completions",
            status=status,
        )


class FilesPageTest(ResourcePagesTestBase):
    def setUp(self):
        super().setUp()
        self.own_file = self.file(self.member_key, "prompts.jsonl")
        self.team_file = self.file(self.pipeline_key, "team-corpus.jsonl")
        self.foreign_file = self.file(self.solo_key, "secret.jsonl")

    def url(self, file: FileObject) -> str:
        return reverse("file_delete", args=[file.id])

    def test_lists_own_and_team_files_and_offers_delete_where_allowed(self):
        page = self.as_user(self.member).get(reverse("files"))
        self.assertContains(page, "prompts.jsonl")
        self.assertContains(page, "team-corpus.jsonl")
        self.assertNotContains(page, "secret.jsonl")
        self.assertContains(page, self.url(self.own_file))
        self.assertNotContains(page, self.url(self.team_file))
        self.assertContains(
            self.as_user(self.team_admin).get(reverse("files")), self.url(self.team_file)
        )

    def test_deleting_a_file(self):
        response = self.as_user(self.member).post(self.url(self.own_file), follow=True)
        self.assertRedirects(response, reverse("files"))
        self.assertContains(response, "prompts.jsonl deleted.")
        self.assertFalse(FileObject.objects.filter(id=self.own_file.id).exists())

    def test_only_files_of_manageable_keys(self):
        for user, file in ((self.member, self.team_file), (self.member, self.foreign_file)):
            with self.subTest(file=file.filename):
                self.assertEqual(self.as_user(user).post(self.url(file)).status_code, 404)
                self.assertTrue(FileObject.objects.filter(id=file.id).exists())

    def test_upstream_failure_keeps_the_file(self):
        with self.mock_server.patch_external_api(
            f"files/{self.own_file.id}",
            MockConfig(status_code=400, response_data={"error": {"message": "boom"}}),
        ):
            response = self.as_user(self.member).post(self.url(self.own_file), follow=True)
        self.assertContains(response, "Could not delete prompts.jsonl.")
        self.assertTrue(FileObject.objects.filter(id=self.own_file.id).exists())

    def test_a_file_gone_upstream_is_removed_here_too(self):
        with self.mock_server.patch_external_api(
            f"files/{self.own_file.id}",
            MockConfig(status_code=404, response_data={"error": {"message": "No such File"}}),
        ):
            self.as_user(self.member).post(self.url(self.own_file))
        self.assertFalse(FileObject.objects.filter(id=self.own_file.id).exists())

    def test_the_input_of_a_running_batch_stays(self):
        running = self.batch(self.member_key, "batch-busy", "in_progress")
        response = self.as_user(self.member).post(self.url(running.input_file), follow=True)
        self.assertContains(response, "batch-busy.jsonl feeds a running batch.")
        self.assertTrue(FileObject.objects.filter(id=running.input_file.id).exists())


class BatchesPageTest(ResourcePagesTestBase):
    def url(self, batch: Batch) -> str:
        return reverse("batch_cancel", args=[batch.id])

    def test_cancel_offered_for_running_batches_of_manageable_keys(self):
        running = self.batch(self.member_key, "batch-run", "in_progress")
        done = self.batch(self.member_key, "batch-done", "completed")
        team_run = self.batch(self.pipeline_key, "batch-team", "in_progress")
        page = self.as_user(self.member).get(reverse("batches"))
        self.assertContains(page, self.url(running))
        self.assertNotContains(page, self.url(done))
        self.assertNotContains(page, self.url(team_run))
        self.assertContains(page, "batch-team")

    def test_reads_every_batch_with_its_files_in_one_query(self):
        client = self.as_user(self.member)
        client.get(reverse("batches"))
        for number in range(3):
            batch = self.batch(self.member_key, f"batch-{number}", "completed")
            batch.output_file = self.file(self.member_key, f"batch-{number}-output.jsonl")
            batch.error_file = self.file(self.member_key, f"batch-{number}-errors.jsonl")
            batch.save()
            with self.subTest(batches=number + 1), self.assertNumQueries(8):
                page = client.get(reverse("batches"))
        self.assertContains(page, "file-batch-2-errors.jsonl")

    def test_cancelling_stores_the_upstream_status(self):
        running = self.batch(self.member_key, "batch-run", "in_progress")
        response = self.as_user(self.member).post(self.url(running), follow=True)
        self.assertRedirects(response, reverse("batches"))
        running.refresh_from_db()
        self.assertIn(running.status, ("cancelling", "cancelled"))

    def test_finished_or_foreign_batches_cannot_be_cancelled(self):
        done = self.batch(self.member_key, "batch-done", "completed")
        foreign = self.batch(self.solo_key, "batch-solo", "in_progress")
        for batch in (done, foreign):
            with self.subTest(batch=batch.id):
                self.assertEqual(self.as_user(self.member).post(self.url(batch)).status_code, 404)


class VectorStoresAndMCPTest(ResourcePagesTestBase):
    def test_vector_stores_list_and_detail(self):
        store = VectorStore.objects.create(
            id="vs_notes", token=self.member_key, name="notes", status="completed", created_at=NOW
        )
        page = self.as_user(self.member).get(reverse("vector_stores"))
        self.assertContains(page, reverse("vector_store_detail", args=[store.id]))
        detail = self.as_user(self.member).get(reverse("vector_store_detail", args=[store.id]))
        self.assertContains(detail, "No files in this vector store.")
        other = self.as_user(self.solo).get(reverse("vector_store_detail", args=[store.id]))
        self.assertContains(other, "Vector store not found")

    @override_settings(MCP_CONFIG_FILE_PATH=Path(settings.BASE_DIR).parent / "test-mcp.json")
    def test_mcp_servers_show_url_and_client_configuration(self):
        page = self.as_user(self.solo).get(reverse("mcp_servers"))
        self.assertTrue(page.context["mcp_servers"])
        for name in page.context["mcp_servers"]:
            self.assertContains(page, f"http://testserver/mcp-servers/{name}/mcp")
        self.assertContains(page, '"mcpServers"')
