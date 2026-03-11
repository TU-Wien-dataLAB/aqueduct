import json
from datetime import timedelta
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from openai.types import FileObject
from openai.types.file_deleted import FileDeleted

from aqueduct.celery import delete_expired_files_and_batches
from gateway.tests.utils.base import GatewayFilesTestCase
from management.models import FileObject as FileObjectModel
from management.models import ServiceAccount, Team, Token
from mock_api.mock_configs import MockConfig


def create_mock_file(
    id_suffix: str = "123",
    filename: str = "test.jsonl",
    purpose: Literal[
        "assistants",
        "assistants_output",
        "batch",
        "batch_output",
        "fine-tune",
        "fine-tune-results",
        "vision",
        "user_data",
    ] = "batch",
    bytes_size: int = 100,
    status: Literal["uploaded", "processed", "error"] = "processed",
) -> FileObject:
    """Create a mock file object matching OpenAI's FileObject structure."""
    return FileObject(
        id=f"file-mock-{id_suffix}",
        filename=filename,
        bytes=bytes_size,
        purpose=purpose,
        created_at=int(timezone.now().timestamp()),
        expires_at=None,
        status=status,
        status_details=None,
        object="file",
    )


def create_mock_files_client():
    """Create a fully mocked OpenAI client for files API."""
    mock_client = MagicMock()
    file_counter = [0]

    async def mock_create(*args, **kwargs):
        file_counter[0] += 1
        return create_mock_file(id_suffix=str(file_counter[0]))

    mock_client.files.create = AsyncMock(side_effect=mock_create)
    mock_client.files.retrieve = AsyncMock(return_value=create_mock_file())

    mock_delete_response = FileDeleted(id="file-mock-123", deleted=True, object="file")
    mock_client.files.delete = AsyncMock(return_value=mock_delete_response)

    mock_content_response = MagicMock()
    mock_content_response.content = b'{"custom_id": "bar"}\n'
    mock_client.files.content = AsyncMock(return_value=mock_content_response)

    return mock_client


class TestFilesAPI(GatewayFilesTestCase):
    def test_file_lifecycle(self):
        """Test uploading, listing, retrieving, downloading, and deleting a file."""
        content = b'{"custom_id": "bar"}\n{"custom_id": "123"}\n{"custom_id": "baz"}\n{"custom_id": "1234"}\n'
        upload_file = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")

        # Upload file
        response = self.client.post(
            self.url_files, {"file": upload_file, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 200, f"Upload failed: {response.json()}")
        upload_data = response.json()
        file_id = upload_data["id"]
        file_obj = FileObjectModel.objects.get()
        self.assertEqual(file_obj.id, file_id)
        self.assertEqual(file_obj.purpose, "batch")

        # List files
        response = self.client.get(self.url_files, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 1)
        self.assertIn(file_id, [f["id"] for f in list_data["data"]])

        # Retrieve file metadata
        file_url = reverse("gateway:file", kwargs={"file_id": file_id})
        response = self.client.get(file_url, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        meta_data = response.json()
        self.assertEqual(meta_data["filename"], upload_data["filename"])

        # Download file content
        # Note: Content may be reformatted by the processing decorator (whitespace normalization)
        content_url = reverse("gateway:file_content", kwargs={"file_id": file_id})
        response = self.client.get(content_url, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        # Verify files contain the same custom_ids in order
        response_lines = response.content.splitlines()
        original_lines = content.splitlines()
        self.assertEqual(len(response_lines), len(original_lines))
        for resp_line, orig_line in zip(response_lines, original_lines):
            resp_data = json.loads(resp_line)
            orig_data = json.loads(orig_line)
            self.assertEqual(resp_data["custom_id"], orig_data["custom_id"])

        # Delete file
        response = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        delete_data = response.json()
        self.assertTrue(delete_data.get("deleted"))
        self.assertEqual(FileObjectModel.objects.filter(id=file_id).count(), 0)

        # Ensure file is no longer listed
        response = self.client.get(self.url_files, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 0)
        self.assertNotIn(file_id, [f["id"] for f in list_data["data"]])

    def test_validation_errors(self):
        """Missing or bad parameters should return 400."""
        # Missing both file and purpose
        resp = self.client.post(self.url_files, {}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "file: Field required, purpose: Field required", resp.json()["error"]["message"]
        )
        # Missing file only
        resp = self.client.post(self.url_files, {"purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file: Field required", resp.json()["error"]["message"])
        # Missing purpose only
        f = SimpleUploadedFile("a.jsonl", b"{}\n", content_type="application/json")
        resp = self.client.post(self.url_files, {"file": f}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("purpose: Field required", resp.json()["error"]["message"])

    def test_unsupported_purpose_and_bad_extension(self):
        """Unsupported purpose or wrong file extension yields 400."""
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": good, "purpose": "fine-tune"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "purpose: Input should be 'assistants', 'batch' or 'user_data'",
            resp.json()["error"]["message"],
        )
        # wrong extension
        bad = SimpleUploadedFile("nope.txt", b"{}\n", content_type="text/plain")
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "Only .jsonl files are currently supported for purpose 'batch'",
            resp.json()["error"]["message"],
        )

    @patch("gateway.views.files.get_files_api_client")
    def test_user_data_purpose(self, mock_get_files_client):
        """File with purpose user_data should return 200."""
        mock_get_files_client.return_value = create_mock_files_client()
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        resp = self.client.post(
            self.url_files, {"file": good, "purpose": "user_data"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        file_obj = FileObjectModel.objects.get()
        self.assertEqual(file_obj.purpose, "user_data")

    @override_settings(AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB=1)
    def test_oversize_file(self):
        """File bigger than `AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB` should be rejected."""
        max_mb = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB
        big = b"a" * (max_mb * 1024 * 1024 + 1)
        f = SimpleUploadedFile("big.jsonl", big, content_type="application/json")
        resp = self.client.post(
            self.url_files, {"file": f, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 413)
        self.assertIn("File 'file' exceeds maximum size", resp.json()["error"]["message"])

    @override_settings(AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB=1)
    def test_exceeded_total_files_size(self):
        """Uploading files with total size bigger than `AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB` should fail."""
        content = b"a" * (600 * 1024)  # 600 kB => two such files exceed the limit of 1 MB
        data = {
            "test_file_1": SimpleUploadedFile("test_file_1.txt", content),
            "test_file_2": SimpleUploadedFile("test_file_2.txt", content),
            "purpose": "user_data",
        }

        # Upload files
        resp = self.client.post(self.url_files, data, headers=self.headers)
        self.assertEqual(resp.status_code, 413)
        self.assertIn("Total file size exceeds maximum of 1MB", resp.json()["error"]["message"])

    @override_settings(AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB=1)
    def test_exceeded_storage_per_token(self):
        """Uploading a file when total storage per token is exceeded should fail."""
        token = Token.objects.first()

        # Setup: Create some files assigned to the token; 1 MB in total
        FileObjectModel.objects.bulk_create(
            [
                FileObjectModel(
                    id="file-storage-1",
                    bytes=512 * 1024,
                    created_at=42,
                    token=token,
                    purpose="batch",
                    preview="",
                ),
                FileObjectModel(
                    id="file-storage-2",
                    bytes=512 * 1024,
                    created_at=43,
                    token=token,
                    purpose="batch",
                    preview="",
                ),
            ]
        )

        content = b"a" * 1024  # 1 kB
        upload_file = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        # Upload file
        resp = self.client.post(
            self.url_files, {"file": upload_file, "purpose": "user_data"}, headers=self.headers
        )

        self.assertEqual(resp.status_code, 413, resp.json())
        error = resp.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertIn("Total files size exceeds 1MB limit", error["message"])

    def test_not_found_cases(self):
        """GET/DELETE on nonexistent file returns 404."""
        nonexistent_file_url = reverse("gateway:file", kwargs={"file_id": "nonexistent"})
        nonexistent_content_url = reverse("gateway:file_content", kwargs={"file_id": "nonexistent"})

        for method in ("get", "delete"):
            resp = getattr(self.client, method)(nonexistent_file_url, headers=self.headers)
            self.assertEqual(resp.status_code, 404)
        resp = self.client.get(nonexistent_content_url, headers=self.headers)
        self.assertEqual(resp.status_code, 404)

    def test_list_empty_and_bulk_operations(self):
        """Listing on empty, multiple uploads, and deletion among many."""

        # Initially empty
        resp = self.client.get(self.url_files, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"], [])

        # Upload several files
        file_data = {
            "filename": "test.jsonl",
            "bytes": 100,
            "purpose": "batch",
            "created_at": 1741476542,
            "expires_at": None,
            "status": "processed",
            "status_details": None,
            "object": "file",
        }

        ids = []
        for i, name in enumerate(["a.jsonl", "b.jsonl", "c.jsonl"]):
            mock_resp = MockConfig(
                response_data=FileObject(id=f"file-mock-{i}", **file_data).model_dump()
            )
            f = SimpleUploadedFile(name, b'{"custom_id": "bar"}\n', content_type="application/json")
            with self.mock_server.patch_external_api(self.url_files, mock_resp):
                resp = self.client.post(
                    self.url_files, {"file": f, "purpose": "batch"}, headers=self.headers
                )
            self.assertEqual(resp.status_code, 200)
            ids.append(resp.json()["id"])

        # Confirm all appear and were created in the database
        resp = self.client.get(self.url_files, headers=self.headers)
        resp_ids = [f["id"] for f in resp.json()["data"]]
        self.assertCountEqual(resp_ids, ids)
        data_ids = FileObjectModel.objects.values_list("id", flat=True)
        self.assertCountEqual(data_ids, ids)

        # Delete the middle one and re-list
        mid = ids[1]
        file_url = reverse("gateway:file", kwargs={"file_id": mid})
        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        remaining = [ids[0], ids[2]]
        resp = self.client.get(self.url_files, headers=self.headers)
        resp_ids = [f["id"] for f in resp.json()["data"]]
        self.assertCountEqual(resp_ids, remaining)
        data_ids = FileObjectModel.objects.values_list("id", flat=True)
        self.assertCountEqual(data_ids, remaining)

    def test_expires_at_and_cleanup_task(self):
        """Verify expires_at is set 1 week ahead and expired files get purged."""
        # Upload a file
        content = b'{"custom_id": 1}\n'
        f = SimpleUploadedFile("e.jsonl", content, content_type="application/json")
        resp = self.client.post(
            self.url_files, {"file": f, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]
        file_obj = FileObjectModel.objects.get(id=file_id)

        # Simulate expiration in the past and run cleanup task
        expires_ts = file_obj.expires_at
        now_ts = int(timezone.now().timestamp())
        past_ts = int(expires_ts - timedelta(days=8).total_seconds())
        self.assertLess(past_ts, now_ts)
        FileObjectModel.objects.filter(id=file_id).update(expires_at=past_ts)
        delete_expired_files_and_batches()
        # record removed
        self.assertFalse(FileObjectModel.objects.filter(id=file_id).exists())

    def test_batch_duplicate_custom_ids(self):
        bad = SimpleUploadedFile(
            "bad.jsonl",
            b'{"custom_id": "bar"}\n{"custom_id": "bar"}\n',
            content_type="application/json",
        )
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "Batch file validation failed: Duplicate custom_id found at line 2",
            resp.json()["error"]["message"],
        )

    def test_batch_no_custom_ids(self):
        bad = SimpleUploadedFile(
            "bad.jsonl",
            b'{"custom_id": "bar"}\n{"something_id": "bar"}\n',
            content_type="application/json",
        )
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "Batch file validation failed: No custom_id found at line 2",
            resp.json()["error"]["message"],
        )

    def test_batch_invalid_json(self):
        bad = SimpleUploadedFile(
            "bad.jsonl", b'{"custom_id": "bar"}\nnot json\n', content_type="application/json"
        )
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(
            "Batch file validation failed: Invalid JSON at line 2", resp.json()["error"]["message"]
        )

    # TODO
    @patch("gateway.views.files.get_files_api_client")
    def test_service_account_file_operations(self, mock_get_client):
        """Test that the files API works with service account tokens."""
        mock_client = MagicMock()

        async def mock_file_create(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                bytes=100,
                filename="test.txt",
                purpose="user_data",
                created_at=int(timezone.now().timestamp()),
                expires_at=None,
                model_dump=MagicMock(
                    return_value={
                        "id": "file-remote-new",
                        "bytes": 100,
                        "filename": "test.txt",
                        "purpose": "user_data",
                        "created_at": int(timezone.now().timestamp()),
                        "expires_at": None,
                    }
                ),
            )

        async def mock_file_retrieve(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                bytes=100,
                filename="test.txt",
                purpose="user_data",
                created_at=int(timezone.now().timestamp()),
                expires_at=None,
                model_dump=MagicMock(
                    return_value={
                        "id": "file-remote-new",
                        "bytes": 100,
                        "filename": "test.txt",
                        "purpose": "user_data",
                        "created_at": int(timezone.now().timestamp()),
                        "expires_at": None,
                    }
                ),
            )

        async def mock_file_delete(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                deleted=True,
                model_dump=MagicMock(return_value={"id": "file-remote-new", "deleted": True}),
            )

        async def mock_file_content(*args, **kwargs):
            return MagicMock(content=b"test file content")

        mock_client.files.create = AsyncMock(side_effect=mock_file_create)
        mock_client.files.retrieve = AsyncMock(side_effect=mock_file_retrieve)
        mock_client.files.delete = AsyncMock(side_effect=mock_file_delete)
        mock_client.files.content = AsyncMock(side_effect=mock_file_content)
        mock_get_client.return_value = mock_client

        team = Team.objects.get(name="Whale")
        service_account = ServiceAccount.objects.create(team=team, name="Test Service Account")

        token = Token.objects.get(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN))
        token.service_account = service_account
        token.save()

        try:
            file = SimpleUploadedFile("test.txt", b"test file content", content_type="text/plain")
            files_url = reverse("gateway:files")
            resp = self.client.post(
                files_url, data={"file": file, "purpose": "user_data"}, headers=self.headers
            )
            self.assertEqual(resp.status_code, 200, f"Upload failed: {resp.json()}")
            file_id = resp.json()["id"]

            resp = self.client.get(files_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"List failed: {resp.json()}")
            self.assertEqual(len(resp.json()["data"]), 1)

            file_detail_url = reverse("gateway:file", kwargs={"file_id": file_id})
            resp = self.client.get(file_detail_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Retrieve failed: {resp.json()}")

            file_content_url = reverse("gateway:file_content", kwargs={"file_id": file_id})
            resp = self.client.get(file_content_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Content failed: {resp.content}")
            self.assertEqual(resp.content, b"test file content")

            resp = self.client.delete(file_detail_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Delete failed: {resp.json()}")

        finally:
            token.service_account = None
            token.save()
            service_account.delete()
