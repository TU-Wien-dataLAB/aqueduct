import json

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from aqueduct.celery import delete_expired_files_and_batches
from gateway.tests.utils.base import GatewayFilesTestCase
from management.models import FileObject, Token


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
        file_url = reverse("gateway:file", kwargs={"file_id": file_id})

        # List files
        response = self.client.get(self.url_files, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 1)
        self.assertIn(file_id, [f["id"] for f in list_data["data"]])

        # Retrieve file metadata
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
        # Missing file only
        resp = self.client.post(self.url_files, {"purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        # Missing purpose only
        f = SimpleUploadedFile("a.jsonl", b"{}\n", content_type="application/json")
        resp = self.client.post(self.url_files, {"file": f}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_unsupported_purpose_and_bad_extension(self):
        """Unsupported purpose or wrong file extension yields 400."""
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": good, "purpose": "fine-tune"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        # wrong extension
        bad = SimpleUploadedFile("nope.txt", b"{}\n", content_type="text/plain")
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)

    def test_user_data_purpose(self):
        """File with purpose user_data should return 200."""
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": good, "purpose": "user_data"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)

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
        self.assertIn("File 'file' exceeds maximum size", resp.json()["error"])

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
        self.assertIn("Total file size exceeds maximum of 1MB", resp.json()["error"])

    @override_settings(AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB=1)
    def test_exceeded_storage_per_token(self):
        """Uploading a file when total storage per token is exceeded should fail."""
        token = Token.objects.first()

        # Setup: Create some files assigned to the token; 1 MB in total
        FileObject.objects.bulk_create(
            [
                FileObject(bytes=512 * 1024, created_at=42, token=token, purpose="batch"),
                FileObject(bytes=512 * 1024, created_at=43, token=token, purpose="batch"),
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
        ids = []
        for name in ("a.jsonl", "b.jsonl", "c.jsonl"):
            f = SimpleUploadedFile(name, b'{"custom_id": "bar"}\n', content_type="application/json")
            resp = self.client.post(
                self.url_files, {"file": f, "purpose": "batch"}, headers=self.headers
            )
            self.assertEqual(resp.status_code, 200)
            ids.append(resp.json()["id"])

        # Confirm all appear
        resp = self.client.get(self.url_files, headers=self.headers)
        data_ids = [f["id"] for f in resp.json()["data"]]
        self.assertCountEqual(data_ids, ids)

        # Delete the middle one and re-list
        mid = ids[1]
        file_url = reverse("gateway:file", kwargs={"file_id": mid})
        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        remaining = [ids[0], ids[2]]
        resp = self.client.get(self.url_files, headers=self.headers)
        self.assertCountEqual([f["id"] for f in resp.json()["data"]], remaining)

    def test_expires_at_and_cleanup_task(self):
        """Verify expires_at is set 1 week ahead and expired files get purged."""

        # Upload a file
        content = b'{"custom_id": 1}\n'
        f = SimpleUploadedFile("e.jsonl", content, content_type="application/json")
        resp = self.client.post(
            self.url_files, {"file": f, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Expires_at ~ now + 7 days
        now = timezone.now()
        expires_ts = data.get("expires_at")
        expected = int(
            (now + timezone.timedelta(days=settings.AQUEDUCT_FILES_API_EXPIRY_DAYS)).timestamp()
        )
        # allow small delta for execution time
        self.assertTrue(abs(expires_ts - expected) <= 5)

        file_id = data["id"]

        # Simulate expiration in the past and run cleanup task
        past = now - timezone.timedelta(days=8)
        FileObject.objects.filter(id=file_id).update(expires_at=int(past.timestamp()))
        delete_expired_files_and_batches()
        # record removed
        self.assertFalse(FileObject.objects.filter(id=file_id).exists())

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

    def test_batch_invalid_json(self):
        bad = SimpleUploadedFile(
            "bad.jsonl", b'{"custom_id": "bar"}\nnot json\n', content_type="application/json"
        )
        # unsupported purpose
        resp = self.client.post(
            self.url_files, {"file": bad, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
