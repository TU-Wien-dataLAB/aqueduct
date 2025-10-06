from django.core.files.uploadedfile import SimpleUploadedFile

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayFilesTestCase


class TestFilesAPI(GatewayFilesTestCase):
    def setUp(self):
        super().setUp()
        # Prepare auth headers for file API
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        # Remove Content-Type header to allow multipart file upload
        headers.pop("Content-Type", None)
        self.headers = headers

    def test_file_lifecycle(self):
        """Test uploading, listing, retrieving, downloading, and deleting a file."""
        content = b'{"custom_id": "bar"}\n{"custom_id": "123"}\n{"custom_id": "baz"}\n{"custom_id": "1234"}\n'
        upload_file = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        # Upload file
        response = self.client.post(
            "/files", {"file": upload_file, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 200, f"Upload failed: {response.json()}")
        upload_data = response.json()
        file_id = upload_data["id"]

        # List files
        response = self.client.get("/files", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 1)
        self.assertIn(file_id, [f["id"] for f in list_data["data"]])

        # Retrieve file metadata
        response = self.client.get(f"/files/{file_id}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        meta_data = response.json()
        self.assertEqual(meta_data["filename"], upload_data["filename"])

        # Download file content
        response = self.client.get(f"/files/{file_id}/content", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, content)

        # Delete file
        response = self.client.delete(f"/files/{file_id}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        delete_data = response.json()
        self.assertTrue(delete_data.get("deleted"))

        # Ensure file is no longer listed
        response = self.client.get("/files", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 0)
        self.assertNotIn(file_id, [f["id"] for f in list_data["data"]])

    def test_validation_errors(self):
        """Missing or bad parameters should return 400."""
        # Missing both file and purpose
        resp = self.client.post("/files", {}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        # Missing file only
        resp = self.client.post("/files", {"purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        # Missing purpose only
        f = SimpleUploadedFile("a.jsonl", b"{}\n", content_type="application/json")
        resp = self.client.post("/files", {"file": f}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_unsupported_and_bad_extension(self):
        """Unsupported purpose or wrong file extension yields 400."""
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        # unsupported purpose
        resp = self.client.post(
            "/files", {"file": good, "purpose": "fine-tune"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)
        # wrong extension
        bad = SimpleUploadedFile("nope.txt", b"{}\n", content_type="text/plain")
        resp = self.client.post("/files", {"file": bad, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_user_data_purpose(self):
        """File with purpose user_data should return 200."""
        good = SimpleUploadedFile("ok.jsonl", b"{}\n", content_type="application/json")
        # unsupported purpose
        resp = self.client.post(
            "/files", {"file": good, "purpose": "user_data"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)

    def test_oversize_file(self):
        """File >8MB should be rejected."""
        from django.conf import settings

        max_mb = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB
        big = b"a" * (max_mb * 1024 * 1024 + 1)
        f = SimpleUploadedFile("big.jsonl", big, content_type="application/json")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_not_found_cases(self):
        """GET/DELETE on nonexistent file returns 404."""
        for method in ("get", "delete"):
            resp = getattr(self.client, method)("/files/nonexistent", headers=self.headers)
            self.assertEqual(resp.status_code, 404)
        resp = self.client.get("/files/nonexistent/content", headers=self.headers)
        self.assertEqual(resp.status_code, 404)

    def test_list_empty_and_bulk_operations(self):
        """Listing on empty, multiple uploads, and deletion among many."""
        # Initially empty
        resp = self.client.get("/files", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"], [])

        # Upload several files
        ids = []
        for name in ("a.jsonl", "b.jsonl", "c.jsonl"):
            f = SimpleUploadedFile(name, b'{"custom_id": "bar"}\n', content_type="application/json")
            resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=self.headers)
            self.assertEqual(resp.status_code, 200)
            ids.append(resp.json()["id"])

        # Confirm all appear
        resp = self.client.get("/files", headers=self.headers)
        data_ids = [f["id"] for f in resp.json()["data"]]
        self.assertCountEqual(data_ids, ids)

        # Delete the middle one and re-list
        mid = ids[1]
        resp = self.client.delete(f"/files/{mid}", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        remaining = [ids[0], ids[2]]
        resp = self.client.get("/files", headers=self.headers)
        self.assertCountEqual([f["id"] for f in resp.json()["data"]], remaining)

    def test_expires_at_and_cleanup_task(self):
        """Verify expires_at is set 1 week ahead and expired files get purged."""
        from django.conf import settings
        from django.utils import timezone

        from aqueduct.celery import delete_expired_files_and_batches
        from management.models import FileObject

        # Upload a file
        content = b'{"custom_id": 1}\n'
        f = SimpleUploadedFile("e.jsonl", content, content_type="application/json")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=self.headers)
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
        obj = FileObject.objects.get(id=file_id)
        file_path = obj.path()

        # Simulate expiration in the past and run cleanup task
        past = now - timezone.timedelta(days=8)
        FileObject.objects.filter(id=file_id).update(expires_at=int(past.timestamp()))
        # ensure file exists before cleanup
        self.assertTrue(file_path.exists())
        delete_expired_files_and_batches()
        # record removed and file deleted
        self.assertFalse(FileObject.objects.filter(id=file_id).exists())
        self.assertFalse(file_path.exists())

    def test_batch_duplicate_custom_ids(self):
        bad = SimpleUploadedFile(
            "bad.jsonl",
            b'{"custom_id": "bar"}\n{"custom_id": "bar"}\n',
            content_type="application/json",
        )
        # unsupported purpose
        resp = self.client.post("/files", {"file": bad, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_batch_no_custom_ids(self):
        bad = SimpleUploadedFile(
            "bad.jsonl",
            b'{"custom_id": "bar"}\n{"something_id": "bar"}\n',
            content_type="application/json",
        )
        # unsupported purpose
        resp = self.client.post("/files", {"file": bad, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_batch_invalid_json(self):
        bad = SimpleUploadedFile(
            "bad.jsonl", b'{"custom_id": "bar"}\nnot json\n', content_type="application/json"
        )
        # unsupported purpose
        resp = self.client.post("/files", {"file": bad, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
