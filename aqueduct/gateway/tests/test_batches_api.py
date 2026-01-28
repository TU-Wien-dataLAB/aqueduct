import json
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.utils import timezone
from openai import OpenAI

from gateway.tests.utils import _build_chat_headers, _build_chat_payload
from gateway.tests.utils.base import GatewayBatchesTestCase
from management.models import Batch, FileObject, Token


def mock_router():
    class DummyResult:
        def model_dump(self, **kwargs):
            return {"choices": [{"message": {"content": "dummy"}}]}

    class DummyEmbeddingResult:
        def model_dump(self, **kwargs):
            return {"data": [{"embedding": [1.0, 2.0]}]}

    m = MagicMock()
    m.acompletion = AsyncMock(return_value=DummyResult())
    m.atext_completion = AsyncMock(return_value=DummyResult())
    m.aembedding = AsyncMock(return_value=DummyEmbeddingResult())
    return m


class TestBatchesAPI(GatewayBatchesTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers.pop("Content-Type", None)
        cls.url_chat = reverse("gateway:v1_chat_completions")

    def tearDown(self):
        # Clean up all files and batches created during tests
        client = OpenAI(
            base_url=settings.AQUEDUCT_FILES_API_URL.rstrip("/") + "/v1",
            api_key=settings.AQUEDUCT_FILES_API_KEY or "unused",
        )

        # Delete all batches upstream and locally
        batch_objects = Batch.objects.all()
        for batch_obj in batch_objects:
            if batch_obj.remote_id:
                try:
                    client.batches.cancel(batch_obj.remote_id)
                except Exception:
                    pass
            batch_obj.delete()

        # Delete all files upstream and locally
        file_objects = FileObject.objects.all()
        for file_obj in file_objects:
            if file_obj.remote_id:
                try:
                    client.files.delete(file_obj.remote_id)
                except Exception:
                    pass  # Ignore deletions in teardown
            file_obj.delete()
        super().tearDown()

    def test_invalid_json(self):
        """POST /batches with invalid JSON returns 400."""
        # Send malformed JSON in the body
        resp = self.client.post(
            "/batches", data="{not: 'json'}", headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("error", body)
        # Should indicate invalid JSON
        self.assertTrue(
            "Invalid JSON" in body.get("error", ""),
            f"Unexpected error message: {body.get('error')}",
        )

    def test_nonexistent_input_file(self):
        """POST /batches with a non-existent file ID returns 404."""
        payload = {
            "input_file_id": "does_not_exist",
            "completion_window": "24h",
            "endpoint": self.url_chat,
        }
        resp = self.client.post(
            "/batches",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)
        err = resp.json()
        self.assertIn("error", err)
        error = err.get("error", {})
        self.assertEqual(error.get("message"), "Input file not found.")

    def test_list_empty_batches(self):
        """GET /batches when there are no batches should return empty list."""
        resp = self.client.get("/batches", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("data", data)
        self.assertEqual(data.get("data"), [])

    def test_list_batches_different_tokens(self):
        """Batches created under one token should not be visible under another."""
        # Prepare a simple JSONL file in batch API format
        payload = _build_chat_payload(
            self.model,
            messages=[{"role": "system", "content": "Hi"}, {"role": "user", "content": "Token1"}],
        )
        wrapped = {"custom_id": 1, "method": "POST", "url": self.url_chat, "body": payload}
        content = json.dumps(wrapped).encode() + b"\n"
        f1 = SimpleUploadedFile("t1.jsonl", content, content_type="application/jsonl")
        # Create batch under token1 (self.headers)
        resp = self.client.post("/files", {"file": f1, "purpose": "batch"}, headers=self.headers)
        fid1 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": fid1, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=self.headers,
            content_type="application/json",
        )
        b1 = resp.json()["id"]

        # Build headers for a second token (from fixture pk=2)
        token2 = Token.objects.get(pk=2)
        # Generate a fresh secret for token2 (update key_hash/preview) and authenticate with raw key
        secret2 = token2._set_new_key()
        token2.save(update_fields=["key_hash", "key_preview"])
        headers2 = _build_chat_headers(secret2)
        headers2.pop("Content-Type", None)

        # Create batch under token2
        f2 = SimpleUploadedFile("t2.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f2, "purpose": "batch"}, headers=headers2)
        fid2 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": fid2, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers2,
            content_type="application/json",
        )
        b2 = resp.json()["id"]

        # Token1 should see all batches
        resp = self.client.get("/batches", headers=self.headers)
        ids1 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids1, [b1, b2])

        # Token2 should see all batches
        resp = self.client.get("/batches", headers=headers2)
        ids2 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids2, [b2, b1])

    def test_max_user_batches_limit(self):
        """POST /batches should enforce MAX_USER_BATCHES and reject the fourth batch."""
        # Upload a simple JSONL file for batching in batch API format
        payload = _build_chat_payload(
            self.model, [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}]
        )
        wrapped = {"custom_id": 1, "method": "POST", "url": self.url_chat, "body": payload}
        content = (json.dumps(wrapped) + "\n").encode("utf-8")
        upload = SimpleUploadedFile("limit.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(
            "/files", {"file": upload, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        batch_payload = {
            "input_file_id": file_id,
            "completion_window": "24h",
            "endpoint": self.url_chat,
        }

        # Allowed up to MAX_USER_BATCHES batches
        created_ids = []
        for _ in range(settings.MAX_USER_BATCHES):
            ok = self.client.post(
                "/batches",
                data=json.dumps(batch_payload),
                headers=self.headers,
                content_type="application/json",
            )
            self.assertEqual(ok.status_code, 200)
            created_ids.append(ok.json()["id"])

        # The next (fourth) batch should be rejected with 403
        over = self.client.post(
            "/batches",
            data=json.dumps(batch_payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(over.status_code, 403)
        err = over.json()
        error = err.get("error", {})
        self.assertEqual(error.get("message"), f"Batch limit reached ({settings.MAX_USER_BATCHES})")

        # Verify that only MAX_USER_BATCHES batches exist
        resp = self.client.get("/batches", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        ids_list = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids_list, created_ids)

    @patch("gateway.views.batches.get_files_api_client")
    def test_max_user_batches_limit_after_cancel(self, mock_get_files_client):
        """Cancelling an active batch frees up a slot for a new batch."""
        # Create mock for OpenAI client methods
        mock_client = MagicMock()
        mock_get_files_client.return_value = mock_client

        # Create mock for batch.create() response

        mock_batch = MagicMock()
        mock_batch.id = "batch-test-123"
        mock_batch.status = "in_progress"
        mock_batch.input_file_id = "file-test-123"
        mock_batch.request_counts = MagicMock()
        mock_batch.request_counts.model_dump = MagicMock(return_value={})
        mock_batch.completed_at = None
        mock_batch.failed_at = None
        mock_batch.cancelled_at = None
        mock_batch.cancelling_at = None
        mock_batch.expired_at = None
        mock_batch.expires_at = None
        mock_batch.finalizing_at = None
        mock_batch.in_progress_at = None
        mock_batch.model_dump = MagicMock(
            return_value={
                "id": mock_batch.id,
                "status": mock_batch.status,
                "input_file_id": mock_batch.input_file_id,
                "request_counts": {},
                "completed_at": None,
                "failed_at": None,
                "cancelled_at": None,
                "expired_at": None,
                "expires_at": None,
                "cancelling_at": None,
                "finalizing_at": None,
                "in_progress_at": None,
            }
        )
        mock_client.batches.create = AsyncMock(return_value=mock_batch)

        # Create mock for batch.cancel() response - this was causing 502 errors
        mock_cancelled_batch = MagicMock()
        mock_cancelled_batch.id = "batch-test-123"
        mock_cancelled_batch.status = "cancelled"
        mock_cancelled_batch.cancelling_at = int(timezone.now().timestamp())
        mock_cancelled_batch.cancelled_at = int(timezone.now().timestamp())
        mock_cancelled_batch.input_file_id = "file-test-123"
        mock_cancelled_batch.completed_at = None
        mock_cancelled_batch.failed_at = None
        mock_cancelled_batch.expired_at = None
        mock_cancelled_batch.expires_at = None
        mock_cancelled_batch.finalizing_at = None
        mock_cancelled_batch.in_progress_at = None
        mock_cancelled_batch.model_dump = MagicMock(
            return_value={
                "id": mock_cancelled_batch.id,
                "status": mock_cancelled_batch.status,
                "input_file_id": mock_cancelled_batch.input_file_id,
                "cancelling_at": mock_cancelled_batch.cancelling_at,
                "cancelled_at": mock_cancelled_batch.cancelled_at,
                "completed_at": None,
                "failed_at": None,
                "expired_at": None,
                "expires_at": None,
                "finalizing_at": None,
                "in_progress_at": None,
            }
        )
        mock_client.batches.cancel = AsyncMock(return_value=mock_cancelled_batch)

        # Create mock for files.create() response (used by file upload)
        mock_file = MagicMock()
        mock_file.id = "file-test-123"
        mock_file.filename = "limit.jsonl"
        mock_file.size = 100
        mock_file.purpose = "batch"
        mock_file.created_at = int(timezone.now().timestamp())
        mock_file.model_dump = MagicMock(
            return_value={
                "id": mock_file.id,
                "filename": mock_file.filename,
                "size": mock_file.size,
                "purpose": mock_file.purpose,
                "created_at": mock_file.created_at,
            }
        )
        mock_client.files.create = AsyncMock(return_value=mock_file)

        # Upload a simple JSONL file for batching
        payload = _build_chat_payload(
            self.model, [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}]
        )
        wrapped = {"custom_id": 1, "method": "POST", "url": self.url_chat, "body": payload}
        content = (json.dumps(wrapped) + "\n").encode("utf-8")
        upload = SimpleUploadedFile("limit.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(
            "/files", {"file": upload, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        batch_payload = {
            "input_file_id": file_id,
            "completion_window": "24h",
            "endpoint": self.url_chat,
        }

        # Create up to the limit
        created = []
        for _ in range(settings.MAX_USER_BATCHES):
            ok = self.client.post(
                "/batches",
                data=json.dumps(batch_payload),
                headers=self.headers,
                content_type="application/json",
            )
            self.assertEqual(ok.status_code, 200)
            created.append(ok.json()["id"])

        # Next batch should be blocked
        blocked = self.client.post(
            "/batches",
            data=json.dumps(batch_payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(blocked.status_code, 403)

        # Cancel one of the active batches
        to_cancel = created[0]
        cancel_resp = self.client.post(f"/batches/{to_cancel}/cancel", headers=self.headers)
        self.assertEqual(cancel_resp.status_code, 200)

        # Now a new batch should succeed
        again = self.client.post(
            "/batches",
            data=json.dumps(batch_payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(again.status_code, 200)

    def test_get_nonexistent_batch(self):
        """GET /batches/{id} for a non-existent batch returns 404."""
        resp = self.client.get("/batches/nonexistent", headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        error = body.get("error", {})
        self.assertEqual(error.get("message"), "Batch not found.")

    def test_cancel_nonexistent_batch(self):
        """POST /batches/{id}/cancel for non-existent batch returns 404."""
        resp = self.client.post("/batches/nonexistent/cancel", headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        error = body.get("error", {})
        self.assertEqual(error.get("message"), "Batch not found.")
