import json
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.utils import timezone

from gateway.tests.utils import _build_chat_headers, _build_chat_payload
from gateway.tests.utils.base import GatewayBatchesTestCase
from management.models import Token


def make_mock_batch(id_suffix: str, status: str = "in_progress") -> MagicMock:
    """Helper to create a mock batch with given ID suffix and status."""
    mock_batch = MagicMock()
    mock_batch.id = f"batch-mock-{id_suffix}"
    mock_batch.status = status
    mock_batch.input_file_id = "file-mock-123"
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

    if status == "cancelled":
        mock_batch.cancelling_at = int(timezone.now().timestamp())
        mock_batch.cancelled_at = int(timezone.now().timestamp())

    mock_batch.model_dump = MagicMock(
        return_value={
            "id": mock_batch.id,
            "status": mock_batch.status,
            "input_file_id": mock_batch.input_file_id,
            "request_counts": {},
            "completed_at": None,
            "failed_at": None,
            "cancelled_at": mock_batch.cancelled_at,
            "expired_at": None,
            "expires_at": None,
            "cancelling_at": mock_batch.cancelling_at,
            "finalizing_at": None,
            "in_progress_at": None,
        }
    )
    return mock_batch


def create_mock_batch_client():
    """Create a fully mocked OpenAI client for files/batches API."""
    mock_client = MagicMock()

    # Mock file creation response
    mock_file = MagicMock()
    mock_file.id = "file-mock-123"
    mock_file.filename = "test.jsonl"
    mock_file.size = 100
    mock_file.bytes = 100
    mock_file.purpose = "batch"
    mock_file.created_at = int(timezone.now().timestamp())
    mock_file.expires_at = None
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
    mock_client.files.retrieve = AsyncMock(return_value=mock_file)

    # Counter for batch IDs to make them unique
    batch_counter = [0]

    async def make_batch_create_response(*args, **kwargs):
        batch_counter[0] += 1
        return make_mock_batch(str(batch_counter[0]), "in_progress")

    mock_client.batches.create = AsyncMock(side_effect=make_batch_create_response)

    # Mock batch response for retrieve and cancel
    async def make_batch_cancel_response(batch_id, *args, **kwargs):
        # Return the same batch ID but with cancelled status
        return make_mock_batch(batch_id.split("-")[-1], "cancelled")

    mock_client.batches.retrieve = AsyncMock(side_effect=make_batch_cancel_response)
    mock_client.batches.cancel = AsyncMock(side_effect=make_batch_cancel_response)

    return mock_client


def create_mock_cancelled_batch_client():
    """Mock client with batches.cancel() returning cancelled batch."""
    mock_client = create_mock_batch_client()

    mock_cancelled_batch = MagicMock()
    mock_cancelled_batch.id = "batch-mock-123"
    mock_cancelled_batch.status = "cancelled"
    mock_cancelled_batch.cancelling_at = int(timezone.now().timestamp())
    mock_cancelled_batch.cancelled_at = int(timezone.now().timestamp())
    mock_cancelled_batch.input_file_id = "file-mock-123"
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

    return mock_client


class TestBatchesAPI(GatewayBatchesTestCase):
    url_chat = reverse("gateway:v1_chat_completions")

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

    @patch("gateway.views.batches.get_files_api_client")
    def test_nonexistent_input_file(self, mock_get_files_client):
        """POST /batches with a non-existent file ID returns 404."""
        mock_get_files_client.return_value = create_mock_batch_client()
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

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_list_batches_different_tokens(
        self, mock_get_files_client_batches, mock_get_files_client_files
    ):
        """Batches created under one token should not be visible under another."""
        mock_get_files_client_files.return_value = create_mock_batch_client()
        mock_get_files_client_batches.return_value = create_mock_batch_client()
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

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_max_user_batches_limit(
        self, mock_get_files_client_batches, mock_get_files_client_files
    ):
        """POST /batches should enforce MAX_USER_BATCHES and reject the fourth batch."""
        mock_get_files_client_files.return_value = create_mock_batch_client()
        mock_get_files_client_batches.return_value = create_mock_batch_client()
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
    @patch("gateway.views.files.get_files_api_client")
    def test_max_user_batches_limit_after_cancel(
        self, mock_get_files_client_batches, mock_get_files_client_files
    ):
        """Cancelling an active batch frees up a slot for a new batch."""
        mock_get_files_client_files.return_value = create_mock_batch_client()
        mock_get_files_client_batches.return_value = create_mock_cancelled_batch_client()

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
