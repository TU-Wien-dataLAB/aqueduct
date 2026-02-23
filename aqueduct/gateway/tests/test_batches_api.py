import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.utils import timezone
from openai.types import Batch, FileObject
from openai.types.batch_request_counts import BatchRequestCounts

from gateway.tests.utils import _build_chat_headers, _build_chat_payload
from gateway.tests.utils.base import GatewayBatchesTestCase
from management.models import Org, ServiceAccount, Team, Token, UserProfile


def make_mock_batch(
    id_suffix: str,
    status: Literal[
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    ] = "in_progress",
) -> Batch:
    """Helper to create a mock batch with given ID suffix and status using OpenAI types."""
    now = int(timezone.now().timestamp())

    cancelled_at = None
    cancelling_at = None

    if status == "cancelled":
        cancelling_at = now
        cancelled_at = now

    return Batch(
        id=f"batch-mock-{id_suffix}",
        status=status,
        input_file_id="file-mock-123",
        request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
        completed_at=None,
        failed_at=None,
        cancelled_at=cancelled_at,
        cancelling_at=cancelling_at,
        expired_at=None,
        expires_at=None,
        finalizing_at=None,
        in_progress_at=None,
        completion_window="24h",
        created_at=now,
        endpoint="/v1/chat/completions",
        error_file_id=None,
        errors=None,
        metadata=None,
        model=None,
        object="batch",
        output_file_id=None,
        usage=None,
    )


def create_mock_batch_client():
    """Create a fully mocked OpenAI client for files/batches API."""
    mock_client = MagicMock()

    # Mock file creation response
    mock_file = FileObject(
        id="file-mock-123",
        filename="test.jsonl",
        bytes=100,
        purpose="batch",
        created_at=int(timezone.now().timestamp()),
        expires_at=None,
        status="processed",
        status_details=None,
        object="file",
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

    now = int(timezone.now().timestamp())
    mock_cancelled_batch = Batch(
        id="batch-mock-123",
        status="cancelled",
        cancelling_at=now,
        cancelled_at=now,
        input_file_id="file-mock-123",
        completed_at=None,
        failed_at=None,
        expired_at=None,
        expires_at=None,
        finalizing_at=None,
        in_progress_at=None,
        completion_window="24h",
        created_at=now,
        endpoint="/v1/chat/completions",
        error_file_id=None,
        errors=None,
        metadata=None,
        model=None,
        object="batch",
        output_file_id=None,
        request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
        usage=None,
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
        error_message = body.get("error", {}).get("message", "")
        self.assertTrue(
            "Invalid JSON" in error_message, f"Unexpected error message: {error_message}"
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
        """Personal token sees user-scoped batches; SA token sees team-scoped batches."""
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
        # Create batch under token1 (personal token, self.headers)
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

        # Build headers for a second token (from fixture pk=2, SA token for TUna)
        token2 = Token.objects.get(pk=2)
        # Generate a fresh secret for token2 (update key_hash/preview) and authenticate with raw key
        secret2 = token2._set_new_key()
        token2.save(update_fields=["key_hash", "key_preview"])
        headers2 = _build_chat_headers(secret2)
        headers2.pop("Content-Type", None)

        # Create batch under token2 (SA token)
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

        # Personal token (token1) sees all batches from user=1 (both tokens have user=1)
        resp = self.client.get("/batches", headers=self.headers)
        ids1 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids1, [b1, b2])

        # SA token (token2) only sees team-scoped batches (only b2, not the personal b1)
        resp = self.client.get("/batches", headers=headers2)
        ids2 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids2, [b2])

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_max_user_batches_limit(
        self, mock_get_files_client_files, mock_get_files_client_batches
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


class TestBatchesServiceAccountAPI(GatewayBatchesTestCase):
    """Tests for service account team-scoped access control on the batches API."""

    url_chat = reverse("gateway:v1_chat_completions")

    def _make_jsonl_content(self):
        """Create valid JSONL content for batch upload."""
        payload = _build_chat_payload(
            self.model,
            messages=[{"role": "system", "content": "Hi"}, {"role": "user", "content": "Test"}],
        )
        wrapped = {"custom_id": "1", "method": "POST", "url": self.url_chat, "body": payload}
        return json.dumps(wrapped).encode() + b"\n"

    def _setup_two_sa_tokens_same_team(self):
        """
        Set up two service account tokens on the same team (Whale).

        Returns (sa_token1_headers, sa_token2_headers, sa1, sa2, token1, token2).
        Token pk=2 is already SA for TUna on team Whale (from fixture).
        We create a second SA on the same team with a new token.
        """
        team = Team.objects.get(name="Whale")

        # Token pk=2 is already a SA token for TUna on team Whale
        token1 = Token.objects.get(pk=2)
        sa1 = token1.service_account
        secret1 = token1._set_new_key()
        token1.save(update_fields=["key_hash", "key_preview"])
        headers1 = _build_chat_headers(secret1)
        headers1.pop("Content-Type", None)

        # Create a second service account on the same team
        sa2 = ServiceAccount.objects.create(name="SecondSA", team=team)
        user = token1.user  # same user for simplicity
        token2 = Token(name="Token for SecondSA", user=user, service_account=sa2)
        secret2 = token2._set_new_key()
        token2.save()
        headers2 = _build_chat_headers(secret2)
        headers2.pop("Content-Type", None)

        return headers1, headers2, sa1, sa2, token1, token2

    def _setup_cross_team_sa_token(self):
        """
        Set up a service account token on a different team.

        Returns (other_team_headers, other_sa, other_token).
        """
        from django.contrib.auth import get_user_model

        User = get_user_model()

        org = Org.objects.get(name="E060")
        other_team = Team.objects.create(name="Dolphin", org=org)

        # Create a new user for the other team
        other_user = User.objects.create_user(username="OtherTeamUser", email="other@team.com")
        UserProfile.objects.create(user=other_user, org=org)

        other_sa = ServiceAccount.objects.create(name="OtherTeamSA", team=other_team)
        other_token = Token(name="Other team token", user=other_user, service_account=other_sa)
        other_secret = other_token._set_new_key()
        other_token.save()
        other_headers = _build_chat_headers(other_secret)
        other_headers.pop("Content-Type", None)

        return other_headers, other_sa, other_token

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_create_batch_sa_cross_token_file(self, mock_files_client, mock_batches_client):
        """SA token can create a batch using a file uploaded by another SA token on the same team."""
        mock_files_client.return_value = create_mock_batch_client()
        mock_batches_client.return_value = create_mock_batch_client()

        headers1, headers2, sa1, sa2, token1, token2 = self._setup_two_sa_tokens_same_team()
        content = self._make_jsonl_content()

        # Upload file with SA token 1
        f = SimpleUploadedFile("sa1.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=headers1)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Create batch with SA token 2 using SA token 1's file
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers2,
            content_type="application/json",
        )
        self.assertEqual(
            resp.status_code, 200, f"Expected 200, got {resp.status_code}: {resp.json()}"
        )

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_create_batch_sa_cross_team_denied(self, mock_files_client, mock_batches_client):
        """SA token cannot create a batch using a file from a different team."""
        mock_files_client.return_value = create_mock_batch_client()
        mock_batches_client.return_value = create_mock_batch_client()

        headers1, _, sa1, sa2, token1, token2 = self._setup_two_sa_tokens_same_team()
        other_headers, other_sa, other_token = self._setup_cross_team_sa_token()
        content = self._make_jsonl_content()

        # Upload file with SA token on team Whale
        f = SimpleUploadedFile("whale.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=headers1)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Try to create batch with SA token on team Dolphin using Whale's file
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=other_headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["message"], "Input file not found.")

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_list_batches_sa_team_scope(self, mock_files_client, mock_batches_client):
        """SA tokens on the same team can see each other's batches."""
        mock_files_client.return_value = create_mock_batch_client()
        mock_batches_client.return_value = create_mock_batch_client()

        headers1, headers2, sa1, sa2, token1, token2 = self._setup_two_sa_tokens_same_team()
        content = self._make_jsonl_content()

        # Upload file and create batch with SA token 1
        f1 = SimpleUploadedFile("sa1.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f1, "purpose": "batch"}, headers=headers1)
        file_id1 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id1, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers1,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_id1 = resp.json()["id"]

        # Upload file and create batch with SA token 2
        f2 = SimpleUploadedFile("sa2.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f2, "purpose": "batch"}, headers=headers2)
        file_id2 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id2, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers2,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_id2 = resp.json()["id"]

        # SA token 1 should see both batches
        resp = self.client.get("/batches", headers=headers1)
        self.assertEqual(resp.status_code, 200)
        ids = [b["id"] for b in resp.json()["data"]]
        self.assertCountEqual(ids, [batch_id1, batch_id2])

        # SA token 2 should also see both batches
        resp = self.client.get("/batches", headers=headers2)
        self.assertEqual(resp.status_code, 200)
        ids = [b["id"] for b in resp.json()["data"]]
        self.assertCountEqual(ids, [batch_id1, batch_id2])

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_retrieve_batch_sa_team_scope(self, mock_files_client, mock_batches_client):
        """SA token can retrieve a batch created by another SA token on the same team."""
        mock_files_client.return_value = create_mock_batch_client()
        mock_batches_client.return_value = create_mock_batch_client()

        headers1, headers2, sa1, sa2, token1, token2 = self._setup_two_sa_tokens_same_team()
        content = self._make_jsonl_content()

        # Upload file and create batch with SA token 1
        f = SimpleUploadedFile("sa1.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=headers1)
        file_id = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers1,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # SA token 2 should be able to retrieve it
        resp = self.client.get(f"/batches/{batch_id}", headers=headers2)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], batch_id)

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_cancel_batch_sa_team_scope(self, mock_files_client, mock_batches_client):
        """SA token can cancel a batch created by another SA token on the same team."""
        mock_client = create_mock_batch_client()
        mock_files_client.return_value = mock_client
        mock_batches_client.return_value = mock_client

        headers1, headers2, sa1, sa2, token1, token2 = self._setup_two_sa_tokens_same_team()
        content = self._make_jsonl_content()

        # Upload file and create batch with SA token 1
        f = SimpleUploadedFile("sa1.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=headers1)
        file_id = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=headers1,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # SA token 2 should be able to cancel it
        resp = self.client.post(f"/batches/{batch_id}/cancel", headers=headers2)
        self.assertEqual(resp.status_code, 200)

    @patch("gateway.views.batches.get_files_api_client")
    @patch("gateway.views.files.get_files_api_client")
    def test_personal_token_cannot_see_sa_team_batches(
        self, mock_files_client, mock_batches_client
    ):
        """A personal token cannot see batches created by SA tokens, even if same user."""
        mock_files_client.return_value = create_mock_batch_client()
        mock_batches_client.return_value = create_mock_batch_client()

        # Token pk=2 is a SA token (TUna, team Whale, user=1)
        sa_token = Token.objects.get(pk=2)
        sa_secret = sa_token._set_new_key()
        sa_token.save(update_fields=["key_hash", "key_preview"])
        sa_headers = _build_chat_headers(sa_secret)
        sa_headers.pop("Content-Type", None)
        content = self._make_jsonl_content()

        # Upload file and create batch with SA token
        f = SimpleUploadedFile("sa.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": f, "purpose": "batch"}, headers=sa_headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps(
                {"input_file_id": file_id, "completion_window": "24h", "endpoint": self.url_chat}
            ),
            headers=sa_headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        sa_batch_id = resp.json()["id"]

        # Create a new personal token for a different user so it can't see SA batches
        from django.contrib.auth import get_user_model

        User = get_user_model()
        other_user = User.objects.create_user(username="PersonalUser", email="personal@test.com")
        from django.contrib.auth.models import Group

        other_user.groups.add(Group.objects.get(name="user"))
        org = Org.objects.get(name="E060")
        UserProfile.objects.create(user=other_user, org=org)
        personal_token = Token(name="PersonalToken", user=other_user)
        personal_secret = personal_token._set_new_key()
        personal_token.save()
        personal_headers = _build_chat_headers(personal_secret)
        personal_headers.pop("Content-Type", None)

        # Personal token should not see the SA batch
        resp = self.client.get("/batches", headers=personal_headers)
        self.assertEqual(resp.status_code, 200)
        ids = [b["id"] for b in resp.json()["data"]]
        self.assertNotIn(sa_batch_id, ids)

        # Personal token should not be able to retrieve the SA batch
        resp = self.client.get(f"/batches/{sa_batch_id}", headers=personal_headers)
        self.assertEqual(resp.status_code, 404)

        # Personal token should not be able to cancel the SA batch
        resp = self.client.post(f"/batches/{sa_batch_id}/cancel", headers=personal_headers)
        self.assertEqual(resp.status_code, 404)
