import asyncio
import json
from django.core.files.uploadedfile import SimpleUploadedFile

from gateway.tests.utils.base import GatewayBatchesTestCase
from gateway.tests.utils import _build_chat_headers, _build_chat_payload
from unittest.mock import patch


class DummyRouter:
    class DummyResult:
        def model_dump(self, **kwargs):
            return {"choices": [{"message": {"content": "dummy"}}]}

    class DummyEmbeddingResult:
        def model_dump(self, **kwargs):
            return {"data": [{"embedding": [1.0, 2.0]}]}

    async def acompletion(self, **params):
        return DummyRouter.DummyResult()

    async def atext_completion(self, **params):
        return DummyRouter.DummyResult()

    async def aembedding(self, **params):
        return DummyRouter.DummyEmbeddingResult()


class TestBatchesAPI(GatewayBatchesTestCase):
    def setUp(self):
        super().setUp()
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers.pop("Content-Type", None)
        self.headers = headers

    def test_batch_lifecycle(self):
        """Test the full batch lifecycle: upload, create, process, and retrieve outputs."""
        # Prepare a JSONL file with two chat completion requests.
        messages1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "World!"},
        ]
        payload1 = _build_chat_payload(self.model, messages1)
        payload2 = _build_chat_payload(self.model, messages2)
        content = (
                json.dumps(payload1).encode("utf-8") + b"\n" +
                json.dumps(payload2).encode("utf-8") + b"\n"
        )
        upload_file = SimpleUploadedFile("batch.jsonl", content, content_type="application/jsonl")
        response = self.client.post(
            "/files",
            {"file": upload_file, "purpose": "batch"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, f"File upload failed: {response.json()}")
        file_id = response.json()["id"]

        # Create a batch using the uploaded file.
        batch_payload = {
            "input_file_id": file_id,
            # Allowed batch completion_window literal
            "completion_window": "24h",
            # Matches OpenAI-compatible endpoint literals
            "endpoint": "/v1/chat/completions",
        }
        response = self.client.post(
            "/batches",
            data=json.dumps(batch_payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        batch_data = response.json()
        batch_id = batch_data["id"]
        self.assertEqual(batch_data["status"], "validating")

        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        # Retrieve the completed batch metadata.
        response = self.client.get(f"/batches/{batch_id}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        completed = response.json()
        self.assertEqual(completed["status"], "completed")
        counts = completed["request_counts"]
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["completed"], 2)

        # Download and inspect the output file content.
        output_file_id = completed.get("output_file_id")
        self.assertIsNotNone(output_file_id, "Missing output_file_id in batch.")
        response = self.client.get(f"/files/{output_file_id}/content", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        lines = response.content.splitlines()
        self.assertEqual(len(lines), 2)
        for raw in lines:
            result = json.loads(raw)
            self.assertIn("choices", result)
            self.assertIsInstance(result["choices"], list)
            self.assertGreater(len(result["choices"]), 0)

    def test_invalid_json(self):
        """POST /batches with invalid JSON returns 400."""
        # Send malformed JSON in the body
        resp = self.client.post(
            "/batches",
            data="{not: 'json'}",
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("error", body)
        # Should indicate invalid JSON
        self.assertTrue(
            "Invalid JSON" in body.get("error", ""),
            f"Unexpected error message: {body.get('error')}"
        )

    def test_nonexistent_input_file(self):
        """POST /batches with a non-existent file ID returns 404."""
        payload = {
            "input_file_id": "does_not_exist",
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
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
        self.assertEqual(err.get("error"), "Input file not found.")

    def test_list_empty_batches(self):
        """GET /batches when there are no batches should return empty list."""
        resp = self.client.get("/batches", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("data", data)
        self.assertEqual(data.get("data"), [])

    def test_list_batches_different_tokens(self):
        """Batches created under one token should not be visible under another."""
        # Prepare a simple JSONL file
        payload = _build_chat_payload(self.model, messages=[
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Token1"},
        ])
        content = json.dumps(payload).encode() + b"\n"
        f1 = SimpleUploadedFile("t1.jsonl", content, content_type="application/jsonl")
        # Create batch under token1 (self.headers)
        resp = self.client.post(
            "/files", {"file": f1, "purpose": "batch"}, headers=self.headers
        )
        fid1 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid1,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }),
            headers=self.headers,
            content_type="application/json",
        )
        b1 = resp.json()["id"]

        # Build headers for a second token (from fixture pk=2)
        from management.models import Token
        token2 = Token.objects.get(pk=2)
        # Generate a fresh secret for token2 (update key_hash/preview) and authenticate with raw key
        secret2 = token2._set_new_key()
        token2.save(update_fields=['key_hash', 'key_preview'])
        headers2 = _build_chat_headers(secret2)
        headers2.pop("Content-Type", None)

        # Create batch under token2
        f2 = SimpleUploadedFile("t2.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(
            "/files", {"file": f2, "purpose": "batch"}, headers=headers2
        )
        fid2 = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid2,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }),
            headers=headers2,
            content_type="application/json",
        )
        b2 = resp.json()["id"]

        # Token1 should see only its own batch
        resp = self.client.get("/batches", headers=self.headers)
        ids1 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids1, [b1])

        # Token2 should see only its batch2
        resp = self.client.get("/batches", headers=headers2)
        ids2 = [b["id"] for b in resp.json().get("data", [])]
        self.assertCountEqual(ids2, [b2])

    def test_various_endpoints(self):
        """Test batch creation and processing for chat, simple completions, and embeddings endpoints."""
        tests = [
            ("/v1/chat/completions", json.dumps(_build_chat_payload(self.model, [
                {"role": "system", "content": "X"}, {"role": "user", "content": "Y"}
            ]))),
            ("/v1/completions", json.dumps({"model": self.model, "prompt": "Hello", "max_tokens": 2})),
            ("/v1/embeddings", json.dumps({"model": self.model, "input": ["Hello"]})),
        ]
        for endpoint, line in tests:
            content = (line + "\n").encode()
            upload = SimpleUploadedFile("e.jsonl", content, content_type="application/jsonl")
            resp = self.client.post("/files", {"file": upload, "purpose": "batch"}, headers=self.headers)
            fid = resp.json()["id"]
            resp = self.client.post(
                "/batches",
                data=json.dumps({
                    "input_file_id": fid,
                    "completion_window": "24h",
                    "endpoint": endpoint,
                }), headers=self.headers, content_type="application/json",
            )
            bid = resp.json()["id"]

            with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
                self.run_batch_processing_loop()

            info = self.client.get(f"/batches/{bid}", headers=self.headers).json()
            out_id = info.get("output_file_id")
            self.assertIsNotNone(out_id)
            lines = self.client.get(f"/files/{out_id}/content", headers=self.headers).content.splitlines()
            self.assertEqual(len(lines), 1)
            result = json.loads(lines[0])
            if endpoint.endswith("/embeddings"):
                self.assertIn("data", result)
            else:
                self.assertIn("choices", result)

    def test_partial_failure_in_batch(self):
        """If one request in a batch JSONL is invalid JSON, it should count as a failed request."""
        # Build JSONL: one valid chat, one invalid line, one valid chat
        chat_line = json.dumps(_build_chat_payload(self.model, [
            {"role": "system", "content": "X"}, {"role": "user", "content": "Ok"}
        ])).encode()
        invalid_line = b"{not valid json}\n"
        content = chat_line + b"\n" + invalid_line + chat_line + b"\n"
        upload = SimpleUploadedFile("mixed.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": upload, "purpose": "batch"}, headers=self.headers)
        fid = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }), headers=self.headers, content_type="application/json",
        )
        bid = resp.json()["id"]
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        info = self.client.get(f"/batches/{bid}", headers=self.headers).json()
        counts = info.get("request_counts", {})
        self.assertEqual(counts.get("total"), 3)
        self.assertEqual(counts.get("completed"), 2)
        self.assertEqual(counts.get("failed"), 1)

    def test_streaming_failure_in_batch(self):
        """If one request in a batch JSONL contains a streaming call, it should count as a failed request."""
        # Build JSONL: one valid chat, one invalid line, one valid chat
        chat_line = json.dumps(_build_chat_payload(self.model, [
            {"role": "system", "content": "X"}, {"role": "user", "content": "Ok"}
        ])).encode()
        invalid_line = json.dumps(_build_chat_payload(self.model, [
            {"role": "system", "content": "X"}, {"role": "user", "content": "Ok"}
        ], stream=True)).encode()
        content = chat_line + b"\n" + invalid_line + b"\n" + chat_line + b"\n"
        upload = SimpleUploadedFile("mixed.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": upload, "purpose": "batch"}, headers=self.headers)
        fid = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }), headers=self.headers, content_type="application/json",
        )
        bid = resp.json()["id"]
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        info = self.client.get(f"/batches/{bid}", headers=self.headers).json()
        counts = info.get("request_counts", {})
        self.assertEqual(counts.get("total"), 3)
        self.assertEqual(counts.get("completed"), 2)
        self.assertEqual(counts.get("failed"), 1)

    def test_get_nonexistent_batch(self):
        """GET /batches/{id} for a non-existent batch returns 404."""
        resp = self.client.get("/batches/nonexistent", headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        self.assertEqual(body.get("error"), "Batch not found.")

    def test_cancel_nonexistent_batch(self):
        """POST /batches/{id}/cancel for non-existent batch returns 404."""
        resp = self.client.post("/batches/nonexistent/cancel", headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        self.assertEqual(body.get("error"), "Batch not found.")

    def test_queue_parallelism(self):
        """Ensure more requests than concurrency limit still all get processed."""
        # Create JSONL with 3 chat requests
        msgs = [
            [{"role": "system", "content": "A"}, {"role": "user", "content": str(i)}]
            for i in range(10)
        ]
        lines = [json.dumps(_build_chat_payload(self.model, m)).encode() for m in msgs]
        content = b"\n".join(lines) + b"\n"
        upload = SimpleUploadedFile("q.jsonl", content, content_type="application/jsonl")
        resp = self.client.post("/files", {"file": upload, "purpose": "batch"}, headers=self.headers)

        file_id = resp.json()["id"]
        batch_payload = {"input_file_id": file_id, "completion_window": "24h", "endpoint": "/v1/chat/completions"}
        resp = self.client.post("/batches", data=json.dumps(batch_payload), headers=self.headers,
                                content_type="application/json")
        batch_id = resp.json()["id"]

        # Process all 10 requests via DummyRouter
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        resp = self.client.get(f"/batches/{batch_id}", headers=self.headers)
        counts = resp.json()["request_counts"]
        self.assertEqual(counts["total"], 10)

    def test_multiple_batches(self):
        """Ensure multiple batches can be created and processed independently."""

        # Setup two input files with different numbers of requests
        def make_batch(n):
            msgs = [[{"role": "system", "content": "X"}, {"role": "user", "content": str(i)}] for i in range(n)]
            content = b"".join(json.dumps(_build_chat_payload(self.model, m)).encode() + b"\n" for m in msgs)
            upload = SimpleUploadedFile(f"b{n}.jsonl", content, content_type="application/jsonl")
            resp = self.client.post("/files", {"file": upload, "purpose": "batch"}, headers=self.headers)
            fid = resp.json()["id"]
            resp = self.client.post("/batches", data=json.dumps({
                "input_file_id": fid, "completion_window": "24h", "endpoint": "/v1/chat/completions"
            }), headers=self.headers, content_type="application/json")
            return resp.json()["id"], n

        batch1, n1 = make_batch(2)
        batch2, n2 = make_batch(3)
        # Process both batches in same run
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        # Check counts
        for b_id, n in ((batch1, n1), (batch2, n2)):
            resp = self.client.get(f"/batches/{b_id}", headers=self.headers)
            counts = resp.json()["request_counts"]
            self.assertEqual(counts["total"], n)

    def test_cancel_before_processing(self):
        """Test cancelling a batch before processing (status validating)."""
        # Upload a single-line file
        payload = _build_chat_payload(self.model, messages=[
            {"role": "system", "content": "X"}, {"role": "user", "content": "Y"}
        ])
        content = json.dumps(payload).encode() + b"\n"
        upload = SimpleUploadedFile("c.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(
            "/files", {"file": upload, "purpose": "batch"}, headers=self.headers
        )
        fid = resp.json()["id"]
        # Create batch and cancel immediately
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }),
            headers=self.headers,
            content_type="application/json",
        )
        b_id = resp.json()["id"]
        resp = self.client.post(f"/batches/{b_id}/cancel", headers=self.headers)
        data = resp.json()
        self.assertEqual(data["status"], "cancelled")

    def test_cancel_in_progress(self):
        """Test cancelling a batch in in_progress state transitions through cancelling to cancelled."""
        # Upload and create a single-line batch
        payload = _build_chat_payload(self.model, messages=[
            {"role": "system", "content": "X"}, {"role": "user", "content": "Z"}
        ])
        content = json.dumps(payload).encode() + b"\n"
        upload = SimpleUploadedFile("c2.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(
            "/files", {"file": upload, "purpose": "batch"}, headers=self.headers
        )
        fid = resp.json()["id"]
        resp = self.client.post(
            "/batches",
            data=json.dumps({
                "input_file_id": fid,
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
            }),
            headers=self.headers,
            content_type="application/json",
        )
        b2 = resp.json()["id"]

        # Force status to in_progress directly (loop consumes lines too quickly in tests)
        from management.models import Batch
        Batch.objects.filter(id=b2).update(status="in_progress")

        # Cancel while in_progress
        resp = self.client.post(f"/batches/{b2}/cancel", headers=self.headers)
        data2 = resp.json()
        self.assertEqual(data2["status"], "cancelling")

        # Finalize cancellation via processing loop
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()
        resp = self.client.get(f"/batches/{b2}", headers=self.headers)
        self.assertEqual(resp.json()["status"], "cancelled")

    def test_expired_batches_marked_expired(self):
        """Expired batches are marked as expired by the processing loop."""
        from django.utils import timezone
        from management.models import Batch

        # Upload a file and create a batch
        payload = _build_chat_payload(self.model, messages=[
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Expire"},
        ])
        content = json.dumps(payload).encode("utf-8") + b"\n"
        upload_file = SimpleUploadedFile(
            "expire.jsonl", content, content_type="application/jsonl"
        )
        resp = self.client.post(
            "/files", {"file": upload_file, "purpose": "batch"}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Create the batch
        batch_payload = {
            "input_file_id": file_id,
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
        }
        resp = self.client.post(
            "/batches",
            data=json.dumps(batch_payload),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_data = resp.json()
        batch_id = batch_data["id"]
        created_at = batch_data["created_at"]

        # Simulate expiration by setting expires_at to the creation time - 1
        Batch.objects.filter(id=batch_id).update(expires_at=created_at-1, created_at=created_at-1)

        # Run processing loop to trigger expiry logic
        with patch('gateway.views.batches.get_router', return_value=DummyRouter()):
            self.run_batch_processing_loop()

        # Verify batch marked as expired
        resp = self.client.get(f"/batches/{batch_id}", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "expired")
        self.assertIn("expired_at", data)
        self.assertTrue(data["expired_at"] >= created_at)
