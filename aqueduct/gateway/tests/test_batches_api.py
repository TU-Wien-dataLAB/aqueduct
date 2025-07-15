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

    async def acompletion(self, **params):
        await asyncio.sleep(0.1)
        return DummyRouter.DummyResult()


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
