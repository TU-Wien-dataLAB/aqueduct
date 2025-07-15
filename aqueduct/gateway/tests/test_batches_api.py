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
