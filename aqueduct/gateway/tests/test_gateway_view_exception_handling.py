import json

from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from gateway.tests.utils.base import GatewayBatchesTestCase, GatewayFilesTestCase, GatewayIntegrationTestCase
from mock_api.mock_configs import MockConfig


class TestCatchRouterExceptionsIntegration(GatewayBatchesTestCase):
    def test_batches_post_bad_request_error(self):
        """Test batches POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid endpoint",
                    "type": "invalid_request_error",
                    "param": "endpoint",
                    "code": "invalid_value",
                }
            },
        )

        # Create and upload a test file
        content = b'{"custom_id": "test"}\n'
        f = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Patch upstream batches endpoint to return 400
        with self.mock_server.patch_external_api("batches", bad_request):
            # Attempt to create batch
            resp = self.client.post(
                reverse("gateway:batches"),
                data=json.dumps(
                    {"input_file_id": file_id, "completion_window": "24h", "endpoint": "/v1/chat/completions"}
                ),
                content_type="application/json",
                headers=self.headers,
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid endpoint", resp.json()["error"]["message"])

    def test_batch_get_bad_request_error(self):
        """Test batch GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid batch ID",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create and upload a test file
        content = b'{"custom_id": "test"}\n'
        f = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Create a batch successfully first
        resp = self.client.post(
            reverse("gateway:batches"),
            data=json.dumps({"input_file_id": file_id, "completion_window": "24h", "endpoint": "/v1/chat/completions"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Patch upstream batch retrieval to return 400
        with self.mock_server.patch_external_api(f"batches/{batch_id}", bad_request):
            resp = self.client.get(reverse("gateway:batch", args=[batch_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid batch ID", resp.json()["error"]["message"])

    def test_batch_cancel_bad_request_error(self):
        """Test batch.cancel POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Cannot cancel batch",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create and upload a test file
        content = b'{"custom_id": "test"}\n'
        f = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "batch"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Create a batch successfully first
        resp = self.client.post(
            reverse("gateway:batches"),
            data=json.dumps({"input_file_id": file_id, "completion_window": "24h", "endpoint": "/v1/chat/completions"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Patch upstream batch cancel to return 400
        with self.mock_server.patch_external_api(f"batches/{batch_id}/cancel", bad_request):
            resp = self.client.post(reverse("gateway:batch_cancel", args=[batch_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot cancel batch", resp.json()["error"]["message"])


class TestCatchRouterExceptionsFiles(GatewayFilesTestCase):
    def test_file_post_bad_request_error(self):
        """Test files POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file purpose",
                    "type": "invalid_request_error",
                    "param": "purpose",
                    "code": "invalid_value",
                }
            },
        )

        # Patch upstream files endpoint to return 400
        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        with self.mock_server.patch_external_api("files", bad_request):
            resp = self.client.post(
                reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file purpose", resp.json()["error"]["message"])

    def test_file_get_bad_request_error(self):
        """Test file GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file ID",
                    "type": "invalid_request_error",
                    "param": "file_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a file successfully first
        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Patch upstream file retrieval to return 400
        with self.mock_server.patch_external_api(f"files/{file_id}", bad_request):
            resp = self.client.get(reverse("gateway:file", args=[file_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file ID", resp.json()["error"]["message"])

    def test_file_content_get_bad_request_error(self):
        """Test file.content GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Cannot retrieve file content",
                    "type": "invalid_request_error",
                    "param": "file_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a file successfully first
        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Patch upstream file content retrieval to return 400
        with self.mock_server.patch_external_api(f"files/{file_id}/content", bad_request):
            resp = self.client.get(reverse("gateway:file_content", args=[file_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot retrieve file content", resp.json()["error"]["message"])


class TestCatchRouterExceptionsVectorStores(GatewayIntegrationTestCase):
    def test_vector_stores_post_bad_request_error(self):
        """Test vector_stores POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid vector store name",
                    "type": "invalid_request_error",
                    "param": "name",
                    "code": "invalid_value",
                }
            },
        )

        # Patch upstream vector_stores endpoint to return 400
        with self.mock_server.patch_external_api("vector_stores", bad_request):
            resp = self.client.post(
                reverse("gateway:vector_stores"),
                data=json.dumps({"name": "test"}),
                content_type="application/json",
                headers=self.headers,
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid vector store name", resp.json()["error"]["message"])

    def test_vector_store_get_bad_request_error(self):
        """Test vector_store GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid vector store",
                    "type": "invalid_request_error",
                    "param": "vector_store_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store successfully first
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        # Patch upstream vector store retrieval to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}", bad_request):
            resp = self.client.get(reverse("gateway:vector_store", args=[vs_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid vector store", resp.json()["error"]["message"])

    def test_vector_store_search_post_bad_request_error(self):
        """Test vector_store.search POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid query",
                    "type": "invalid_request_error",
                    "param": "query",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store successfully first
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        # Patch upstream vector store search to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/search", bad_request):
            resp = self.client.post(
                reverse("gateway:vector_store_search", args=[vs_id]),
                data=json.dumps({"query": "test query"}),
                content_type="application/json",
                headers=self.headers,
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid query", resp.json()["error"]["message"])


class TestCatchRouterExceptionsVectorStoreFiles(GatewayFilesTestCase):
    def test_vector_store_files_post_bad_request_error(self):
        """Test vector_stores.files POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file for vector store",
                    "type": "invalid_request_error",
                    "param": "file_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store and file first
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Patch upstream vector_stores.files endpoint to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/files", bad_request):
            resp = self.client.post(
                reverse("gateway:vector_store_files", args=[vs_id]),
                data=json.dumps({"file_id": file_id}),
                content_type="application/json",
                headers=self.headers,
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file for vector store", resp.json()["error"]["message"])

    def test_vector_store_file_get_bad_request_error(self):
        """Test vector_stores.files GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid vector store file",
                    "type": "invalid_request_error",
                    "param": "file_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store and file, add to vector store
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        resp = self.client.post(
            reverse("gateway:vector_store_files", args=[vs_id]),
            data=json.dumps({"file_id": file_id}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_file_id = resp.json()["id"]

        # Patch upstream vector store file retrieval to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/files/{vs_file_id}", bad_request):
            resp = self.client.get(reverse("gateway:vector_store_file", args=[vs_id, vs_file_id]), headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid vector store file", resp.json()["error"]["message"])

    def test_vector_store_file_content_get_bad_request_error(self):
        """Test vector_stores.files.content GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Cannot retrieve vector store file content",
                    "type": "invalid_request_error",
                    "param": "file_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store and file, add to vector store
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        resp = self.client.post(
            reverse("gateway:vector_store_files", args=[vs_id]),
            data=json.dumps({"file_id": file_id}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_file_id = resp.json()["id"]

        # Patch upstream vector store file content retrieval to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/files/{vs_file_id}/content", bad_request):
            resp = self.client.get(
                reverse("gateway:vector_store_file_content", args=[vs_id, vs_file_id]), headers=self.headers
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot retrieve vector store file content", resp.json()["error"]["message"])


class TestCatchRouterExceptionsVectorStoreFileBatches(GatewayFilesTestCase):
    def test_vector_store_file_batches_post_bad_request_error(self):
        """Test file_batches POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file batch",
                    "type": "invalid_request_error",
                    "param": "file_ids",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store and file
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        # Patch upstream file_batches endpoint to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/file_batches", bad_request):
            resp = self.client.post(
                reverse("gateway:vector_store_file_batches", args=[vs_id]),
                data=json.dumps({"file_ids": [file_id]}),
                content_type="application/json",
                headers=self.headers,
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file batch", resp.json()["error"]["message"])

    def test_vector_store_file_batch_get_bad_request_error(self):
        """Test file_batches GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file batch ID",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store, file, and batch
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        resp = self.client.post(
            reverse("gateway:vector_store_file_batches", args=[vs_id]),
            data=json.dumps({"file_ids": [file_id]}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Patch upstream file batch retrieval to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/file_batches/{batch_id}", bad_request):
            resp = self.client.get(
                reverse("gateway:vector_store_file_batch", args=[vs_id, batch_id]), headers=self.headers
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file batch ID", resp.json()["error"]["message"])

    def test_vector_store_file_batch_cancel_post_bad_request_error(self):
        """Test file_batches.cancel POST view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Cannot cancel file batch",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store, file, and batch
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        resp = self.client.post(
            reverse("gateway:vector_store_file_batches", args=[vs_id]),
            data=json.dumps({"file_ids": [file_id]}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Patch upstream file batch cancel to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/file_batches/{batch_id}/cancel", bad_request):
            resp = self.client.post(
                reverse("gateway:vector_store_file_batch_cancel", args=[vs_id, batch_id]), headers=self.headers
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot cancel file batch", resp.json()["error"]["message"])

    def test_vector_store_file_batch_files_get_bad_request_error(self):
        """Test file_batches.list_files GET view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Cannot list files in batch",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )

        # Create a vector store, file, and batch
        resp = self.client.post(
            reverse("gateway:vector_stores"),
            data=json.dumps({"name": "test"}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        vs_id = resp.json()["id"]

        content = b"test content"
        f = SimpleUploadedFile("test.txt", content, content_type="text/plain")
        resp = self.client.post(reverse("gateway:files"), {"file": f, "purpose": "assistants"}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["id"]

        resp = self.client.post(
            reverse("gateway:vector_store_file_batches", args=[vs_id]),
            data=json.dumps({"file_ids": [file_id]}),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Patch upstream file batch files listing to return 400
        with self.mock_server.patch_external_api(f"vector_stores/{vs_id}/file_batches/{batch_id}/files", bad_request):
            resp = self.client.get(
                reverse("gateway:vector_store_file_batch_files", args=[vs_id, batch_id]), headers=self.headers
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot list files in batch", resp.json()["error"]["message"])
