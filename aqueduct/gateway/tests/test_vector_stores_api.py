import json
from http import HTTPStatus

from django.conf import settings
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from openai.pagination import AsyncCursorPage, AsyncPage
from openai.types.vector_stores import FileContentResponse, VectorStoreFile, VectorStoreFileBatch
from openai.types.vector_stores.vector_store_file_batch import FileCounts

from gateway.tests.utils.base import GatewayFilesTestCase
from management.models import (
    FileObject,
    Token,
    VectorStoreFileBatchStatus,
    VectorStoreFileStatus,
    VectorStoreStatus,
)
from management.models import VectorStore as VectorStoreModel
from management.models import VectorStoreFile as VectorStoreFileModel
from management.models import VectorStoreFileBatch as VectorStoreFileBatchModel
from mock_api.mock_configs import MockConfig


@override_settings(MAX_USER_VECTOR_STORES=3, MAX_TEAM_VECTOR_STORES=10, MAX_VECTOR_STORE_FILES=100)
class TestVectorStoresBase(GatewayFilesTestCase):
    """Base class with shared helpers for vector store tests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.token = Token.objects.first()

    def setUp(self):
        super().setUp()
        self.vs_obj = self._create_vector_store()
        self.vs_id = self.vs_obj.id
        self.vs_files_url = reverse(
            "gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id}
        )

    def _create_vector_store(
        self, vs_id: str | None = "vs-mock-123", name: str | None = "Test Store"
    ) -> VectorStoreModel:
        """Helper to create a VectorStore in the database."""
        return VectorStoreModel.objects.create(
            id=vs_id,
            token=self.token,
            name=name,
            status=VectorStoreStatus.COMPLETED,
            usage_bytes=0,
            created_at=int(timezone.now().timestamp()),
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )

    def _create_file_object(self, file_id: str | None = "file-mock-123") -> FileObject:
        """Helper to create a FileObject in the database."""
        return FileObject.objects.create(
            id=file_id,
            bytes=100,
            created_at=int(timezone.now().timestamp()),
            filename="test.txt",
            purpose="user_data",
            token=self.token,
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )

    @property
    def _mock_files_list(self) -> MockConfig:
        """Prepare the files list mock to return upstream files"""
        now = int(timezone.now().timestamp())
        return MockConfig(
            response_data=AsyncCursorPage[VectorStoreFile](
                data=[
                    VectorStoreFile(
                        id="file-mock-1",
                        status="completed",
                        usage_bytes=100,
                        created_at=now,
                        last_error=None,
                        object="vector_store.file",
                        vector_store_id=self.vs_id,
                    ),
                    VectorStoreFile(
                        id="file-mock-2",
                        status="completed",
                        usage_bytes=200,
                        created_at=now,
                        last_error=None,
                        object="vector_store.file",
                        vector_store_id=self.vs_id,
                    ),
                ],
                has_more=False,
            ).model_dump()
        )

    def _create_batch(self) -> int:
        """Helper to create two files and a batch, calling the vector_store_file_batches view"""
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["id"]


class TestVectorStores(TestVectorStoresBase):
    url_vector_stores = reverse("gateway:vector_stores")

    def test_vector_store_lifecycle(self):
        """Test full lifecycle: create, list, get, modify, delete vector store."""
        VectorStoreModel.objects.all().delete()

        # Create vector store
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({"name": "Test Store"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create failed: {resp.json()}")
        data = resp.json()
        # ID is now the upstream ID
        vs_obj = VectorStoreModel.objects.first()
        vs_id = vs_obj.id
        self.assertEqual(data["id"], vs_id)

        # List vector stores
        resp = self.client.get(self.url_vector_stores, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], vs_id)

        # Get vector store
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_id})
        resp = self.client.get(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], vs_id)

        # Modify vector store
        resp = self.client.post(
            vs_url,
            data=json.dumps({"name": "Updated Name"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "Updated Name")

        # Delete vector store
        resp = self.client.delete(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["deleted"])
        self.assertEqual(data["id"], vs_id)

        # Verify deleted
        resp = self.client.get(self.url_vector_stores, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 0)

    def test_max_vector_stores_limit(self):
        """Test that MAX_USER_VECTOR_STORES limit is enforced."""

        # Create up to limit
        VectorStoreModel.objects.all().delete()
        for i in range(settings.MAX_USER_VECTOR_STORES):
            self._create_vector_store(vs_id=f"vs-mock-{i}", name=f"Store nr {i}")

        # Try to exceed limit
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({"name": "Extra Store"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("limit reached", resp.json()["error"]["message"].lower())

    def test_nonexistent_vector_store(self):
        """GET/DELETE on nonexistent vector store returns 404."""
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": "nonexistent"})

        resp = self.client.get(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 404)

        resp = self.client.delete(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 404)

        resp = self.client.post(
            vs_url,
            data=json.dumps({"name": "Updated"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_validation_errors(self):
        """Missing required fields return 400."""
        # Try to create a vector store without name
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Missing required parameter: name", resp.json()["error"]["message"])

    def test_isolation_between_tokens(self):
        """Users can't see each other's vector stores."""
        # Create new user and token
        other_token_value, _other_user_id = self.create_new_user()
        other_headers = {"Authorization": f"Bearer {other_token_value}"}

        # Try to access the vector store of one user with the different user's secret
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.get(vs_url, headers=other_headers)
        self.assertEqual(resp.status_code, 404)

        # Verify other user sees empty list
        resp = self.client.get(self.url_vector_stores, headers=other_headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 0)

        # Original user still sees their store
        resp = self.client.get(self.url_vector_stores, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 1)

    def test_upstream_failure_create(self):
        """Test vector_stores list view catches Bad Request (400) from upstream."""
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
        with self.mock_server.patch_external_api(self.url_vector_stores, bad_request):
            resp = self.client.post(
                self.url_vector_stores,
                data=json.dumps({"name": "Wrong Store"}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid vector store name", resp.json()["error"]["message"])

    def test_upstream_failure_retrieve(self):
        """Test vector_store retrieve view catches Bad Request (400) from upstream."""
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
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        with self.mock_server.patch_external_api(vs_url, bad_request):
            resp = self.client.get(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid vector store", resp.json()["error"]["message"])

    def test_upstream_failure_update(self):
        """Test vector_stores update view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Update failed upstream",
                    "type": "invalid_request_error",
                    "param": "vector_store_id",
                    "code": "invalid_value",
                }
            },
        )
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})

        with self.mock_server.patch_external_api(vs_url, bad_request):
            resp = self.client.post(
                vs_url,
                data=json.dumps({"name": "Updated Name"}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Update failed upstream", resp.json()["error"]["message"])

    def test_upstream_failure_delete(self):
        """Test vector_stores delete view catches Bad Request (400) from upstream."""
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Delete failed upstream",
                    "type": "invalid_request_error",
                    "param": "vector_store_id",
                    "code": "invalid_value",
                }
            },
        )
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        with self.mock_server.patch_external_api(vs_url, bad_request):
            resp = self.client.delete(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Delete failed upstream", resp.json()["error"]["message"])

    def test_vector_store_search(self):
        """Test searching a vector store with file_id mapping."""
        file_obj = self._create_file_object("file-mock-123")

        search_url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            search_url,
            data=json.dumps({"query": "test query", "max_num_results": 10}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Search failed: {resp.json()}")
        data = resp.json()
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["file_id"], file_obj.id)

    def test_vector_store_search_upstream_failure(self):
        """Test vector_store.search view catches Bad Request (400) from upstream."""
        search_url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
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
        with self.mock_server.patch_external_api(search_url, bad_request):
            resp = self.client.post(
                search_url,
                data=json.dumps({"query": "test query"}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid query", resp.json()["error"]["message"])

    def test_vector_store_search_missing_query(self):
        """Test search without query returns 400."""
        search_url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            search_url, data=json.dumps({}), headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("query: Field required", resp.json()["error"]["message"])


class TestVectorStoreFiles(TestVectorStoresBase):
    """Tests for vector store file endpoints (add, list, get, delete, search, content)."""

    def test_vector_store_file_lifecycle(self):
        """Test adding, listing, retrieving, and removing files from vector store."""

        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Add file failed: {resp.json()}")
        data = resp.json()
        # ID is now the upstream ID
        vsf_id = data["id"]
        vsf = VectorStoreFileModel.objects.get()
        self.assertEqual(vsf.id, vsf_id)

        # List files in vector store
        resp = self.client.get(self.vs_files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], vsf_id)

        # Get file
        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id}
        )
        resp = self.client.get(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], vsf_id)

        # Delete file
        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["deleted"])
        self.assertEqual(VectorStoreFileModel.objects.count(), 0)

    def test_list_vector_store_files_response_structure(self):
        """Test that list vector store files endpoint returns complete, correctly-mapped items."""

        with self.mock_server.patch_external_api(self.vs_files_url, self._mock_files_list):
            resp = self.client.get(self.vs_files_url, headers=self.headers)

        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 2)

        for item in data["data"]:
            # ID is now the upstream ID
            # file_id and vector_store_id are upstream IDs
            self.assertEqual(item["status"], "completed")
            self.assertEqual(item["object"], "vector_store.file")

    def test_list_files_returns_upstream_data(self):
        """Test that listing files returns upstream data directly without local syncing."""

        # Assert that there are no files in the database
        self.assertEqual(VectorStoreFileModel.objects.count(), 0)

        resp = self.client.get(self.vs_files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        # Verify file is returned in response with upstream ID
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "vsf-mock-123")
        self.assertEqual(data["data"][0]["status"], "completed")
        self.assertEqual(data["data"][0]["vector_store_id"], self.vs_id)

    def test_nonexistent_file_for_vs_file(self):
        """POST with invalid file_id returns 404."""
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": "nonexistent-file-id"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_validation_errors(self):
        """Missing required fields return 400."""
        # Try to create a VS file without file_id
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file_id: Field required", resp.json()["error"]["message"])

    @override_settings(MAX_VECTOR_STORE_FILES=3)
    def test_max_vector_store_files_limit(self):
        """Test that MAX_VECTOR_STORE_FILES limit is enforced when adding files."""

        # Create MAX_VECTOR_STORE_FILES files and vector store files in the database
        file_objs = []
        vsf_objs = []
        for i in range(settings.MAX_VECTOR_STORE_FILES):
            file_obj = self._create_file_object(f"file-mock-{i}")
            vsf_objs.append(
                VectorStoreFileModel(
                    id=f"vsf-mock-{i}",
                    vector_store=self.vs_obj,
                    file_obj=file_obj,
                    status="in_progress",
                    created_at=int(timezone.now().timestamp()),
                )
            )
            file_objs.append(file_obj)
        VectorStoreFileModel.objects.bulk_create(vsf_objs)

        # Try to add one more file - should fail with 403
        extra_file = self._create_file_object("file-mock-extra")
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": extra_file.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("limit reached", resp.json()["error"]["message"].lower())

    def test_vector_store_file_create_upstream_failure(self):
        """Test vector_stores.files create view catches Bad Request (400) from upstream."""
        # Create file object
        file_obj = self._create_file_object()

        # Try to add a vector store file - should fail with 400
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
        with self.mock_server.patch_external_api(self.vs_files_url, bad_request):
            resp = self.client.post(
                self.vs_files_url,
                data=json.dumps({"file_id": file_obj.id}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid file for vector store", resp.json()["error"]["message"])

    def test_vector_store_file_retrieve_and_delete_upstream_failure(self):
        """Test vector_stores.files retrieve view catches Bad Request (400) from upstream."""
        # Create a vector store file successfully and test retrieve failure
        file_obj = self._create_file_object()
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        vsf_id = resp.json()["id"]

        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id}
        )
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
        with self.mock_server.patch_external_api(file_url, bad_request):
            resp = self.client.get(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid vector store file", resp.json()["error"]["message"])

        # Now test delete failure
        with self.mock_server.patch_external_api(file_url, bad_request):
            resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid vector store file", resp.json()["error"]["message"])

    def test_vector_store_file_update_attributes(self):
        """Test updating file attributes."""

        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        vsf_id = resp.json()["id"]
        self.assertTrue(VectorStoreFileModel.objects.filter(id=vsf_id).exists())

        # Update file attributes
        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id}
        )
        new_attributes = {"key": "value"}
        resp = self.client.post(
            file_url,
            data=json.dumps({"attributes": new_attributes}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Update failed: {resp.json()}")
        data = resp.json()
        self.assertEqual(data["attributes"], new_attributes)

    def test_vector_store_file_update_missing_attributes(self):
        """Test updating file without attributes returns 400."""
        # Create file object
        file_obj = self._create_file_object()
        # Add file to vector store
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        vsf_id = resp.json()["id"]

        # Update without attributes
        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id}
        )
        resp = self.client.post(
            file_url, data=json.dumps({}), headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Missing required parameter: attributes", resp.json()["error"]["message"])

    def test_vector_store_file_content(self):
        """Test getting file content with both FileContentResponse and AsyncPage types."""
        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        vsf_obj = VectorStoreFileModel.objects.get()
        vsf_id = vsf_obj.id
        self.assertEqual(resp.json()["id"], vsf_id)

        content_url = reverse(
            "gateway:vector_store_file_content",
            kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id},
        )

        # Test primary case: FileContentResponse
        resp = self.client.get(content_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        json_resp = resp.json()
        self.assertIn("text", json_resp)
        self.assertIn("type", json_resp)
        self.assertEqual(json_resp["text"], "Test file content")
        self.assertEqual(json_resp["type"], "text")

        # Test secondary case: AsyncPage[FileContentResponse]
        mock_resp = MockConfig(
            response_data=AsyncPage[FileContentResponse](
                data=[FileContentResponse(text="Page content 1", type="text")],
                object="vector_store.file_content.page",
            ).model_dump()
        )
        with self.mock_server.patch_external_api(content_url, mock_resp):
            resp = self.client.get(content_url, headers=self.headers)

        self.assertEqual(resp.status_code, 200)
        json_resp = resp.json()
        self.assertIn("data", json_resp)
        self.assertIn("object", json_resp)
        self.assertEqual(json_resp["object"], "vector_store.file_content.page")
        self.assertEqual(len(json_resp["data"]), 1)
        self.assertEqual(json_resp["data"][0]["text"], "Page content 1")
        self.assertEqual(json_resp["data"][0]["type"], "text")

    def test_vector_store_file_content_upstream_failure(self):
        """Test vector_stores.files.content view catches Bad Request (400) from upstream."""
        # Add file to vector store
        file_obj = self._create_file_object()
        resp = self.client.post(
            self.vs_files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        vsf_id = resp.json()["id"]

        content_url = reverse(
            "gateway:vector_store_file_content",
            kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id},
        )
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
        with self.mock_server.patch_external_api(content_url, bad_request):
            resp = self.client.get(content_url, headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot retrieve vector store file content", resp.json()["error"]["message"])


class TestVectorStoreFileBatches(TestVectorStoresBase):
    def test_file_batch_lifecycle(self):
        """Test creating, retrieving, and canceling file batches."""

        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        # Create batch
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create batch failed: {resp.json()}")
        data = resp.json()
        # ID is now the upstream ID
        batch_id = data["id"]
        batch = VectorStoreFileBatchModel.objects.get(id=batch_id)
        self.assertEqual(batch.status, VectorStoreFileBatchStatus.IN_PROGRESS)
        self.assertEqual(batch.file_counts["total"], 2)
        self.assertEqual(batch.file_counts["in_progress"], 2)

        # Get batch
        batch_url = reverse(
            "gateway:vector_store_file_batch",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], batch_id)
        self.assertEqual(data["file_counts"]["total"], 2)
        # TODO: more assertions about the objects in the db

        # Cancel batch
        cancel_url = reverse(
            "gateway:vector_store_file_batch_cancel",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        resp = self.client.post(cancel_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "cancelled")
        batch = VectorStoreFileBatchModel.objects.get(id=batch_id)
        self.assertEqual(batch.status, VectorStoreFileBatchStatus.CANCELLED)
        self.assertEqual(batch.file_counts["cancelled"], 2)

        vs_files = VectorStoreFileModel.objects.filter(batch_id=batch_id)
        self.assertEqual(vs_files.count(), 2)
        for vsf in vs_files:
            self.assertEqual(
                vsf.status,
                VectorStoreFileStatus.CANCELLED,
                f"Wrong status for VS file {vsf.id}: {vsf.status}",
            )
            self.assertEqual(
                vsf.last_error,
                f"Batch {batch.status}: files were not processed",
                f"Wrong last_error for VS file {vsf.id}: {vsf.last_error}",
            )

    def test_file_batch_file_counts(self):
        """Batch correctly tracks completed/failed files."""
        # Create a batch with two files
        batch_id = self._create_batch()
        batch = VectorStoreFileBatchModel.objects.get(id=batch_id)
        self.assertEqual(batch.status, VectorStoreFileBatchStatus.IN_PROGRESS)
        self.assertEqual(batch.file_counts["total"], 2)
        self.assertEqual(batch.file_counts["completed"], 0)

        # Now assume the batch failed upstream; get it and verify file status is updated
        failed_resp = MockConfig(
            response_data=VectorStoreFileBatch(
                id="vsb-mock-1",
                status="failed",
                created_at=1741476542,
                file_counts=FileCounts(total=2, completed=0, failed=2, in_progress=0, cancelled=0),
                object="vector_store.files_batch",
                vector_store_id="vs-mock-123",
            ).model_dump()
        )
        batch_url = reverse(
            "gateway:vector_store_file_batch",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        with self.mock_server.patch_external_api(batch_url, failed_resp):
            resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        batch.refresh_from_db()
        file_counts = resp.json()["file_counts"]
        self.assertEqual(batch.file_counts["total"], file_counts["total"])
        self.assertEqual(batch.file_counts["failed"], file_counts["failed"])
        vs_files = VectorStoreFileModel.objects.filter(batch_id=batch_id)
        self.assertEqual(vs_files.count(), 2)
        for vsf in vs_files:
            self.assertEqual(
                vsf.status,
                VectorStoreFileStatus.FAILED,
                f"Wrong status for VS file {vsf.id}: {vsf.status}",
            )
            self.assertEqual(
                vsf.last_error,
                f"Batch {batch.status}: files were not processed",
                f"Wrong last_error for VS file {vsf.id}: {vsf.last_error}",
            )

    def test_file_batch_upstream_failure_create(self):
        """Test file_batches create view catches Bad Request (400) from upstream."""
        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id}
        )
        bad_request = MockConfig(
            status_code=400,
            response_data={
                "error": {
                    "message": "Invalid file batch",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                    "code": "invalid_value",
                }
            },
        )
        with self.mock_server.patch_external_api(batches_url, bad_request):
            resp = self.client.post(
                batches_url,
                data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid file batch", resp.json()["error"]["message"])

    def test_file_batch_upstream_failure_retrieve(self):
        """Test file_batches GET view catches Bad Request (400) from upstream."""
        batch_id = self._create_batch()
        batch_url = reverse("gateway:vector_store_file_batch", args=[self.vs_id, batch_id])
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
        with self.mock_server.patch_external_api(batch_url, bad_request):
            resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid file batch ID", resp.json()["error"]["message"])

    def test_file_batch_cancel_upstream_failure(self):
        """Test file_batches.cancel view catches Bad Request (400) from upstream."""
        batch_id = self._create_batch()
        cancel_url = reverse(
            "gateway:vector_store_file_batch_cancel",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
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
        with self.mock_server.patch_external_api(cancel_url, bad_request):
            resp = self.client.post(
                cancel_url, headers=self.headers, content_type="application/json"
            )
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Cannot cancel file batch", resp.json()["error"]["message"])

    def test_file_batch_list_files(self):
        """Test listing files in a batch."""
        # Create a batch
        batch_id = self._create_batch()

        batch_files_url = reverse(
            "gateway:vector_store_file_batch_files",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        resp = self.client.get(batch_files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200, f"List batch files failed: {resp.json()}")
        data = resp.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 2)

        for item in data["data"]:
            # ID is now the upstream ID
            # file_id and vector_store_id are upstream IDs
            self.assertEqual(item["status"], "completed")
            self.assertEqual(item["object"], "vector_store.file")

    def test_file_batch_list_files_upstream_failure(self):
        """Test file_batches.list_files view catches Bad Request (400) from upstream."""
        # Create a batch
        batch_id = self._create_batch()

        batch_files_url = reverse(
            "gateway:vector_store_file_batch_files",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )

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

        # Patch upstream file batch files listing to return 400
        with self.mock_server.patch_external_api(batch_files_url, bad_request):
            resp = self.client.get(batch_files_url, headers=self.headers)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot list files in batch", resp.json()["error"]["message"])

    def test_batch_created_files_tracked_locally(self):
        """Test that batch-created VectorStoreFile records are created locally
        with proper upstream IDs, and listing returns upstream data directly."""

        # Create batch - this creates VectorStoreFile records with upstream IDs
        self._create_batch()

        # Verify batch-created records exist locally with file IDs as their IDs
        batch_files = VectorStoreFileModel.objects.filter(vector_store=self.vs_obj)
        self.assertEqual(batch_files.count(), 2)
        batch_file_ids = {f.id for f in batch_files}
        self.assertEqual(batch_file_ids, {"file-mock-1", "file-mock-2"})

        # List files - returns upstream data directly
        with self.mock_server.patch_external_api(self.vs_files_url, self._mock_files_list):
            resp = self.client.get(self.vs_files_url, headers=self.headers)

        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        # Should return exactly 2 files from upstream
        self.assertEqual(len(data["data"]), 2)

        # Verify local records still exist (no duplicates created)
        total_files = VectorStoreFileModel.objects.filter(vector_store=self.vs_obj).count()
        self.assertEqual(
            total_files, 2, f"Expected 2 records but found {total_files} (duplicates created)"
        )
