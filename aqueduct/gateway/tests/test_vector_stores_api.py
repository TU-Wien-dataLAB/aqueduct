import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from openai.pagination import AsyncCursorPage, AsyncPage
from openai.types import VectorStore
from openai.types.vector_store import FileCounts as VectorStoreFileCounts
from openai.types.vector_store_search_response import Content, VectorStoreSearchResponse
from openai.types.vector_stores import VectorStoreFile, VectorStoreFileBatch, VectorStoreFileDeleted
from openai.types.vector_stores.vector_store_file_batch import FileCounts

from gateway.tests.utils.base import GatewayFilesTestCase
from management.models import (
    FileObject,
    ServiceAccount,
    Team,
    Token,
    VectorStoreFileBatchStatus,
    VectorStoreStatus,
)
from management.models import VectorStore as VectorStoreModel
from management.models import VectorStoreFile as VectorStoreFileModel
from management.models import VectorStoreFileBatch as VectorStoreFileBatchModel
from mock_api.mock_configs import MockConfig


def create_mock_vector_store(
    id_suffix: str = "123",
    status: Literal["expired", "in_progress", "completed"] = "completed",
    name: str = "Test Store",
) -> VectorStore:
    """Create a mock vector store with given ID suffix and status using OpenAI types."""
    now = int(timezone.now().timestamp())
    return VectorStore(
        id=f"vs-mock-{id_suffix}",
        name=name,
        status=status,
        usage_bytes=0,
        created_at=now,
        expires_after=None,
        metadata=None,
        object="vector_store",
        file_counts=VectorStoreFileCounts(
            total=0, completed=0, failed=0, in_progress=0, cancelled=0
        ),
        last_active_at=None,
        expires_at=None,
    )


def create_mock_vector_store_file(
    id_suffix: str = "123",
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "completed",
) -> VectorStoreFile:
    """Create a mock vector store file with given ID suffix and status using OpenAI types."""
    now = int(timezone.now().timestamp())
    return VectorStoreFile(
        id=f"vsf-mock-{id_suffix}",
        status=status,
        usage_bytes=100,
        created_at=now,
        last_error=None,
        object="vector_store.file",
        vector_store_id=f"vs-mock-{id_suffix}",
        attributes=None,
        chunking_strategy=None,
    )


def create_mock_file_batch(
    id_suffix: str = "123",
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "in_progress",
) -> VectorStoreFileBatch:
    """Create a mock vector store file batch with given ID suffix and status using OpenAI types."""
    now = int(timezone.now().timestamp())
    return VectorStoreFileBatch(
        id=f"vsb-mock-{id_suffix}",
        status=status,
        created_at=now,
        file_counts=FileCounts(total=2, completed=0, failed=0, in_progress=2, cancelled=0),
        object="vector_store.files_batch",
        vector_store_id=f"vs-mock-{id_suffix}",
    )


def create_mock_vector_store_client():
    """Create a fully mocked OpenAI client for vector stores API."""
    mock_client = MagicMock()

    # Mock vector store operations
    vs_counter = [0]

    async def mock_vs_create(*args, **kwargs):
        vs_counter[0] += 1
        now = int(timezone.now().timestamp())
        return VectorStore(
            id=f"vs-mock-{vs_counter[0]}",
            name=kwargs.get("name", "Test Store"),
            status="completed",
            usage_bytes=0,
            created_at=now,
            expires_after=kwargs.get("expires_after"),
            metadata=kwargs.get("metadata"),
            object="vector_store",
            file_counts=VectorStoreFileCounts(
                total=0, completed=0, failed=0, in_progress=0, cancelled=0
            ),
            last_active_at=None,
            expires_at=None,
        )

    async def mock_vs_retrieve(*args, **kwargs):
        vs_id = kwargs.get("vector_store_id", args[0] if args else "vs-mock-123")
        # Extract the suffix from the full ID (e.g. "vs-mock-1" -> "1")
        suffix = vs_id.replace("vs-mock-", "") if vs_id.startswith("vs-mock-") else "123"
        return create_mock_vector_store(suffix, "completed")

    async def mock_vs_delete(*args, **kwargs):
        return MagicMock()

    async def mock_vs_update(*args, **kwargs):
        vs_id = kwargs.get("vector_store_id", args[0] if args else "vs-mock-123")
        suffix = vs_id.replace("vs-mock-", "") if vs_id.startswith("vs-mock-") else "123"
        return create_mock_vector_store(suffix, "completed", name=kwargs.get("name", "Test Store"))

    mock_client.vector_stores.create = AsyncMock(side_effect=mock_vs_create)
    mock_client.vector_stores.retrieve = AsyncMock(side_effect=mock_vs_retrieve)
    mock_client.vector_stores.delete = AsyncMock(side_effect=mock_vs_delete)
    mock_client.vector_stores.update = AsyncMock(side_effect=mock_vs_update)

    # Mock vector store search - returns a proper AsyncPage with VectorStoreSearchResponse
    async def mock_vs_search(*args, **kwargs):
        return AsyncPage[VectorStoreSearchResponse](
            object="vector_store.search_results.page",
            data=[
                VectorStoreSearchResponse(
                    file_id="file-mock-123",
                    filename="test.txt",
                    score=0.95,
                    attributes={},
                    content=[Content(text="Test content", type="text")],
                )
            ],
        )

    mock_client.vector_stores.search = AsyncMock(side_effect=mock_vs_search)

    # Mock vector store files - track created files so list returns them
    # In the OpenAI API, VectorStoreFile.id IS the source file ID.
    vsf_counter = [0]
    created_vs_files = []

    async def mock_vsf_create(*args, **kwargs):
        vsf_counter[0] += 1
        file_id = kwargs.get("file_id", f"vsf-mock-{vsf_counter[0]}")
        now = int(timezone.now().timestamp())
        vs_file = VectorStoreFile(
            id=file_id,
            status="completed",
            usage_bytes=100,
            created_at=now,
            last_error=None,
            object="vector_store.file",
            vector_store_id=kwargs.get("vector_store_id", f"vs-mock-{vsf_counter[0]}"),
            attributes=None,
            chunking_strategy=None,
        )
        created_vs_files.append(vs_file)
        return vs_file

    mock_client.vector_stores.files.create = AsyncMock(side_effect=mock_vsf_create)

    async def mock_vsf_retrieve(*args, **kwargs):
        file_id = kwargs.get("file_id", "")
        for f in created_vs_files:
            if f.id == file_id:
                return f
        return create_mock_vector_store_file("123", "completed")

    mock_client.vector_stores.files.retrieve = AsyncMock(side_effect=mock_vsf_retrieve)

    # Mock file delete - returns a proper delete response and removes from tracking
    async def mock_vs_file_delete(*args, **kwargs):
        file_id = kwargs.get("file_id", "")
        for i, f in enumerate(created_vs_files):
            if f.id == file_id:
                created_vs_files.pop(i)
                break
        return VectorStoreFileDeleted(id=file_id, deleted=True, object="vector_store.file.deleted")

    mock_client.vector_stores.files.delete = AsyncMock(side_effect=mock_vs_file_delete)

    # Mock file update - returns file with updated attributes
    async def mock_vs_file_update(*args, **kwargs):
        file_id = kwargs.get("file_id", "")
        for f in created_vs_files:
            if f.id == file_id:
                if kwargs.get("attributes"):
                    f.attributes = kwargs["attributes"]
                return f
        file = create_mock_vector_store_file("123", "completed")
        if kwargs.get("attributes"):
            file.attributes = kwargs["attributes"]
        return file

    mock_client.vector_stores.files.update = AsyncMock(side_effect=mock_vs_file_update)

    # Mock file content - returns bytes
    async def mock_vs_file_content(*args, **kwargs):
        return b"Test file content"

    mock_client.vector_stores.files.content = AsyncMock(side_effect=mock_vs_file_content)

    # Mock files list - returns tracked created files
    async def mock_vs_files_list(*args, **kwargs):
        return AsyncCursorPage[VectorStoreFile](data=list(created_vs_files), has_more=False)

    mock_client.vector_stores.files.list = AsyncMock(side_effect=mock_vs_files_list)

    # Mock file batches
    batch_counter = [0]

    async def mock_batch_create(*args, **kwargs):
        batch_counter[0] += 1
        return create_mock_file_batch(str(batch_counter[0]), "in_progress")

    async def mock_batch_retrieve(*args, **kwargs):
        batch_id = kwargs.get("batch_id", args[0] if args else "vsb-mock-123")
        suffix = batch_id.replace("vsb-mock-", "") if batch_id.startswith("vsb-mock-") else "123"
        return create_mock_file_batch(suffix, "completed")

    async def mock_batch_cancel(*args, **kwargs):
        batch_id = kwargs.get("batch_id", args[0] if args else "vsb-mock-123")
        suffix = batch_id.replace("vsb-mock-", "") if batch_id.startswith("vsb-mock-") else "123"
        return create_mock_file_batch(suffix, "cancelled")

    mock_client.vector_stores.file_batches.create = AsyncMock(side_effect=mock_batch_create)
    mock_client.vector_stores.file_batches.retrieve = AsyncMock(side_effect=mock_batch_retrieve)
    mock_client.vector_stores.file_batches.cancel = AsyncMock(side_effect=mock_batch_cancel)

    # Mock batch files list - returns a mock response with data attribute
    async def mock_batch_files_list(*args, **kwargs):
        return AsyncCursorPage[VectorStoreFile](
            data=[create_mock_vector_store_file("123", "completed")], has_more=False
        )

    mock_client.vector_stores.file_batches.list_files = AsyncMock(side_effect=mock_batch_files_list)

    return mock_client


@override_settings(MAX_USER_VECTOR_STORES=3, MAX_TEAM_VECTOR_STORES=10, MAX_VECTOR_STORE_FILES=100)
class TestVectorStoresAPI(GatewayFilesTestCase):
    url_vector_stores = reverse("gateway:vector_stores")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.token = Token.objects.first()

    def setUp(self):
        super().setUp()
        self.vs_obj = self._create_vector_store()
        self.vs_id = self.vs_obj.id

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
        batch_id = resp.json()["id"]
        return batch_id

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
        user_stores = []
        for i in range(settings.MAX_USER_VECTOR_STORES):
            user_stores.append(
                self._create_vector_store(vs_id=f"vs-mock-{i}", name=f"Store nr {i}")
            )

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

    def test_vector_store_file_lifecycle(self):
        """Test adding, listing, retrieving, and removing files from vector store."""

        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
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
        resp = self.client.get(files_url, headers=self.headers)
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

    def test_list_files_returns_upstream_data(self):
        """Test that listing files returns upstream data directly without local syncing."""

        # Assert that there are no files in the database
        self.assertEqual(VectorStoreFileModel.objects.count(), 0)

        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.get(files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        # Verify file is returned in response with upstream ID
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "vsf-mock-123")
        self.assertEqual(data["data"][0]["status"], "completed")
        self.assertEqual(data["data"][0]["vector_store_id"], self.vs_id)

    def test_nonexistent_file_for_vs_file(self):
        """POST with invalid file_id returns 404."""
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": "nonexistent-file-id"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

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

        # Try to create a VS file without file_id
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url, data=json.dumps({}), headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file_id: Field required", resp.json()["error"]["message"])

    def test_isolation_between_tokens(self):
        """Users can't see each other's vector stores."""
        # Create new user and token
        other_token_value, other_user_id = self.create_new_user()
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

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_create(self, mock_get_client):
        """Test 502 response when upstream create fails."""
        mock_client = MagicMock()
        mock_client.vector_stores.create = AsyncMock(
            side_effect=Exception("Upstream connection failed")
        )
        mock_get_client.return_value = mock_client

        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({"name": "Wrong Store"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_retrieve(self, mock_get_client):
        """Test 502 response when upstream retrieve fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client

        # Mock failure for vector store retrieve
        mock_client.vector_stores.retrieve = AsyncMock(side_effect=Exception("Upstream timeout"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.get(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_update(self, mock_get_client):
        """Test 502 response when upstream update fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client
        # Mock failure for update
        mock_client.vector_stores.update = AsyncMock(side_effect=Exception("Upstream error"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            vs_url,
            data=json.dumps({"name": "Updated Name"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_delete(self, mock_get_client):
        """Test 502 response when upstream delete fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client
        # Mock failure for delete
        mock_client.vector_stores.delete = AsyncMock(side_effect=Exception("Upstream error"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.delete(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    def test_batch_file_counts(self):
        """Batch correctly tracks completed/failed files."""
        # TODO: Where do we track completed/failed files there?
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
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["file_counts"]["total"], 2)
        self.assertEqual(data["file_counts"]["completed"], 0)
        batch_id = data["id"]

        # Get batch and verify file counts are returned
        batch_url = reverse(
            "gateway:vector_store_file_batch",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("file_counts", data)
        self.assertEqual(data["file_counts"]["total"], 2)

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
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        extra_file = self._create_file_object("file-mock-extra")
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": extra_file.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("limit reached", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_file_batch_upstream_failure(self, mock_batch_client, mock_vs_client):
        """Test 502 response when upstream batch create fails."""
        mock_vs_client.return_value = create_mock_vector_store_client()

        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        # Mock batch client to fail
        mock_batch = MagicMock()
        mock_batch.vector_stores.file_batches.create = AsyncMock(
            side_effect=Exception("Upstream batch error")
        )
        mock_batch_client.return_value = mock_batch

        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_file_batch_cancel_upstream_failure(self, mock_batch_client, mock_vs_client):
        """Test 502 response when upstream batch cancel fails."""
        mock_vs_client.return_value = create_mock_vector_store_client()
        mock_batch_client.return_value = create_mock_vector_store_client()

        batch_id = self._create_batch()

        # Now mock failure for cancel
        mock_batch_client.return_value.vector_stores.file_batches.cancel = AsyncMock(
            side_effect=Exception("Upstream cancel error")
        )

        cancel_url = reverse(
            "gateway:vector_store_file_batch_cancel",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        resp = self.client.post(cancel_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    # TODO
    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_files.get_files_api_client")
    def test_vector_store_file_upstream_failure(self, mock_vs_files_client, mock_vs_client):
        """Test 502 response when upstream file operations fail."""
        mock_vs_client.return_value = create_mock_vector_store_client()

        # Create file object
        file_obj = self._create_file_object()

        # Mock file operations to fail
        mock_vs_files_client.return_value = MagicMock()
        mock_vs_files_client.return_value.vector_stores.files.create = AsyncMock(
            side_effect=Exception("Upstream file create error")
        )

        # Try to add file - should fail with 502
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

        # Now create successfully and test retrieve failure
        mock_vs_files_client.return_value = create_mock_vector_store_client()
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        vsf_id = resp.json()["id"]

        # Mock retrieve to fail
        mock_vs_files_client.return_value.vector_stores.files.retrieve = AsyncMock(
            side_effect=Exception("Upstream file retrieve error")
        )

        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id}
        )
        resp = self.client.get(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)

        # Mock delete to fail
        mock_vs_files_client.return_value.vector_stores.files.delete = AsyncMock(
            side_effect=Exception("Upstream file delete error")
        )

        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)

    def test_vector_store_search(self):
        """Test searching a vector store with file_id mapping."""
        file_obj = self._create_file_object("file-mock-123")

        search_url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            search_url,
            data=json.dumps({"query": "test query"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Search failed: {resp.json()}")
        data = resp.json()
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["file_id"], file_obj.id)

    def test_vector_store_search_missing_query(self):
        """Test search without query returns 400."""
        search_url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            search_url, data=json.dumps({}), headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Missing required parameter: query", resp.json()["error"]["message"])

    def test_vector_store_file_update_attributes(self):
        """Test updating file attributes."""

        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
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
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
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

    # # TODO
    # def test_vector_store_file_content(self):
    #     """Test getting file content."""
    #     # Create file object
    #     file_obj = self._create_file_object()
    #
    #     # Add file to vector store
    #     files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
    #     resp = self.client.post(
    #         files_url,
    #         data=json.dumps({"file_id": file_obj.id}),
    #         headers=self.headers,
    #         content_type="application/json",
    #     )
    #     self.assertEqual(resp.status_code, 200)
    #     vsf_id = resp.json()["id"]
    #
    #     # Get file content
    #     content_url = reverse(
    #         "gateway:vector_store_file_content",
    #         kwargs={"vector_store_id": self.vs_id, "file_id": vsf_id},
    #     )
    #     resp = self.client.get(content_url, headers=self.headers)
    #     self.assertEqual(resp.status_code, 200)
    #     self.assertEqual(resp.content, b"Test file content")

    def test_vector_store_file_batch_files(self):
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

    def test_batch_created_files_tracked_locally(self):
        """Test that batch-created VectorStoreFile records are created locally
        with proper upstream IDs, and listing returns upstream data directly."""

        # Create batch - this creates VectorStoreFile records with upstream IDs
        self._create_batch()

        # Verify batch-created records exist locally with file IDs as their IDs
        batch_files = VectorStoreFileModel.objects.filter(vector_store=self.vs_obj)
        self.assertEqual(batch_files.count(), 2)
        batch_file_ids = set(f.id for f in batch_files)
        self.assertEqual(batch_file_ids, {"file-mock-1", "file-mock-2"})

        # List files - returns upstream data directly
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})

        with self.mock_server.patch_external_api(files_url, self._mock_files_list):
            resp = self.client.get(files_url, headers=self.headers)

        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        # Should return exactly 2 files from upstream
        self.assertEqual(len(data["data"]), 2)

        # Verify local records still exist (no duplicates created)
        total_files = VectorStoreFileModel.objects.filter(vector_store=self.vs_obj).count()
        self.assertEqual(
            total_files, 2, f"Expected 2 records but found {total_files} (duplicates created)"
        )

    def test_list_vector_store_files_response_structure(self):
        """Test that list vector store files endpoint returns complete, correctly-mapped response items."""

        file_obj1 = self._create_file_object("file-remote-1")
        file_obj2 = self._create_file_object("file-remote-2")

        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": file_obj1.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": file_obj2.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        with self.mock_server.patch_external_api(files_url, self._mock_files_list):
            resp = self.client.get(files_url, headers=self.headers)

        self.assertEqual(resp.status_code, 200, f"List files failed: {resp.json()}")
        data = resp.json()

        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 2)

        for item in data["data"]:
            # ID is now the upstream ID
            # file_id and vector_store_id are upstream IDs
            self.assertEqual(item["status"], "completed")
            self.assertEqual(item["object"], "vector_store.file")

    # TODO
    @patch("gateway.views.files.get_files_api_client")
    def test_service_account_file_operations(self, mock_get_client):
        """Test that the files API works with service account tokens."""
        mock_client = MagicMock()

        async def mock_file_create(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                bytes=100,
                filename="test.txt",
                purpose="user_data",
                created_at=int(timezone.now().timestamp()),
                expires_at=None,
                model_dump=MagicMock(
                    return_value={
                        "id": "file-remote-new",
                        "bytes": 100,
                        "filename": "test.txt",
                        "purpose": "user_data",
                        "created_at": int(timezone.now().timestamp()),
                        "expires_at": None,
                    }
                ),
            )

        async def mock_file_retrieve(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                bytes=100,
                filename="test.txt",
                purpose="user_data",
                created_at=int(timezone.now().timestamp()),
                expires_at=None,
                model_dump=MagicMock(
                    return_value={
                        "id": "file-remote-new",
                        "bytes": 100,
                        "filename": "test.txt",
                        "purpose": "user_data",
                        "created_at": int(timezone.now().timestamp()),
                        "expires_at": None,
                    }
                ),
            )

        async def mock_file_delete(*args, **kwargs):
            return MagicMock(
                id="file-remote-new",
                deleted=True,
                model_dump=MagicMock(return_value={"id": "file-remote-new", "deleted": True}),
            )

        async def mock_file_content(*args, **kwargs):
            return MagicMock(content=b"test file content")

        mock_client.files.create = AsyncMock(side_effect=mock_file_create)
        mock_client.files.retrieve = AsyncMock(side_effect=mock_file_retrieve)
        mock_client.files.delete = AsyncMock(side_effect=mock_file_delete)
        mock_client.files.content = AsyncMock(side_effect=mock_file_content)
        mock_get_client.return_value = mock_client

        team = Team.objects.get(name="Whale")
        service_account = ServiceAccount.objects.create(team=team, name="Test Service Account")

        token = Token.objects.get(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN))
        token.service_account = service_account
        token.save()

        try:
            file = SimpleUploadedFile("test.txt", b"test file content", content_type="text/plain")
            files_url = reverse("gateway:files")
            resp = self.client.post(
                files_url, data={"file": file, "purpose": "user_data"}, headers=self.headers
            )
            self.assertEqual(resp.status_code, 200, f"Upload failed: {resp.json()}")
            file_id = resp.json()["id"]

            resp = self.client.get(files_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"List failed: {resp.json()}")
            self.assertEqual(len(resp.json()["data"]), 1)

            file_detail_url = reverse("gateway:file", kwargs={"file_id": file_id})
            resp = self.client.get(file_detail_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Retrieve failed: {resp.json()}")

            file_content_url = reverse("gateway:file_content", kwargs={"file_id": file_id})
            resp = self.client.get(file_content_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Content failed: {resp.content}")
            self.assertEqual(resp.content, b"test file content")

            resp = self.client.delete(file_detail_url, headers=self.headers)
            self.assertEqual(resp.status_code, 200, f"Delete failed: {resp.json()}")

        finally:
            token.service_account = None
            token.save()
            service_account.delete()
