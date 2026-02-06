import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from openai.types import VectorStore
from openai.types.vector_store import FileCounts as VectorStoreFileCounts
from openai.types.vector_stores import VectorStoreFile, VectorStoreFileBatch
from openai.types.vector_stores.vector_store_file_batch import FileCounts

from gateway.tests.utils.base import GatewayFilesTestCase
from management.models import FileObject, Token
from management.models import VectorStore as VectorStoreModel
from management.models import VectorStoreFile as VectorStoreFileModel
from management.models import VectorStoreFileBatch as VectorStoreFileBatchModel


def create_mock_vector_store(
    id_suffix: str = "123", status: Literal["expired", "in_progress", "completed"] = "completed"
) -> VectorStore:
    """Create a mock vector store with given ID suffix and status using OpenAI types."""
    now = int(timezone.now().timestamp())
    return VectorStore(
        id=f"vs-mock-{id_suffix}",
        name="Test Store",
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
        return create_mock_vector_store("123", "completed")

    async def mock_vs_delete(*args, **kwargs):
        return MagicMock()

    async def mock_vs_update(*args, **kwargs):
        return create_mock_vector_store("123", "completed")

    mock_client.vector_stores.create = AsyncMock(side_effect=mock_vs_create)
    mock_client.vector_stores.retrieve = AsyncMock(side_effect=mock_vs_retrieve)
    mock_client.vector_stores.delete = AsyncMock(side_effect=mock_vs_delete)
    mock_client.vector_stores.update = AsyncMock(side_effect=mock_vs_update)

    # Mock vector store files
    mock_client.vector_stores.files.create = AsyncMock(
        return_value=create_mock_vector_store_file("123", "completed")
    )
    mock_client.vector_stores.files.retrieve = AsyncMock(
        return_value=create_mock_vector_store_file("123", "completed")
    )
    mock_client.vector_stores.files.delete = AsyncMock(return_value=MagicMock())

    # Mock file batches
    batch_counter = [0]

    async def mock_batch_create(*args, **kwargs):
        batch_counter[0] += 1
        return create_mock_file_batch(str(batch_counter[0]), "in_progress")

    async def mock_batch_retrieve(*args, **kwargs):
        return create_mock_file_batch("123", "completed")

    async def mock_batch_cancel(*args, **kwargs):
        return create_mock_file_batch("123", "cancelled")

    mock_client.vector_stores.file_batches.create = AsyncMock(side_effect=mock_batch_create)
    mock_client.vector_stores.file_batches.retrieve = AsyncMock(side_effect=mock_batch_retrieve)
    mock_client.vector_stores.file_batches.cancel = AsyncMock(side_effect=mock_batch_cancel)

    return mock_client


@override_settings(
    MAX_USER_VECTOR_STORES=3,
    MAX_TEAM_VECTOR_STORES=10,
    MAX_VECTOR_STORE_FILES=100,
    AQUEDUCT_FILES_API_URL="https://api.openai.com",
    AQUEDUCT_FILES_API_KEY="test_key",
)
class TestVectorStoresAPI(GatewayFilesTestCase):
    url_vector_stores = reverse("gateway:vector_stores")

    def tearDown(self):
        # Clean up local records
        VectorStoreFileModel.objects.all().delete()
        VectorStoreFileBatchModel.objects.all().delete()
        VectorStoreModel.objects.all().delete()
        super().tearDown()

    def _create_vector_store(self, mock_client, name="Test Store"):
        """Helper to create a vector store and return its ID."""
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({"name": name}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create failed: {resp.json()}")
        return resp.json()["id"]

    def _create_file_object(self, remote_id="file-mock-123"):
        """Helper to create a FileObject for testing."""

        token = Token.objects.first()
        file_obj = FileObject.objects.create(
            bytes=100,
            created_at=int(timezone.now().timestamp()),
            filename="test.txt",
            purpose="user_data",
            token=token,
            remote_id=remote_id,
        )
        return file_obj

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_vector_store_lifecycle(self, mock_get_client):
        """Test full lifecycle: create, list, get, modify, delete vector store."""
        mock_get_client.return_value = create_mock_vector_store_client()

        # Create vector store
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({"name": "My Test Store"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create failed: {resp.json()}")
        data = resp.json()
        self.assertTrue(data["id"].startswith("vs-"))  # Aqueduct ID
        self.assertEqual(data["name"], "My Test Store")
        vs_id = data["id"]

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

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_max_vector_stores_limit(self, mock_get_client):
        """Test that MAX_USER_VECTOR_STORES limit is enforced."""
        mock_get_client.return_value = create_mock_vector_store_client()

        # Create up to limit
        for i in range(3):  # MAX_USER_VECTOR_STORES=3
            resp = self.client.post(
                self.url_vector_stores,
                data=json.dumps({"name": f"Store {i}"}),
                headers=self.headers,
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)

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

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_files.get_files_api_client")
    def test_vector_store_file_lifecycle(self, mock_vs_files_client, mock_vs_client):
        """Test adding, listing, retrieving, and removing files from vector store."""
        mock_vs_client.return_value = create_mock_vector_store_client()
        mock_vs_files_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

        # Create file object
        file_obj = self._create_file_object()

        # Add file to vector store
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": vs_id})
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": file_obj.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Add file failed: {resp.json()}")
        data = resp.json()
        self.assertTrue(data["id"].startswith("vsf-"))  # Aqueduct ID
        vsf_id = data["id"]

        # List files in vector store
        resp = self.client.get(files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], vsf_id)

        # Get file
        file_url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": vs_id, "file_id": vsf_id}
        )
        resp = self.client.get(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], vsf_id)
        self.assertEqual(data["file_id"], file_obj.id)

        # Delete file
        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["deleted"])

        # Verify deleted
        resp = self.client.get(files_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["data"]), 0)

    @patch("gateway.views.vector_store_files.get_files_api_client")
    def test_nonexistent_file_for_vs_file(self, mock_get_client):
        """POST with invalid file_id returns 404."""
        mock_get_client.return_value = create_mock_vector_store_client()

        # Create vector store first
        with patch("gateway.views.vector_stores.get_files_api_client") as mock_vs_client:
            mock_vs_client.return_value = create_mock_vector_store_client()
            vs_id = self._create_vector_store(mock_vs_client)

        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": vs_id})
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": "nonexistent-file-id"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_file_batch_lifecycle(self, mock_batch_client, mock_vs_client):
        """Test creating, retrieving, and canceling file batches."""
        mock_vs_client.return_value = create_mock_vector_store_client()
        mock_batch_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        # Create batch
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create batch failed: {resp.json()}")
        data = resp.json()
        self.assertTrue(data["id"].startswith("vsb-"))  # Aqueduct ID
        batch_id = data["id"]

        # Get batch
        batch_url = reverse(
            "gateway:vector_store_file_batch",
            kwargs={"vector_store_id": vs_id, "batch_id": batch_id},
        )
        resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], batch_id)
        self.assertEqual(data["file_counts"]["total"], 2)

        # Cancel batch
        cancel_url = reverse(
            "gateway:vector_store_file_batch_cancel",
            kwargs={"vector_store_id": vs_id, "batch_id": batch_id},
        )
        resp = self.client.post(cancel_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "cancelled")

    def test_validation_errors(self):
        """Missing required fields return 400."""
        # Create without name
        resp = self.client.post(
            self.url_vector_stores,
            data=json.dumps({}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

        # Create file in vector store without file_id
        with patch("gateway.views.vector_stores.get_files_api_client") as mock_vs_client:
            mock_vs_client.return_value = create_mock_vector_store_client()
            vs_id = self._create_vector_store(mock_vs_client)

        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": vs_id})
        resp = self.client.post(
            files_url, data=json.dumps({}), headers=self.headers, content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_isolation_between_tokens(self, mock_get_client):
        """Users can't see each other's vector stores."""
        mock_get_client.return_value = create_mock_vector_store_client()

        # Create vector store with first user
        vs_id = self._create_vector_store(mock_get_client, "User1 Store")

        # Create new user and token
        other_token_value, other_user_id = self.create_new_user()
        other_headers = {"Authorization": f"Bearer {other_token_value}"}

        # Try to access with different user
        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_id})
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
            data=json.dumps({"name": "Test Store"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_retrieve(self, mock_get_client):
        """Test 502 response when upstream retrieve fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client

        # Create vector store first
        vs_id = self._create_vector_store(mock_get_client)

        # Now mock failure for retrieve
        mock_client.vector_stores.retrieve = AsyncMock(side_effect=Exception("Upstream timeout"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_id})
        resp = self.client.get(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_update(self, mock_get_client):
        """Test 502 response when upstream update fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client

        # Create vector store first
        vs_id = self._create_vector_store(mock_get_client)

        # Now mock failure for update
        mock_client.vector_stores.update = AsyncMock(side_effect=Exception("Upstream error"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_id})
        resp = self.client.post(
            vs_url,
            data=json.dumps({"name": "Updated Name"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    def test_upstream_failure_delete(self, mock_get_client):
        """Test 502 response when upstream delete fails."""
        mock_client = create_mock_vector_store_client()
        mock_get_client.return_value = mock_client

        # Create vector store first
        vs_id = self._create_vector_store(mock_get_client)

        # Now mock failure for delete
        mock_client.vector_stores.delete = AsyncMock(side_effect=Exception("Upstream error"))

        vs_url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_id})
        resp = self.client.delete(vs_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_batch_file_counts(self, mock_batch_client, mock_vs_client):
        """Batch correctly tracks completed/failed files."""
        mock_vs_client.return_value = create_mock_vector_store_client()
        mock_batch_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        # Create batch
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": vs_id}
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
            kwargs={"vector_store_id": vs_id, "batch_id": batch_id},
        )
        resp = self.client.get(batch_url, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("file_counts", data)
        self.assertEqual(data["file_counts"]["total"], 2)

    @patch("gateway.views.vector_store_files.get_files_api_client")
    def test_max_vector_store_files_limit(self, mock_get_client):
        """Test that MAX_VECTOR_STORE_FILES limit is enforced when adding files."""
        mock_get_client.return_value = create_mock_vector_store_client()

        # Create vector store first
        with patch("gateway.views.vector_stores.get_files_api_client") as mock_vs_client:
            mock_vs_client.return_value = create_mock_vector_store_client()
            vs_id = self._create_vector_store(mock_vs_client)

        # Create MAX_VECTOR_STORE_FILES files (set to 100 in settings)
        file_objs = []
        for i in range(100):
            file_obj = self._create_file_object(f"file-mock-{i}")
            file_objs.append(file_obj)

        # Add files up to limit
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": vs_id})
        for i, file_obj in enumerate(file_objs):
            resp = self.client.post(
                files_url,
                data=json.dumps({"file_id": file_obj.id}),
                headers=self.headers,
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200, f"Failed to add file {i}: {resp.json()}")

        # Try to add one more file - should fail with 403
        extra_file = self._create_file_object("file-mock-extra")
        resp = self.client.post(
            files_url,
            data=json.dumps({"file_id": extra_file.id}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("limit reached", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_file_batch_upstream_failure(self, mock_batch_client, mock_vs_client):
        """Test 502 response when upstream batch create fails."""
        mock_vs_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

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
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_file_batches.get_files_api_client")
    def test_file_batch_cancel_upstream_failure(self, mock_batch_client, mock_vs_client):
        """Test 502 response when upstream batch cancel fails."""
        mock_vs_client.return_value = create_mock_vector_store_client()
        mock_batch_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

        # Create file objects
        file_obj1 = self._create_file_object("file-mock-1")
        file_obj2 = self._create_file_object("file-mock-2")

        # Create batch
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": vs_id}
        )
        resp = self.client.post(
            batches_url,
            data=json.dumps({"file_ids": [file_obj1.id, file_obj2.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        batch_id = resp.json()["id"]

        # Now mock failure for cancel
        mock_batch_client.return_value.vector_stores.file_batches.cancel = AsyncMock(
            side_effect=Exception("Upstream cancel error")
        )

        cancel_url = reverse(
            "gateway:vector_store_file_batch_cancel",
            kwargs={"vector_store_id": vs_id, "batch_id": batch_id},
        )
        resp = self.client.post(cancel_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("upstream", resp.json()["error"]["message"].lower())

    @patch("gateway.views.vector_stores.get_files_api_client")
    @patch("gateway.views.vector_store_files.get_files_api_client")
    def test_vector_store_file_upstream_failure(self, mock_vs_files_client, mock_vs_client):
        """Test 502 response when upstream file operations fail."""
        mock_vs_client.return_value = create_mock_vector_store_client()

        # Create vector store
        vs_id = self._create_vector_store(mock_vs_client)

        # Create file object
        file_obj = self._create_file_object()

        # Mock file operations to fail
        mock_vs_files_client.return_value = MagicMock()
        mock_vs_files_client.return_value.vector_stores.files.create = AsyncMock(
            side_effect=Exception("Upstream file create error")
        )

        # Try to add file - should fail with 502
        files_url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": vs_id})
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
            "gateway:vector_store_file", kwargs={"vector_store_id": vs_id, "file_id": vsf_id}
        )
        resp = self.client.get(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)

        # Mock delete to fail
        mock_vs_files_client.return_value.vector_stores.files.delete = AsyncMock(
            side_effect=Exception("Upstream file delete error")
        )

        resp = self.client.delete(file_url, headers=self.headers)
        self.assertEqual(resp.status_code, 502)
