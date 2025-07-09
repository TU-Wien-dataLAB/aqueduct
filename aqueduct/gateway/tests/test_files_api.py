from django.core.files.uploadedfile import SimpleUploadedFile

from gateway.tests.utils.base import GatewayFilesTestCase
from gateway.tests.utils import _build_chat_headers


class TestFilesAPI(GatewayFilesTestCase):
    def setUp(self):
        super().setUp()
        # Prepare auth headers for file API
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        # Remove Content-Type header to allow multipart file upload
        headers.pop("Content-Type", None)
        self.headers = headers

    def test_file_lifecycle(self):
        """Test uploading, listing, retrieving, downloading, and deleting a file."""
        content = b'{"foo": "bar"}\n{"baz": 123}\n{"foo": "bar"}\n{"baz": 123}\n'
        upload_file = SimpleUploadedFile("test.jsonl", content, content_type="application/jsonl")
        # Upload file
        response = self.client.post(
            "/files",
            {"file": upload_file, "purpose": "batch"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, f"Upload failed: {response.json()}")
        upload_data = response.json()
        file_id = upload_data["id"]

        # List files
        response = self.client.get("/files", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 1)
        self.assertIn(file_id, [f["id"] for f in list_data["data"]])

        # Retrieve file metadata
        response = self.client.get(f"/files/{file_id}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        meta_data = response.json()
        self.assertEqual(meta_data["filename"], upload_data["filename"])

        # Download file content
        response = self.client.get(f"/files/{file_id}/content", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, content)

        # Delete file
        response = self.client.delete(f"/files/{file_id}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        delete_data = response.json()
        self.assertTrue(delete_data.get("deleted"))

        # Ensure file is no longer listed
        response = self.client.get("/files", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        list_data = response.json()
        self.assertEqual(len(list_data["data"]), 0)
        self.assertNotIn(file_id, [f["id"] for f in list_data["data"]])
