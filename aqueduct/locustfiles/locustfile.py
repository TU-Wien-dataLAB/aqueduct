import logging
from typing import ClassVar

from django.core.files.uploadedfile import SimpleUploadedFile
from locust import HttpUser, between, task

log = logging.getLogger(__name__)


class GatewayUser(HttpUser):
    wait_time = between(1, 3)
    headers: dict[str, str]
    multipart_headers: dict[str, str]
    # Headers with the token of the initial user that was loaded from the fixture:
    _init_user_headers: ClassVar[dict[str, str]] = {
        "Authorization": "Bearer sk-123abc",
        "Content-Type": "application/json",
    }
    host = "http://localhost:8000/"
    # Store user-specific cleanup info:
    _username: str | None = None

    def on_start(self):
        """Creates a Django user, related profile and token, and some initial resources."""
        self._create_user()

        # Create a file object for batches/files API tasks
        file = SimpleUploadedFile("test.txt", b"test user data file\n", content_type="test/plain")
        file_resp = self.client.post(
            "files",
            files={"file": ("test.txt", file, "test/plain")},
            data={"purpose": "user_data"},
            headers=self.multipart_headers,
        )
        file_resp.raise_for_status()
        self._file_id = file_resp.json()["id"]

        # Add an initial response to the cache
        resp_resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
        )
        resp_resp.raise_for_status()
        self._response_id = resp_resp.json()["id"]

        # Create a vector store for the new user
        resp = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers
        )
        resp.raise_for_status()
        self._vector_store_id = resp.json()["id"]

    def _create_user(self):
        # Note: Use the initial user's token to authenticate the request
        resp = self.client.post(
            "aqueduct/management/test-auth/generate-token/", headers=self._init_user_headers
        )
        resp.raise_for_status()
        token_data = resp.json()

        self.headers = {
            "Authorization": f"Bearer {token_data['token']}",
            "Content-Type": "application/json",
        }
        self.multipart_headers = {"Authorization": f"Bearer {token_data['token']}"}
        self._username = token_data["username"]

    def on_stop(self):
        """
        Cleans up this user's data: user, profile, and token.

        Note: This method won't run if Locus is not stopped normally; if Locus is killed
        or crashes, the user-related objects will stay in the db.
        """
        try:
            resp = self.client.post(
                "aqueduct/management/test-auth/cleanup-token/", headers=self.headers
            )
            if resp.ok:
                log.info("Cleaned up Locust user: %s", self._username)
            else:
                log.warning("Cleanup failed for %s: %s", self._username, resp.text)
        except Exception as e:
            log.exception("Cleanup exception for %s: %s", self._username, e)
            # Don't re-raise - cleanup failure shouldn't affect test metrics

    @task
    def chat_completions(self):
        _ = self.client.post(
            "chat/completions",
            json={
                "model": "main",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write me a short poem!"},
                ],
                "max_completion_tokens": 50,
            },
            headers=self.headers,
        )

    @task
    def embeddings(self):
        _ = self.client.post(
            "embeddings",
            json={"model": "embedding", "input": "Hello, world!"},
            headers=self.headers,
        )

    @task
    def image_generation(self):
        _ = self.client.post(
            "images/generations",
            json={
                "model": "image",
                "prompt": "A simple landscape painting",
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
            headers=self.headers,
        )

    @task
    def speech(self):
        _ = self.client.post(
            "audio/speech",
            json={"model": "tts", "input": "Hello, this is a test.", "voice": "alloy"},
            headers=self.headers,
        )

    @task
    def transcriptions(self):
        file = SimpleUploadedFile("test.oga", b"fake audio data", content_type="audio/ogg")
        _ = self.client.post(
            "audio/transcriptions",
            files={"file": ("test.oga", file, "audio/ogg")},
            data={"model": "transcribe"},
            headers=self.multipart_headers,
        )

    # Files API tasks
    @task
    def files_lifecycle(self):
        """Create, list, retrieve, download content, and delete a file"""

        file = SimpleUploadedFile(
            "test.jsonl",
            b'{"custom_id": "test-1", "method": "POST", "url": "/chat/completions", '
            b'"body": {"model": "main", "messages": [{"role": "user", "content": "Hello"}]}}\n',
            content_type="application/jsonl",
        )
        # Upload a file
        resp = self.client.post(
            "files",
            files={"file": ("test.jsonl", file, "application/jsonl")},
            data={"purpose": "batch"},
            headers=self.multipart_headers,
        )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        # List files
        self.client.get("files", headers=self.headers)
        # Retrieve file metadata
        self.client.get(f"files/{file_id}", headers=self.headers)
        # Download file content
        self.client.get(f"files/{file_id}/content", headers=self.headers)
        # Delete file
        self.client.delete(f"files/{file_id}", headers=self.headers)

    # Batch API tasks
    @task
    def batches_lifecycle(self):
        """Create, list, retrieve, and cancel a batch"""
        # Create a batch
        resp = self.client.post(
            "batches",
            json={
                "input_file_id": self._file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            headers=self.headers,
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        try:
            # List batches
            self.client.get("batches", headers=self.headers)
            # Retrieve batch
            self.client.get(f"batches/{batch_id}", headers=self.headers)
        finally:
            # Cancel batch
            self.client.post(f"batches/{batch_id}/cancel", headers=self.headers)

    # Responses API tasks
    @task
    def responses_lifecycle(self):
        """Create, retrieve, get input items, and delete a response"""
        # Create a response
        resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
        )
        resp.raise_for_status()
        response_id = resp.json()["id"]
        try:
            # Retrieve response
            self.client.get(f"responses/{response_id}", headers=self.headers)
            # Get input items of the response
            self.client.get(f"responses/{response_id}/input_items", headers=self.headers)
        finally:
            # Delete response
            self.client.delete(f"responses/{response_id}", headers=self.headers)

    # Vector Stores tasks
    @task
    def vector_store_lifecycle(self):
        """Create, list, retrieve, search, update, and delete a vector store"""

        # Create a vector store
        resp = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers
        )
        resp.raise_for_status()
        vs_id = resp.json()["id"]
        try:
            # List vector stores
            self.client.get("vector_stores", headers=self.headers)
            # Retrieve a vector store
            self.client.get(f"vector_stores/{vs_id}", headers=self.headers)
            # Search a vector store
            self.client.post(
                f"vector_stores/{vs_id}/search", json={"query": "test query"}, headers=self.headers
            )
            # Update a vector store
            self.client.post(
                f"vector_stores/{vs_id}",
                json={"name": "updated-vector-store"},
                headers=self.headers,
            )
        finally:
            # Delete a vector store
            self.client.delete(f"vector_stores/{vs_id}", headers=self.headers)

    # Vector Store Files tasks
    @task
    def vector_store_files_lifecycle(self):
        """Create, list, retrieve, get content, update, and delete a vector store file"""
        # Add a vector store file
        resp = self.client.post(
            f"vector_stores/{self._vector_store_id}/files",
            json={"file_id": self._file_id},
            headers=self.headers,
        )
        resp.raise_for_status()
        vsf_id = resp.json()["id"]
        try:
            # List vector store files
            self.client.get(f"vector_stores/{self._vector_store_id}/files", headers=self.headers)
            # Retrieve vector store file
            self.client.get(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}", headers=self.headers
            )
            # Get content of vector store file
            self.client.get(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}/content",
                headers=self.headers,
            )
            # Update vector store file
            self.client.post(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}",
                json={"attributes": {"key": "value"}},
                headers=self.headers,
            )
        finally:
            # Delete vector store file
            self.client.delete(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}", headers=self.headers
            )

    # Vector Store File Batches tasks
    @task
    def vector_store_file_batch_lifecycle(self):
        """Create, retrieve, list files of, and cancel a vector store file batch"""
        # Setup: create a new file object
        file = SimpleUploadedFile("test.txt", b"test user data file\n", content_type="test/plain")
        file_resp = self.client.post(
            "files",
            files={"file": ("test.txt", file, "test/plain")},
            data={"purpose": "user_data"},
            headers=self.multipart_headers,
        )
        file_resp.raise_for_status()
        new_file_id = file_resp.json()["id"]

        try:
            # Create a vector store file batch
            resp = self.client.post(
                f"vector_stores/{self._vector_store_id}/file_batches",
                json={"file_ids": [new_file_id]},
                headers=self.headers,
            )
            resp.raise_for_status()
            vsb_id = resp.json()["id"]
            try:
                # Retrieve a vector store file batch
                self.client.get(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}",
                    headers=self.headers,
                )
                # List files of a vector store file batch
                self.client.get(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}/files",
                    headers=self.headers,
                )
            finally:
                # Cancel a vector store file batch
                self.client.post(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}/cancel",
                    headers=self.headers,
                )
        finally:
            # Cleanup: delete the created file object
            self.client.delete(f"files/{new_file_id}", headers=self.headers)
