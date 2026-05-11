import logging
from typing import ClassVar

from locust import HttpUser, between, events, task
from prometheus_client import Counter, Histogram, start_http_server

log = logging.getLogger(__name__)


# Define Prometheus metrics, just for the sake of testing what works.
# TODO: if locust runs in a container, prometheus_client is not available!
requests_total = Counter("locust_requests_total", "Total requests", ["method", "name", "status"])
request_latency = Histogram("locust_request_latency_seconds", "Request latency")

start_http_server(8090)


@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    start_time,
    url,
    **kwargs,
):
    status = response.status_code if response else 0
    requests_total.labels(method=request_type, name=name, status=status).inc()
    request_latency.observe(response_time / 1000)  # Convert ms to seconds


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
        file_resp = self.client.post(
            "files",
            files={"file": ("test.txt", b"test user data file\n", "text/plain")},
            data={"purpose": "user_data"},
            headers=self.multipart_headers,
            name="setup",
        )
        file_resp.raise_for_status()
        self._file_id = file_resp.json()["id"]

        # Add an initial response to the cache
        resp_resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
            name="setup",
        )
        resp_resp.raise_for_status()
        self._response_id = resp_resp.json()["id"]

        # Create a vector store for the new user
        resp = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers, name="setup"
        )
        resp.raise_for_status()
        self._vector_store_id = resp.json()["id"]

    def _create_user(self):
        # Note: Use the initial user's token to authenticate the request
        resp = self.client.post(
            "aqueduct/management/test-auth/generate-token/",
            headers=self._init_user_headers,
            name="setup",
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
        _ = self.client.post(
            "audio/transcriptions",
            files={"file": ("test.oga", b"fake audio data", "audio/ogg")},
            data={"model": "transcribe"},
            headers=self.multipart_headers,
        )

    # Files API tasks
    @task
    def files_lifecycle(self):
        """Create, list, retrieve, download content, and delete a file"""
        content = (
            b'{"custom_id": "test-1", "method": "POST", "url": "/chat/completions", '
            b'"body": {"model": "main", "messages": [{"role": "user", "content": "Hello"}]}}\n'
        )
        # Upload a file
        resp = self.client.post(
            "files",
            files={"file": ("test.jsonl", content, "application/jsonl")},
            data={"purpose": "batch"},
            headers=self.multipart_headers,
        )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        # List files
        self.client.get("files", headers=self.headers)
        # Retrieve file metadata
        self.client.get(f"files/{file_id}", headers=self.headers, name="/files/[id]")
        # Download file content
        self.client.get(
            f"files/{file_id}/content", headers=self.headers, name="/files/[id]/content"
        )
        # Delete file
        self.client.delete(f"files/{file_id}", headers=self.headers, name="/files/[id]")

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
            self.client.get(f"batches/{batch_id}", headers=self.headers, name="/batches/[id]")
        finally:
            # Cancel batch
            self.client.post(
                f"batches/{batch_id}/cancel", headers=self.headers, name="/batches/[id]/cancel"
            )

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
            self.client.get(
                f"responses/{response_id}", headers=self.headers, name="/responses/[id]"
            )
            # Get input items of the response
            self.client.get(
                f"responses/{response_id}/input_items",
                headers=self.headers,
                name="/responses/[id]/input_items",
            )
        finally:
            # Delete response
            self.client.delete(
                f"responses/{response_id}", headers=self.headers, name="/responses/[id]"
            )

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
            self.client.get(
                f"vector_stores/{vs_id}", headers=self.headers, name="/vector_stores/[id]"
            )
            # Search a vector store
            self.client.post(
                f"vector_stores/{vs_id}/search",
                json={"query": "test query"},
                headers=self.headers,
                name="/vector_stores/[id]/search",
            )
            # Update a vector store
            self.client.post(
                f"vector_stores/{vs_id}",
                json={"name": "updated-vector-store"},
                headers=self.headers,
                name="/vector_stores/[id]",
            )
        finally:
            # Delete a vector store
            self.client.delete(
                f"vector_stores/{vs_id}", headers=self.headers, name="/vector_stores/[id]"
            )

    # Vector Store Files tasks
    @task
    def vector_store_files_lifecycle(self):
        """Create, list, retrieve, get content, update, and delete a vector store file"""
        # Add a vector store file
        resp = self.client.post(
            f"vector_stores/{self._vector_store_id}/files",
            json={"file_id": self._file_id},
            headers=self.headers,
            name="/vector_stores/[id]/files",
        )
        resp.raise_for_status()
        vsf_id = resp.json()["id"]
        try:
            # List vector store files
            self.client.get(
                f"vector_stores/{self._vector_store_id}/files",
                headers=self.headers,
                name="/vector_stores/[id]/files",
            )
            # Retrieve vector store file
            self.client.get(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}",
                headers=self.headers,
                name="/vector_stores/[id]/files/[id]",
            )
            # Get content of vector store file
            self.client.get(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}/content",
                headers=self.headers,
                name="/vector_stores/[id]/files/[id]/content",
            )
            # Update vector store file
            self.client.post(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}",
                json={"attributes": {"key": "value"}},
                headers=self.headers,
                name="/vector_stores/[id]/files/[id]",
            )
        finally:
            # Delete vector store file
            self.client.delete(
                f"vector_stores/{self._vector_store_id}/files/{vsf_id}",
                headers=self.headers,
                name="/vector_stores/[id]/files/[id]",
            )

    # Vector Store File Batches tasks
    @task
    def vector_store_file_batch_lifecycle(self):
        """Create, retrieve, list files of, and cancel a vector store file batch"""
        # Setup: create a new file object
        file_resp = self.client.post(
            "files",
            files={"file": ("test.txt", b"test user data file\n", "text/plain")},
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
                name="/vector_stores/[id]/file_batches",
            )
            resp.raise_for_status()
            vsb_id = resp.json()["id"]
            try:
                # Retrieve a vector store file batch
                self.client.get(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}",
                    headers=self.headers,
                    name="/vector_stores/[id]/file_batches/[id]",
                )
                # List files of a vector store file batch
                self.client.get(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}/files",
                    headers=self.headers,
                    name="/vector_stores/[id]/file_batches/[id]/files",
                )
            finally:
                # Cancel a vector store file batch
                self.client.post(
                    f"vector_stores/{self._vector_store_id}/file_batches/{vsb_id}/cancel",
                    headers=self.headers,
                    name="/vector_stores/[id]/file_batches/[id]/cancel",
                )
        finally:
            # Cleanup: delete the created file object
            self.client.delete(f"files/{new_file_id}", headers=self.headers, name="/files/[id]")
