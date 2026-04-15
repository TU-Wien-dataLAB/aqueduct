# ruff: noqa: ERA001  # TODO: fix this later
import logging
from http import HTTPStatus
from typing import ClassVar

from django.core.files.uploadedfile import SimpleUploadedFile
from locust import HttpUser, between, task

log = logging.getLogger(__name__)


class GatewayUser(HttpUser):
    wait_time = between(1, 3)
    headers: ClassVar[dict[str, str]] = {
        "Authorization": "Bearer sk-123abc",
        "Content-Type": "application/json",
    }
    multipart_headers: ClassVar[dict[str, str]] = {"Authorization": "Bearer sk-123abc"}
    host = "http://localhost:8000/"

    def on_start(self):
        """Runs once per user when they start - creates test resources"""

        # Create a test file for files, batches, and vector store endpoints
        f = SimpleUploadedFile("test.txt", b"test file content", content_type="application/json")
        file_resp = self.client.post(
            "files",
            files={"file": ("test.txt", f, "application/json")},
            data={"purpose": "user_data"},
            headers=self.multipart_headers,
        )
        if file_resp.status_code == HTTPStatus.OK:
            self._test_file_id = file_resp.json()["id"]
        else:
            log.warning(
                "Creation of test file on start failed with code %s: %s",
                file_resp.status_code,
                file_resp.json(),
            )
            self._test_file_id = "file-123456789"  # fallback

        # Create a test batch for "retrieve" batch requests
        batch_resp = self.client.post(
            "batches",
            json={
                "input_file_id": self._test_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            headers=self.headers,
        )
        if batch_resp.status_code == HTTPStatus.OK:
            self._batch_id = batch_resp.json()["id"]
        else:
            log.warning(
                "Creation of test batch on start failed with code %s: %s",
                batch_resp.status_code,
                batch_resp.json(),
            )
            self._batch_id = "batch_123456789"  # fallback

        # Create an initial response
        resp_resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
        )
        if resp_resp.status_code == HTTPStatus.OK:
            self._response_id = resp_resp.json()["id"]
        else:
            log.warning(
                "Creation of test response on start failed with code %s: %s",
                resp_resp.status_code,
                resp_resp.json(),
            )
            self._response_id = "resp_12345abc"  # fallback

        # Create a vector store for "retrieve" and "search" vector store requests
        vs_resp = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers
        )
        if vs_resp.status_code == HTTPStatus.OK:
            self._vector_store_id = vs_resp.json()["id"]
        else:
            log.warning(
                "Creation of test vector store on start failed with code %s: %s",
                vs_resp.status_code,
                vs_resp.json(),
            )
            self._vector_store_id = "vs-mock-123"  # fallback

        # Create an initial VS file
        vsf_resp = self.client.post(
            f"vector_stores/{self._vector_store_id}/files",
            json={"file_id": self._test_file_id},
            headers=self.headers,
        )
        if vsf_resp.status_code == HTTPStatus.OK:
            self._vector_store_file_id = vsf_resp.json()["id"]
        else:
            log.warning(
                "Creation of test vector store file on start failed with code %s: %s",
                vsf_resp.status_code,
                vsf_resp.json(),
            )
            self._vector_store_file_id = "vsf-mock-123"  # fallback

        # Create an initial VS file batch
        response = self.client.post(
            f"vector_stores/{self._vector_store_id}/file_batches",
            json={"file_ids": [self._test_file_id]},
            headers=self.headers,
        )
        if response.status_code == HTTPStatus.OK:
            self._file_batch_id = response.json()["id"]
        else:
            log.warning(
                "Creation of test vector store file batch on start failed with code %s: %s",
                response.status_code,
                response.json(),
            )
            self._file_batch_id = "vsb-mock-123"  # fallback

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

    # Batch API tasks
    @task
    def list_batches(self):
        _ = self.client.get("batches", headers=self.headers)

    @task
    def create_batch(self):
        resp = self.client.post(
            "batches",
            json={
                "input_file_id": self._test_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            headers=self.headers,
        )
        if resp.status_code == HTTPStatus.OK:
            self._batch_id = resp.json()["id"]

    @task
    def get_batch(self):
        _ = self.client.get(f"batches/{self._batch_id}", headers=self.headers)

    @task
    def cancel_batch(self):
        _ = self.client.post(f"batches/{self._batch_id}/cancel", headers=self.headers)

    # Responses API tasks
    @task
    def create_response(self):
        resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
        )
        if resp.status_code == HTTPStatus.OK:
            self._response_id = resp.json()["id"]

    @task
    def get_response(self):
        _ = self.client.get(f"responses/{self._response_id}", headers=self.headers)

    @task
    # def delete_response(self):
    #     _ = self.client.delete(f"responses/{self._response_id}", headers=self.headers)

    @task
    def get_response_input_items(self):
        _ = self.client.get(f"responses/{self._response_id}/input_items", headers=self.headers)

    # Vector Stores tasks
    @task
    def list_vector_stores(self):
        _ = self.client.get("vector_stores", headers=self.headers)

    @task
    def create_vector_store(self):
        response = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers
        )
        if response.status_code == HTTPStatus.OK:
            self._vector_store_id = response.json()["id"]

    @task
    def get_vector_store(self):
        _ = self.client.get(f"vector_stores/{self._vector_store_id}", headers=self.headers)

    @task
    def update_vector_store(self):
        _ = self.client.post(
            f"vector_stores/{self._vector_store_id}",
            json={"name": "updated-vector-store"},
            headers=self.headers,
        )

    # @task
    # def delete_vector_store(self):
    #     _ = self.client.delete(f"vector_stores/{self._vector_store_id}", headers=self.headers)

    @task
    def search_vector_store(self):
        _ = self.client.post(
            f"vector_stores/{self._vector_store_id}/search",
            json={"query": "test query"},
            headers=self.headers,
        )

    # Vector Store Files tasks
    @task
    def list_vector_store_files(self):
        _ = self.client.get(f"vector_stores/{self._vector_store_id}/files", headers=self.headers)

    @task
    def add_file_to_vector_store(self):
        response = self.client.post(
            f"vector_stores/{self._vector_store_id}/files",
            json={"file_id": self._test_file_id},
            headers=self.headers,
        )
        if response.status_code == HTTPStatus.OK:
            self._vector_store_file_id = response.json()["id"]

    @task
    def get_vector_store_file(self):
        _ = self.client.get(
            f"vector_stores/{self._vector_store_id}/files/{self._vector_store_file_id}",
            headers=self.headers,
        )

    @task
    def update_vector_store_file(self):
        _ = self.client.post(
            f"vector_stores/{self._vector_store_id}/files/{self._vector_store_file_id}",
            json={"attributes": {"key": "value"}},
            headers=self.headers,
        )

    # @task
    # def delete_vector_store_file(self):
    #     _ = self.client.delete(
    #         f"vector_stores/{self._vector_store_id}/files/{self._vector_store_file_id}",
    #         headers=self.headers,
    #     )

    @task
    def get_vector_store_file_content(self):
        _ = self.client.get(
            f"vector_stores/{self._vector_store_id}/files/{self._vector_store_file_id}/content",
            headers=self.headers,
        )

    # Vector Store File Batches tasks
    @task
    def create_vector_store_file_batch(self):
        response = self.client.post(
            f"vector_stores/{self._vector_store_id}/file_batches",
            json={"file_ids": [self._test_file_id]},
            headers=self.headers,
        )
        if response.status_code == HTTPStatus.OK:
            self._file_batch_id = response.json()["id"]

    @task
    def get_vector_store_file_batch(self):
        _ = self.client.get(
            f"vector_stores/{self._vector_store_id}/file_batches/{self._file_batch_id}",
            headers=self.headers,
        )

    @task
    def cancel_vector_store_file_batch(self):
        _ = self.client.post(
            f"vector_stores/{self._vector_store_id}/file_batches/{self._file_batch_id}/cancel",
            headers=self.headers,
        )

    @task
    def list_vector_store_file_batch_files(self):
        _ = self.client.get(
            f"vector_stores/{self._vector_store_id}/file_batches/{self._file_batch_id}/files",
            headers=self.headers,
        )
