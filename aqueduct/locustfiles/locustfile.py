# ruff: noqa: ERA001  # TODO: fix this later
import logging
from http import HTTPStatus

from django.core.files.uploadedfile import SimpleUploadedFile
from locust import HttpUser, between, task

log = logging.getLogger(__name__)


class GatewayUser(HttpUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wait_time = between(1, 3)
        self.headers = {"Authorization": "Bearer sk-123abc", "Content-Type": "application/json"}
        self.multipart_headers = {"Authorization": "Bearer sk-123abc"}
        self.host = "http://localhost:8000/"

    def on_start(self):
        """Runs once per user when they start - creates test resources"""
        # Create a test file for transcriptions/batches...?
        f = SimpleUploadedFile("test.txt", b"test file content", content_type="application/json")
        response = self.client.post(
            "files",
            files={"file": ("test.txt", f, "application/json")},
            data={"purpose": "user_data"},
            headers=self.multipart_headers,
        )
        if response.status_code == HTTPStatus.OK:
            self._test_file_id = response.json()["id"]
        else:
            log.warning(
                "Creation of test file on start failed with code %s: %s",
                response.status_code,
                response.json(),
            )
            self._test_file_id = "file-123456789"  # fallback

        # TODO: create a test batch? Otherwise batch `get`/`cancel` tasks fail
        #  if they run before batch create task.
        # TODO: same for responses (get/delete need an existing one);
        #  seems like sometimes getting resp. from cache fails nonetheless
        #  (probably delete fails if it runs twice)
        # TODO: same for vector stores
        # Initialize IDs with fallback values
        self.batch_id = "batch_123456789"
        self.vector_store_id = "vs-mock-123"
        self.vector_store_file_id = "vsf-mock-123"
        self.response_id = "resp_12345abc"
        self.file_batch_id = "vsb-mock-123"

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
            self.batch_id = resp.json()["id"]

    @task
    def get_batch(self):
        _ = self.client.get(f"batches/{self.batch_id}", headers=self.headers)

    @task
    def cancel_batch(self):
        _ = self.client.post(f"batches/{self.batch_id}/cancel", headers=self.headers)

    # Responses API tasks
    @task
    def create_response(self):
        resp = self.client.post(
            "responses",
            json={"model": "main", "input": "Hello, how are you?"},
            headers=self.headers,
        )
        if resp.status_code == HTTPStatus.OK:
            self.response_id = resp.json()["id"]

    @task
    def get_response(self):
        _ = self.client.get(f"responses/{self.response_id}", headers=self.headers)

    @task
    def delete_response(self):
        _ = self.client.delete(f"responses/{self.response_id}", headers=self.headers)

    @task
    def get_response_input_items(self):
        _ = self.client.get(f"responses/{self.response_id}/input_items", headers=self.headers)

    # # Vector Stores tasks
    @task
    def list_vector_stores(self):
        _ = self.client.get("vector_stores", headers=self.headers)

    @task
    def create_vector_store(self):
        response = self.client.post(
            "vector_stores", json={"name": "test-vector-store"}, headers=self.headers
        )
        if response.status_code == HTTPStatus.OK:
            self.vector_store_id = response.json()["id"]

    @task
    def get_vector_store(self):
        _ = self.client.get(f"vector_stores/{self.vector_store_id}", headers=self.headers)

    @task
    def update_vector_store(self):
        _ = self.client.post(
            f"vector_stores/{self.vector_store_id}",
            json={"name": "updated-vector-store"},
            headers=self.headers,
        )

    @task
    def delete_vector_store(self):
        _ = self.client.delete(f"vector_stores/{self.vector_store_id}", headers=self.headers)

    @task
    def search_vector_store(self):
        _ = self.client.post(
            f"vector_stores/{self.vector_store_id}/search",
            json={"query": "test query"},
            headers=self.headers,
        )

    # # Vector Store Files tasks
    # @task
    # def list_vector_store_files(self):
    #     _ = self.client.get(
    #         f"vector_stores/{self.vector_store_id}/files",
    #         headers=self.headers,
    #     )
    #
    # @task
    # def add_file_to_vector_store(self):
    #     response = self.client.post(
    #         f"vector_stores/{self.vector_store_id}/files",
    #         json={
    #             "file_id": self._test_file_id,
    #         },
    #         headers=self.headers,
    #     )
    #     if response.status_code == HTTPStatus.OK:
    #         self.vector_store_file_id = response.json()["id"]
    #
    # @task
    # def get_vector_store_file(self):
    #     _ = self.client.get(
    #         f"vector_stores/{self.vector_store_id}/files/{self.vector_store_file_id}",
    #         headers=self.headers,
    #     )
    #
    # @task
    # def update_vector_store_file(self):
    #     _ = self.client.post(
    #         f"vector_stores/{self.vector_store_id}/files/{self.vector_store_file_id}",
    #         json={
    #             "attributes": {"key": "value"},
    #         },
    #         headers=self.headers,
    #     )
    #
    # @task
    # def delete_vector_store_file(self):
    #     _ = self.client.delete(
    #         f"vector_stores/{self.vector_store_id}/files/{self.vector_store_file_id}",
    #         headers=self.headers,
    #     )
    #
    # @task
    # def get_vector_store_file_content(self):
    #     _ = self.client.get(
    #         f"vector_stores/{self.vector_store_id}/files/{self.vector_store_file_id}/content",
    #         headers=self.headers,
    #     )
    #
    # # Vector Store File Batches tasks
    # @task
    # def create_vector_store_file_batch(self):
    #     response = self.client.post(
    #         f"vector_stores/{self.vector_store_id}/file_batches",
    #         json={
    #             "file_ids": [self._test_file_id],
    #         },
    #         headers=self.headers,
    #     )
    #     if response.status_code == HTTPStatus.OK:
    #         self.file_batch_id = response.json()["id"]
    #
    # @task
    # def get_vector_store_file_batch(self):
    #     _ = self.client.get(
    #         f"vector_stores/{self.vector_store_id}/file_batches/{self.file_batch_id}",
    #         headers=self.headers,
    #     )
    #
    # @task
    # def cancel_vector_store_file_batch(self):
    #     _ = self.client.post(
    #         f"vector_stores/{self.vector_store_id}/file_batches/{self.file_batch_id}/cancel",
    #         headers=self.headers,
    #     )
    #
    # @task
    # def list_vector_store_file_batch_files(self):
    #     _ = self.client.get(
    #         f"vector_stores/{self.vector_store_id}/file_batches/{self.file_batch_id}/files",
    #         headers=self.headers,
    #     )
