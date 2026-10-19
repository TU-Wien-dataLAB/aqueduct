import logging
from typing import Optional

import openai.types.vector_stores
from django.db import models
from openai import AsyncOpenAI

from management.models.vector_store import VectorStore

log = logging.getLogger("aqueduct")


class VectorStoreFileBatchStatus(models.TextChoices):
    """Vector store file batch status choices."""

    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"
    FAILED = "failed", "failed"
    CANCELLED = "cancelled", "cancelled"


class VectorStoreFileBatch(models.Model):
    """
    Represents a batch operation for adding files to a vector store.
    """

    id = models.CharField(
        max_length=100,
        primary_key=True,
        editable=False,
        help_text="The vector store file batch identifier.",
    )
    vector_store = models.ForeignKey(
        VectorStore, on_delete=models.CASCADE, related_name="file_batches"
    )
    file_counts = models.JSONField(
        default=dict, help_text="Processing counts (completed/failed/total/cancelled/in_progress)."
    )
    status = models.CharField(
        max_length=20,
        choices=VectorStoreFileBatchStatus.choices,
        default=VectorStoreFileBatchStatus.IN_PROGRESS,
        help_text="The current status of the batch.",
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the batch was created."
    )

    class Meta:
        verbose_name = "Vector Store File Batch"
        verbose_name_plural = "Vector Store File Batches"

    def __str__(self) -> str:
        return self.id

    async def areload_from_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> Optional["openai.types.vector_stores.VectorStoreFileBatch"]:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        # Ensure vector_store is loaded
        if not self.vector_store_id:
            if raise_on_error:
                raise ValueError("Vector store not set")
            log.warning("Cannot reload vector store file batch %s: vector_store not set", self.id)
            return None

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.file_batches.retrieve(
                vector_store_id=self.vector_store_id, batch_id=self.id
            )
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to reload vector store file batch %s from upstream: %s", self.id, e)
            return None
        else:
            self.status = remote.status or self.status
            if hasattr(remote, "file_counts") and remote.file_counts:
                self.file_counts = remote.file_counts.model_dump()
            await self.asave()
            return remote
