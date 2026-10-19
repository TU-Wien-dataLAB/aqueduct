import asyncio
import logging
from typing import Optional

import openai.types.vector_stores
from django.db import models
from openai import AsyncOpenAI

from management.models.file_object import FileObject
from management.models.vector_store import VectorStore

log = logging.getLogger("aqueduct")


class VectorStoreFileStatus(models.TextChoices):
    """Vector store file processing status choices."""

    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"
    FAILED = "failed", "failed"
    CANCELLED = "cancelled", "cancelled"


class VectorStoreFile(models.Model):
    """
    Represents a file within a vector store.
    This is NOT the same as FileObject - it's a join table that tracks
    a FileObject's association with a specific VectorStore.
    """

    id = models.CharField(
        max_length=100,
        primary_key=True,
        editable=False,
        help_text="The vector store file identifier.",
    )
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="files")
    file_obj = models.ForeignKey(
        FileObject,
        on_delete=models.CASCADE,
        related_name="vector_store_files",
        help_text="Reference to the actual file object.",
    )
    batch = models.ForeignKey(
        "VectorStoreFileBatch",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="files",
        help_text="The batch that created this file, if any.",
    )
    status = models.CharField(
        max_length=20,
        choices=VectorStoreFileStatus.choices,
        default=VectorStoreFileStatus.IN_PROGRESS,
        help_text="Processing status of the file in the vector store.",
    )
    last_error = models.JSONField(
        null=True, blank=True, help_text="Error details if processing failed."
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the file was added to the vector store."
    )
    usage_bytes = models.BigIntegerField(
        default=0,
        help_text="Bytes used in vector store (may differ from original file after chunking).",
    )

    class Meta:
        verbose_name = "Vector Store File"
        verbose_name_plural = "Vector Store Files"

    def __str__(self) -> str:
        return self.id

    async def adelete_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> bool:
        """
        Delete the vector store file from the upstream API.

        Note: Callers should use select_related("vector_store") when fetching
        VectorStoreFile instances to avoid extra queries.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns False.

        Returns:
            True on success.
            False on failure (only if raise_on_error=False).
        """
        if not self.vector_store_id:
            log.warning("Cannot delete vector store file %s: vector_store not set", self.id)
            if raise_on_error:
                raise ValueError("Vector store not loaded")
            return False

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            await client.vector_stores.files.delete(
                vector_store_id=self.vector_store_id, file_id=self.id
            )
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to delete vector store file %s from upstream: %s", self.id, e)
            return False
        else:
            return True

    async def areload_from_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> Optional["openai.types.vector_stores.VectorStoreFile"]:
        """
        Fetch current state from upstream and update local DB fields.

        Note: Callers should use select_related("vector_store") when fetching
        VectorStoreFile instances to avoid extra queries.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        if not self.vector_store_id:
            if raise_on_error:
                raise ValueError("Vector store not set")
            log.warning("Cannot reload vector store file %s: vector_store not set", self.id)
            return None

        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=self.id
            )
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to reload vector store file %s from upstream: %s", self.id, e)
            return None
        else:
            self.status = remote.status or self.status
            self.usage_bytes = remote.usage_bytes
            if hasattr(remote, "last_error") and remote.last_error:
                self.last_error = remote.last_error
            await self.asave()
            return remote

    def delete(
        self, using: str | None = None, keep_parents: bool = False, delete_upstream: bool = False
    ) -> tuple[int, dict[str, int]]:
        """
        Override ORM delete.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            asyncio.run(self.adelete_upstream())
        return super().delete(using=using, keep_parents=keep_parents)
