import asyncio
import logging
from typing import Optional

import openai.types
import openai.types.vector_stores
from django.db import models
from django.utils import timezone
from openai import AsyncOpenAI

from management.models.files import FileObject
from management.models.tokens import Token

log = logging.getLogger("aqueduct")


class VectorStoreStatus(models.TextChoices):
    """Vector store status choices matching OpenAI's API."""

    EXPIRED = "expired", "expired"
    IN_PROGRESS = "in_progress", "in_progress"
    COMPLETED = "completed", "completed"


class VectorStore(models.Model):
    """
    Mirrors the structure of OpenAI's VectorStore type.
    Stores metadata for vector stores with upstream relay.
    """

    id = models.CharField(
        max_length=100, primary_key=True, editable=False, help_text="The vector store identifier."
    )
    token = models.ForeignKey(Token, on_delete=models.CASCADE, related_name="vector_stores")
    name = models.CharField(max_length=255, help_text="The name of the vector store.")
    expires_after = models.JSONField(
        null=True, blank=True, help_text="Expiration policy for the vector store."
    )
    chunking_strategy = models.JSONField(
        null=True, blank=True, help_text="Chunking configuration for the vector store."
    )
    metadata = models.JSONField(
        null=True, blank=True, help_text="Custom metadata for the vector store."
    )
    status = models.CharField(
        max_length=20,
        choices=VectorStoreStatus.choices,
        default=VectorStoreStatus.IN_PROGRESS,
        help_text="The current status of the vector store.",
    )
    usage_bytes = models.BigIntegerField(
        default=0, help_text="The total number of bytes used by the vector store."
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the vector store was created."
    )
    last_active_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the vector store was last active.",
    )
    upstream_url = models.URLField(
        blank=True, default="", help_text="The upstream API URL this vector store was created on"
    )

    class Meta:
        verbose_name = "Vector Store"
        verbose_name_plural = "Vector Stores"

    def __str__(self) -> str:
        return self.id

    async def adelete_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> bool:
        """
        Delete the vector store from the upstream API.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns False.

        Returns:
            True on success.
            False on failure (only if raise_on_error=False).
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            await client.vector_stores.delete(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to delete vector store %s from upstream: %s", self.id, e)
            return False
        else:
            return True

    async def areload_from_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> openai.types.VectorStore | None:
        """
        Fetch current state from upstream and update local DB fields.

        Args:
            client: Optional AsyncOpenAI client instance to reuse.
            raise_on_error: If True, raises on failure; if False, logs and returns None.

        Returns:
            The upstream response object on success, or None on failure.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote = await client.vector_stores.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to reload vector store %s from upstream: %s", self.id, e)
            return None
        else:
            self.status = remote.status or self.status
            self.usage_bytes = getattr(remote, "usage_bytes", self.usage_bytes)
            self.last_active_at = int(timezone.now().timestamp())
            await self.asave()
            await self.async_file_statuses(client)
            return remote

    async def async_file_statuses(self, client: AsyncOpenAI | None = None) -> tuple[int, int]:
        """
        Sync all VectorStoreFile statuses from upstream by listing files.
        """
        from gateway.config import get_files_api_client

        if client is None:
            client = get_files_api_client()

        try:
            remote_files_response = await client.vector_stores.files.list(vector_store_id=self.id)
            remote_files = (
                remote_files_response.data if hasattr(remote_files_response, "data") else []
            )
        except Exception as e:
            log.warning("Failed to list files for vector store %s from upstream: %s", self.id, e)
            return 0, 0

        success = 0
        failed = 0

        for remote_file in remote_files:
            try:
                local_file = await VectorStoreFile.objects.aget(
                    id=remote_file.id, vector_store=self
                )
                local_file.status = remote_file.status or local_file.status
                local_file.usage_bytes = remote_file.usage_bytes
                if hasattr(remote_file, "last_error") and remote_file.last_error:
                    local_file.last_error = remote_file.last_error
                await local_file.asave()
                success += 1
            except VectorStoreFile.DoesNotExist:
                log.debug("VectorStoreFile %s not found locally, skipping sync", remote_file.id)
            except Exception as e:
                log.warning("Failed to sync VectorStoreFile %s: %s", remote_file.id, e)
                failed += 1

        return success, failed

    def delete(
        self, using: str | None = None, keep_parents: bool = False, delete_upstream: bool = False
    ) -> tuple[int, dict[str, int]]:
        """
        Override ORM delete - vector stores are stored upstream,
        so we only delete the local DB record.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            asyncio.run(self.adelete_upstream())
        return super().delete(using=using, keep_parents=keep_parents)


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
