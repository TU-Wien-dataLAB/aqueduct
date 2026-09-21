import asyncio
import logging

import openai.types
from django.db import models
from django.utils import timezone
from openai import AsyncOpenAI

from management.models.token import Token

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

        from management.models.vector_store_file import VectorStoreFile

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
