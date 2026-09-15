# models/batches.py
"""Batch model mirroring OpenAI's Batch type."""

import logging
import secrets
from typing import Literal, cast

import openai.types
import openai.types.batch
from django.db import models
from django.db.models import JSONField
from openai import AsyncOpenAI

from .accounts import Token
from .files import FileObject

log = logging.getLogger("aqueduct")


# Legacy ID generator. Referenced by historical migrations (0005) and must remain
# importable from management.models. No longer used at runtime.
def generate_batch_id() -> str:
    return f"batch-{secrets.token_hex(12)}"


def default_request_counts() -> dict[str, int]:
    return {"total": 0, "completed": 0, "failed": 0}


class BatchStatus(models.TextChoices):
    VALIDATING = "validating", "validating"
    FAILED = "failed", "failed"
    IN_PROGRESS = "in_progress", "in_progress"
    FINALIZING = "finalizing", "finalizing"
    COMPLETED = "completed", "completed"
    EXPIRED = "expired", "expired"
    CANCELLING = "cancelling", "cancelling"
    CANCELLED = "cancelled", "cancelled"


class Batch(models.Model):
    """
    Mirrors the structure of OpenAI's Batch type.
    """

    id = models.CharField(
        max_length=100, primary_key=True, editable=False, help_text="The batch identifier."
    )
    completion_window = models.CharField(
        max_length=100, help_text="The time frame within which the batch should be processed."
    )
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the batch was created."
    )
    endpoint = models.CharField(
        max_length=255, help_text="The OpenAI API endpoint used by the batch."
    )
    input_file = models.ForeignKey(
        FileObject,
        on_delete=models.CASCADE,
        related_name="batches",
        help_text="The input file for the batch.",
    )
    output_file = models.ForeignKey(
        FileObject,
        on_delete=models.SET_NULL,
        related_name="output_batches",
        null=True,
        blank=True,
        help_text="The output file for the batch (set when batch completes).",
    )
    error_file = models.ForeignKey(
        FileObject,
        on_delete=models.SET_NULL,
        related_name="error_batches",
        null=True,
        blank=True,
        help_text="The error file for the batch (set when batch completes with errors).",
    )
    status = models.CharField(
        max_length=20, choices=BatchStatus.choices, help_text="The current status of the batch."
    )
    cancelled_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch was cancelled.",
    )
    cancelling_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch started cancelling.",
    )
    completed_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch was completed.",
    )
    expired_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch expired.",
    )
    expires_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch will expire.",
    )
    failed_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch failed.",
    )
    finalizing_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch started finalizing.",
    )
    in_progress_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the batch started processing.",
    )
    metadata = JSONField(null=True, blank=True, help_text="Metadata attached to the batch.")
    # {"input": 0, "total": 0, "completed": 0, "failed": 0 }  # noqa: ERA001
    request_counts = JSONField(
        default=default_request_counts,
        null=True,
        blank=True,
        help_text="The request counts for different statuses within the batch.",
    )

    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,
        related_name="batches",
        null=True,
        blank=True,
        help_text="The token (API key) that created this batch.",
    )

    class Meta:
        verbose_name = "Batch"
        verbose_name_plural = "Batches"

    def __str__(self) -> str:
        return self.id

    @property
    def model(self) -> openai.types.batch.Batch:
        return openai.types.batch.Batch(
            id=self.id,
            completion_window=self.completion_window,
            created_at=self.created_at,
            endpoint=self.endpoint,
            input_file_id=self.input_file_id,
            object="batch",
            status=cast(
                (
                    "Literal['validating', 'failed', 'in_progress', 'finalizing', 'completed',"
                    " 'expired', 'cancelling', 'cancelled']"
                ),
                self.status,
            ),
            cancelled_at=self.cancelled_at,
            cancelling_at=self.cancelling_at,
            completed_at=self.completed_at,
            error_file_id=self.error_file_id if self.error_file else None,
            errors=None,
            expired_at=self.expired_at,
            expires_at=self.expires_at,
            failed_at=self.failed_at,
            finalizing_at=self.finalizing_at,
            in_progress_at=self.in_progress_at,
            metadata=self.metadata,
            output_file_id=self.output_file_id if self.output_file else None,
            request_counts=openai.types.BatchRequestCounts(
                total=self.request_counts.get("total", 0) if self.request_counts else 0,
                completed=self.request_counts.get("completed", 0) if self.request_counts else 0,
                failed=self.request_counts.get("failed", 0) if self.request_counts else 0,
            ),
        )

    async def areload_from_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> openai.types.Batch | None:
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
            remote = await client.batches.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to reload batch %s from upstream: %s", self.id, e)
            return None
        else:
            self.status = remote.status
            if remote.request_counts:
                self.request_counts = remote.request_counts.model_dump()
            self.completed_at = remote.completed_at
            self.failed_at = remote.failed_at
            self.cancelled_at = remote.cancelled_at
            self.cancelling_at = remote.cancelling_at
            self.expired_at = remote.expired_at
            self.in_progress_at = remote.in_progress_at
            self.finalizing_at = remote.finalizing_at
            await self.asave()
            return remote
