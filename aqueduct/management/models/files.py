import asyncio
import logging
import secrets
from typing import ClassVar, Literal, cast

import openai.types
from django.db import models
from openai import AsyncOpenAI

from management.models.accounts import Token

log = logging.getLogger("aqueduct")


def generate_file_id() -> str:
    return f"file-{secrets.token_hex(12)}"


class FileObject(models.Model):
    """
    Mirrors the structure of OpenAI's FileObject type, excluding deprecated fields.
    """

    id = models.CharField(
        max_length=100,
        primary_key=True,
        editable=False,
        help_text="The file identifier, which can be referenced in the API endpoints.",
    )
    bytes = models.BigIntegerField(help_text="The size of the file, in bytes.")
    created_at = models.PositiveIntegerField(
        help_text="The Unix timestamp (in seconds) for when the file was created."
    )
    filename = models.CharField(max_length=255, help_text="The name of the file.")
    PURPOSE_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("assistants", "assistants"),
        ("assistants_output", "assistants_output"),
        ("batch", "batch"),
        ("batch_output", "batch_output"),
        ("fine-tune", "fine-tune"),
        ("vision", "vision"),
        ("user_data", "user_data"),
        ("evals", "evals"),
    ]
    purpose = models.CharField(
        max_length=20, choices=PURPOSE_CHOICES, help_text="The intended purpose of the file."
    )
    expires_at = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="The Unix timestamp (in seconds) for when the file will expire.",
    )

    token = models.ForeignKey(
        Token,
        on_delete=models.CASCADE,  # If Token is deleted, delete its associated Files
        related_name="files",
    )

    preview = models.TextField(
        blank=True, default="", help_text="Preview of file content (first 10 lines for JSONL files)"
    )

    upstream_url = models.URLField(
        blank=True, default="", help_text="The upstream API URL this file was uploaded to"
    )

    class Meta:
        verbose_name = "File Object"
        verbose_name_plural = "File Objects"

    def __str__(self) -> str:
        return self.id

    @property
    def model(self) -> openai.types.FileObject:
        return openai.types.FileObject(
            id=self.id,
            bytes=self.bytes,
            created_at=self.created_at,
            filename=self.filename,
            purpose=cast(
                (
                    "Literal['assistants', 'assistants_output', 'batch', 'batch_output',"
                    " 'fine-tune', 'fine-tune-results', 'vision', 'user_data']"
                ),
                self.purpose,
            ),
            expires_at=self.expires_at,
            object="file",
            status="processed",
        )

    async def adelete_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> bool:
        """
        Delete the file from the upstream API.

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
            await client.files.delete(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to delete file %s from upstream: %s", self.id, e)
            return False
        else:
            return True

    async def areload_from_upstream(
        self, client: AsyncOpenAI | None = None, raise_on_error: bool = True
    ) -> openai.types.FileObject | None:
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
            remote = await client.files.retrieve(self.id)
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Failed to reload file %s from upstream: %s", self.id, e)
            return None
        else:
            self.purpose = remote.purpose
            self.expires_at = remote.expires_at
            await self.asave()
            return remote

    def delete(
        self, using: str | None = None, keep_parents: bool = False, delete_upstream: bool = False
    ) -> tuple[int, dict[str, int]]:
        """
        Override ORM delete - files are stored upstream, so we only delete the local DB record.

        Args:
            delete_upstream: If True, also delete from upstream API before local delete.
        """
        if delete_upstream:
            asyncio.run(self.adelete_upstream())
        return super().delete(using=using, keep_parents=keep_parents)
