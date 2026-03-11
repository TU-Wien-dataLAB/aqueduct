import logging
from typing import Optional

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db import transaction
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from openai.types.vector_stores.file_batch_create_params import FileBatchCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import (
    FileObject,
    Token,
    VectorStore,
    VectorStoreFile,
    VectorStoreFileBatch,
    VectorStoreFileBatchStatus,
    VectorStoreFileStatus,
)

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
from .errors import error_response

logger = logging.getLogger(__name__)


async def mark_orphaned_files(
    batch_obj: VectorStoreFileBatch, status: VectorStoreFileStatus, error_msg: str
):
    """Mark in-progress files in a batch as failed when the batch fails or is cancelled.

    Args:
        batch_obj: The VectorStoreFileBatch whose orphaned files should be marked.
        status: The new status to set on orphaned files (typically "failed").
        error_msg: The error message to set on orphaned files.
    """
    orphaned_files = await sync_to_async(list)(
        VectorStoreFile.objects.filter(batch=batch_obj, status=VectorStoreFileStatus.IN_PROGRESS)
    )
    if orphaned_files:

        @sync_to_async
        def _bulk_update_orphans():
            for vs_file in orphaned_files:
                vs_file.status = status
                vs_file.last_error = error_msg
            VectorStoreFile.objects.bulk_update(orphaned_files, ["status", "last_error"])

        await _bulk_update_orphans()


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FileBatchCreateParams))
@log_request
async def vector_store_file_batches(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    pydantic_model: Optional[dict] = None,
    *args,
    **kwargs,
):
    """
    POST /v1/vector_stores/{vector_store_id}/file_batches - Create file batch
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    # Check vector store ownership
    try:
        if token.service_account:
            vs_obj = await VectorStore.objects.aget(
                id=vector_store_id, token__service_account__team=token.service_account.team
            )
        else:
            vs_obj = await VectorStore.objects.aget(id=vector_store_id, token__user=token.user)
    except VectorStore.DoesNotExist:
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    params = pydantic_model if pydantic_model else {}
    file_ids = params.get("file_ids", [])
    files = params.get("files", [])

    if not file_ids and not files:
        return error_response(
            "Missing required parameter: file_ids or files", param="file_ids", status=400
        )

    # Handle both file_ids and files formats
    # Extract file_ids for lookup; pass full files dicts to upstream to preserve attributes and chunking_strategy
    if files:
        file_ids = [f["file_id"] for f in files]

    # Lookup all FileObjects by Aqueduct IDs
    file_objs = []
    for file_id in file_ids:
        try:
            if token.service_account:
                file_obj = await FileObject.objects.aget(
                    id=file_id, token__service_account__team=token.service_account.team
                )
            else:
                file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
        except FileObject.DoesNotExist:
            return error_response(f"File not found: {file_id}", param="file_ids", status=404)
        file_objs.append(file_obj)

    # Create batch on upstream
    create_kwargs = {"vector_store_id": vs_obj.id}
    if files:
        # Pass full files dicts with attributes and chunking_strategy
        create_kwargs["files"] = files
    else:
        create_kwargs["file_ids"] = file_ids
    if params.get("chunking_strategy"):
        create_kwargs["chunking_strategy"] = params["chunking_strategy"]
    if params.get("attributes"):
        create_kwargs["attributes"] = params["attributes"]
    try:
        remote_batch = await client.vector_stores.file_batches.create(**create_kwargs)
    except Exception as e:
        return error_response(
            f"Failed to create file batch on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local batch record with upstream ID
    now = timezone.now()
    batch_obj = VectorStoreFileBatch(
        id=remote_batch.id,
        vector_store=vs_obj,
        status=remote_batch.status or VectorStoreFileBatchStatus.IN_PROGRESS,
        file_counts=(
            remote_batch.file_counts.model_dump(mode="json")
            if hasattr(remote_batch, "file_counts") and remote_batch.file_counts
            else {
                "total": len(file_ids),
                "completed": 0,
                "failed": 0,
                "in_progress": len(file_ids),
                "cancelled": 0,
            }
        ),
        created_at=int(now.timestamp()),
    )
    await sync_to_async(batch_obj.save)()

    # Create VectorStoreFile records for each file in the batch
    # Use transaction and check file limit
    @sync_to_async
    def create_batch_files():
        with transaction.atomic():
            max_files = settings.MAX_VECTOR_STORE_FILES
            current_count = VectorStoreFile.objects.filter(vector_store=vs_obj).count()
            if current_count + len(file_objs) > max_files:
                return None
            records = [
                VectorStoreFile(
                    id=file_obj.id,
                    vector_store=vs_obj,
                    file_obj=file_obj,
                    batch=batch_obj,
                    status=VectorStoreFileStatus.IN_PROGRESS,
                    usage_bytes=0,
                    created_at=int(now.timestamp()),
                )
                for file_obj in file_objs
            ]
            return VectorStoreFile.objects.bulk_create(records)

    batch_file_records = await create_batch_files()
    if batch_file_records is None:
        return error_response("File limit exceeded", status=403)

    # Return upstream response directly (IDs already match)
    response_data = remote_batch.model_dump(mode="json")

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file_batch(
    request: ASGIRequest, token: Token, vector_store_id: str, batch_id: str, *args, **kwargs
):
    """
    GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id} - Retrieve batch
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    # Check vector store ownership
    try:
        if token.service_account:
            vs_obj = await VectorStore.objects.aget(
                id=vector_store_id, token__service_account__team=token.service_account.team
            )
        else:
            vs_obj = await VectorStore.objects.aget(id=vector_store_id, token__user=token.user)
    except VectorStore.DoesNotExist:
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    # Get the batch
    try:
        batch_obj = await VectorStoreFileBatch.objects.select_related("vector_store").aget(
            id=batch_id, vector_store=vs_obj
        )
    except VectorStoreFileBatch.DoesNotExist:
        return error_response("File batch not found.", param="batch_id", status=404)

    # Retrieve from upstream and sync status
    try:
        remote_batch = await batch_obj.areload_from_upstream(client)
    except Exception as e:
        return error_response(
            f"Failed to retrieve file batch from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Handle orphaned VectorStoreFile records when batch fails or is cancelled
    if batch_obj.status in (
        VectorStoreFileBatchStatus.FAILED,
        VectorStoreFileBatchStatus.CANCELLED,
    ):
        await mark_orphaned_files(
            batch_obj,
            status=VectorStoreFileStatus.FAILED,
            error_msg=f"Batch {batch_obj.status}: files were not processed",
        )

    # Return upstream response directly (IDs already match)
    response_data = remote_batch.model_dump(mode="json")

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file_batch_cancel(
    request: ASGIRequest, token: Token, vector_store_id: str, batch_id: str, *args, **kwargs
):
    """
    POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel - Cancel batch
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    # Check vector store ownership
    try:
        if token.service_account:
            vs_obj = await VectorStore.objects.aget(
                id=vector_store_id, token__service_account__team=token.service_account.team
            )
        else:
            vs_obj = await VectorStore.objects.aget(id=vector_store_id, token__user=token.user)
    except VectorStore.DoesNotExist:
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    # Get the batch
    try:
        batch_obj = await VectorStoreFileBatch.objects.select_related("vector_store").aget(
            id=batch_id, vector_store=vs_obj
        )
    except VectorStoreFileBatch.DoesNotExist:
        return error_response("File batch not found.", param="batch_id", status=404)

    # Cancel on upstream
    try:
        remote_batch = await client.vector_stores.file_batches.cancel(
            vector_store_id=vs_obj.id, batch_id=batch_obj.id
        )
    except Exception as e:
        return error_response(
            f"Failed to cancel file batch on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Update local record
    batch_obj.status = remote_batch.status or VectorStoreFileBatchStatus.CANCELLED
    if hasattr(remote_batch, "file_counts") and remote_batch.file_counts:
        batch_obj.file_counts = remote_batch.file_counts.model_dump(mode="json")
    await sync_to_async(batch_obj.save)()

    # Handle orphaned VectorStoreFile records when batch is cancelled
    await mark_orphaned_files(
        batch_obj,
        status=VectorStoreFileStatus.CANCELLED,
        error_msg="Batch cancelled: files were not processed",
    )

    # Return upstream response directly (IDs already match)
    response_data = remote_batch.model_dump(mode="json")

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file_batch_files(
    request: ASGIRequest, token: Token, vector_store_id: str, batch_id: str, *args, **kwargs
):
    """
    GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files - List files in batch
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    # Check vector store ownership
    try:
        if token.service_account:
            vs_obj = await VectorStore.objects.aget(
                id=vector_store_id, token__service_account__team=token.service_account.team
            )
        else:
            vs_obj = await VectorStore.objects.aget(id=vector_store_id, token__user=token.user)
    except VectorStore.DoesNotExist:
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    # Get the batch
    try:
        batch_obj = await VectorStoreFileBatch.objects.select_related("vector_store").aget(
            id=batch_id, vector_store=vs_obj
        )
    except VectorStoreFileBatch.DoesNotExist:
        return error_response("File batch not found.", param="batch_id", status=404)

    # Get files in batch from upstream
    try:
        remote_files_response = await client.vector_stores.file_batches.list_files(
            vector_store_id=vs_obj.id, batch_id=batch_obj.id
        )
    except Exception as e:
        return error_response(
            f"Failed to retrieve file batch files from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Return upstream response directly (IDs already match)
    remote_files = remote_files_response.data if hasattr(remote_files_response, "data") else []
    response_files = []
    for remote_file in remote_files:
        file_data = remote_file.model_dump(mode="json")
        response_files.append(file_data)

    return JsonResponse({"object": "list", "data": response_files, "has_more": False}, status=200)
