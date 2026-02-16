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
from management.models import FileObject, Token, VectorStore, VectorStoreFile, VectorStoreFileBatch

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
from .errors import error_response

logger = logging.getLogger(__name__)


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
    if files:
        # files format: list of dicts with file_id, attributes, chunking_strategy
        file_ids = [f["file_id"] for f in files]

    # Lookup all FileObjects by Aqueduct IDs
    file_objs = []
    remote_file_ids = []
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
        if not file_obj.remote_id:
            return error_response(
                "File has no remote reference.", error_type="server_error", status=500
            )
        file_objs.append(file_obj)
        remote_file_ids.append(file_obj.remote_id)

    # Create batch on upstream
    try:
        create_kwargs = {"vector_store_id": vs_obj.remote_id}
        if files:
            # Transform files to include remote file IDs
            create_kwargs["files"] = [
                {"file_id": file_obj.remote_id, **{k: v for k, v in f.items() if k != "file_id"}}
                for f, file_obj in zip(files, file_objs)
            ]
        else:
            create_kwargs["file_ids"] = remote_file_ids
        if params.get("chunking_strategy"):
            create_kwargs["chunking_strategy"] = params["chunking_strategy"]
        if params.get("attributes"):
            create_kwargs["attributes"] = params["attributes"]
        remote_batch = await client.vector_stores.file_batches.create(**create_kwargs)
    except Exception as e:
        return error_response(
            f"Failed to create file batch on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local batch record
    now = timezone.now()
    batch_obj = VectorStoreFileBatch(
        vector_store=vs_obj,
        remote_id=remote_batch.id,
        status=remote_batch.status or "in_progress",
        file_counts=remote_batch.file_counts.model_dump(mode="json")
        if hasattr(remote_batch, "file_counts") and remote_batch.file_counts
        else {
            "total": len(file_ids),
            "completed": 0,
            "failed": 0,
            "in_progress": len(file_ids),
            "cancelled": 0,
        },
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
            records = []
            for file_obj in file_objs:
                vs_file_obj = VectorStoreFile(
                    vector_store=vs_obj,
                    file_obj=file_obj,
                    batch=batch_obj,
                    remote_id=None,
                    status="in_progress",
                    usage_bytes=0,
                    created_at=int(now.timestamp()),
                )
                vs_file_obj.save()
                records.append(vs_file_obj)
            return records

    batch_file_records = await create_batch_files()
    if batch_file_records is None:
        return error_response("File limit exceeded", status=403)

    # Return upstream response with only id and vector_store_id replaced
    response_data = remote_batch.model_dump(mode="json")
    response_data["id"] = batch_obj.id
    response_data["vector_store_id"] = vs_obj.id

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

    if not batch_obj.remote_id:
        return error_response(
            "File batch has no remote reference.", error_type="server_error", status=500
        )

    # Retrieve from upstream and sync status
    try:
        remote_batch = await client.vector_stores.file_batches.retrieve(
            vector_store_id=vs_obj.remote_id, batch_id=batch_obj.remote_id
        )
    except Exception as e:
        return error_response(
            f"Failed to retrieve file batch from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Update local record with latest status
    batch_obj.status = remote_batch.status or batch_obj.status
    if hasattr(remote_batch, "file_counts") and remote_batch.file_counts:
        batch_obj.file_counts = remote_batch.file_counts.model_dump(mode="json")
    await sync_to_async(batch_obj.save)()

    # Handle orphaned VectorStoreFile records when batch fails
    # Find records with remote_id=None that were created for this batch
    if batch_obj.status in ("failed", "cancelled"):
        orphaned_files = await sync_to_async(list)(
            VectorStoreFile.objects.filter(
                batch=batch_obj, remote_id__isnull=True, status="in_progress"
            )
        )
        if orphaned_files:

            @sync_to_async
            def mark_orphans_failed():
                for vs_file in orphaned_files:
                    vs_file.status = "failed"
                    vs_file.last_error = f"Batch {batch_obj.status}: files were not processed"
                    vs_file.save()

            await mark_orphans_failed()

    # Return upstream response with only id and vector_store_id replaced
    response_data = remote_batch.model_dump(mode="json")
    response_data["id"] = batch_obj.id
    response_data["vector_store_id"] = vs_obj.id

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

    if not batch_obj.remote_id:
        return error_response(
            "File batch has no remote reference.", error_type="server_error", status=500
        )

    # Cancel on upstream
    try:
        remote_batch = await client.vector_stores.file_batches.cancel(
            vector_store_id=vs_obj.remote_id, batch_id=batch_obj.remote_id
        )
    except Exception as e:
        return error_response(
            f"Failed to cancel file batch on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Update local record
    batch_obj.status = remote_batch.status or "cancelled"
    if hasattr(remote_batch, "file_counts") and remote_batch.file_counts:
        batch_obj.file_counts = remote_batch.file_counts.model_dump(mode="json")
    await sync_to_async(batch_obj.save)()

    # Handle orphaned VectorStoreFile records when batch is cancelled
    orphaned_files = await sync_to_async(list)(
        VectorStoreFile.objects.filter(
            batch=batch_obj, remote_id__isnull=True, status="in_progress"
        )
    )
    if orphaned_files:

        @sync_to_async
        def mark_orphans_cancelled():
            for vs_file in orphaned_files:
                vs_file.status = "failed"
                vs_file.last_error = "Batch cancelled: files were not processed"
                vs_file.save()

        await mark_orphans_cancelled()

    # Return upstream response with only id and vector_store_id replaced
    response_data = remote_batch.model_dump(mode="json")
    response_data["id"] = batch_obj.id
    response_data["vector_store_id"] = vs_obj.id

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

    if not batch_obj.remote_id:
        return error_response(
            "File batch has no remote reference.", error_type="server_error", status=500
        )

    # Get files in batch from upstream
    try:
        remote_files_response = await client.vector_stores.file_batches.list_files(
            vector_store_id=vs_obj.remote_id, batch_id=batch_obj.remote_id
        )
    except Exception as e:
        return error_response(
            f"Failed to retrieve file batch files from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Get all local vector store files to map remote IDs to local IDs
    all_vs_files = await sync_to_async(list)(
        VectorStoreFile.objects.filter(vector_store=vs_obj).select_related("file_obj")
    )
    remote_to_local_map = {f.remote_id: f for f in all_vs_files if f.remote_id}

    # Build map of batch-created files (remote_id=None) by file_obj.remote_id
    # Only include files from the current batch to prevent cross-batch linkage
    batch_vs_files = [f for f in all_vs_files if f.batch_id == batch_obj.id]
    batch_file_by_file_remote_id = {}
    for f in batch_vs_files:
        if f.remote_id is None and f.file_obj and f.file_obj.remote_id:
            batch_file_by_file_remote_id[f.file_obj.remote_id] = f

    # Map remote file IDs to local IDs in response
    remote_files = remote_files_response.data if hasattr(remote_files_response, "data") else []
    response_files = []
    files_to_link = []
    for remote_file in remote_files:
        file_data = remote_file.model_dump(mode="json")
        local_file = remote_to_local_map.get(remote_file.id)
        if local_file:
            file_data["id"] = local_file.id
            file_data["vector_store_id"] = vs_obj.id
            file_data["file_id"] = local_file.file_obj.id if local_file.file_obj else None
            response_files.append(file_data)
        else:
            batch_file = batch_file_by_file_remote_id.get(remote_file.id)
            if batch_file:
                files_to_link.append((batch_file, remote_file))
                file_data["id"] = batch_file.id
                file_data["vector_store_id"] = vs_obj.id
                file_data["file_id"] = batch_file.file_obj.id if batch_file.file_obj else None
                response_files.append(file_data)
            else:
                logger.warning(f"Batch file {remote_file.id} has no local record, skipping")

    # Update batch-created records with their upstream remote_id
    if files_to_link:

        @sync_to_async
        def link_batch_files():
            for local_file, remote_file in files_to_link:
                local_file.remote_id = remote_file.id
                local_file.status = remote_file.status or local_file.status
                local_file.usage_bytes = remote_file.usage_bytes
                if hasattr(remote_file, "last_error") and remote_file.last_error:
                    local_file.last_error = remote_file.last_error
                local_file.save()

        await link_batch_files()

    return JsonResponse({"object": "list", "data": response_files, "has_more": False}, status=200)
