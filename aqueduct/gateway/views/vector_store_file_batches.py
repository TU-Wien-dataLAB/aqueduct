from typing import Optional

from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest
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
        await sync_to_async(vs_file_obj.save)()

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
    vs_files = await sync_to_async(list)(
        VectorStoreFile.objects.filter(vector_store=vs_obj).select_related("file_obj")
    )
    remote_to_local_map = {f.remote_id: f for f in vs_files if f.remote_id}

    # Map remote file IDs to local IDs in response
    remote_files = remote_files_response.data if hasattr(remote_files_response, "data") else []
    response_files = []
    for remote_file in remote_files:
        file_data = remote_file.model_dump(mode="json")
        # Try to map to local ID if we have this file tracked
        local_file = remote_to_local_map.get(remote_file.id)
        if local_file:
            file_data["id"] = local_file.id
        file_data["vector_store_id"] = vs_obj.id
        response_files.append(file_data)

    return JsonResponse({"object": "list", "data": response_files, "has_more": False}, status=200)
