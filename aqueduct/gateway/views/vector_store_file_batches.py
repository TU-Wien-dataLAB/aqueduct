import json

from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from gateway.config import get_files_api_client
from management.models import FileObject, Token, VectorStore, VectorStoreFile, VectorStoreFileBatch

from .decorators import log_request, token_authenticated, tos_accepted
from .errors import error_response


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file_batches(
    request: ASGIRequest, token: Token, vector_store_id: str, *args, **kwargs
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

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body", status=400)

    file_ids = body.get("file_ids", [])
    if not file_ids:
        return error_response("Missing required parameter: file_ids", param="file_ids", status=400)

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
        remote_batch = await client.vector_stores.file_batches.create(
            vector_store_id=vs_obj.remote_id, file_ids=remote_file_ids
        )
    except Exception as e:
        return error_response(
            f"Failed to create file batch on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local batch record
    now = timezone.now()
    file_counts = {}
    if hasattr(remote_batch, "file_counts") and remote_batch.file_counts:
        file_counts = {
            "total": getattr(remote_batch.file_counts, "total", len(file_ids)),
            "completed": getattr(remote_batch.file_counts, "completed", 0),
            "failed": getattr(remote_batch.file_counts, "failed", 0),
            "in_progress": getattr(remote_batch.file_counts, "in_progress", len(file_ids)),
            "cancelled": getattr(remote_batch.file_counts, "cancelled", 0),
        }
    else:
        file_counts = {
            "total": len(file_ids),
            "completed": 0,
            "failed": 0,
            "in_progress": len(file_ids),
            "cancelled": 0,
        }

    batch_obj = VectorStoreFileBatch(
        vector_store=vs_obj,
        remote_id=remote_batch.id,
        status=remote_batch.status or "in_progress",
        file_counts=file_counts,
        created_at=int(now.timestamp()),
    )
    await sync_to_async(batch_obj.save)()

    # Create VectorStoreFile records for each file in the batch
    for file_obj in file_objs:
        vs_file_obj = VectorStoreFile(
            vector_store=vs_obj,
            file_obj=file_obj,
            remote_id=None,  # Will be set when batch processes
            status="in_progress",
            usage_bytes=0,
            created_at=int(now.timestamp()),
        )
        await sync_to_async(vs_file_obj.save)()

    # Return with Aqueduct ID
    response_data = {
        "id": batch_obj.id,
        "object": "vector_store.file_batch",
        "vector_store_id": vs_obj.id,
        "status": batch_obj.status,
        "file_counts": batch_obj.file_counts,
        "created_at": batch_obj.created_at,
    }
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
        batch_obj.file_counts = {
            "total": getattr(remote_batch.file_counts, "total", 0),
            "completed": getattr(remote_batch.file_counts, "completed", 0),
            "failed": getattr(remote_batch.file_counts, "failed", 0),
            "in_progress": getattr(remote_batch.file_counts, "in_progress", 0),
            "cancelled": getattr(remote_batch.file_counts, "cancelled", 0),
        }
    await sync_to_async(batch_obj.save)()

    # Return with Aqueduct ID
    response_data = {
        "id": batch_obj.id,
        "object": "vector_store.file_batch",
        "vector_store_id": vs_obj.id,
        "status": batch_obj.status,
        "file_counts": batch_obj.file_counts,
        "created_at": batch_obj.created_at,
    }
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
        batch_obj.file_counts = {
            "total": getattr(
                remote_batch.file_counts, "total", batch_obj.file_counts.get("total", 0)
            ),
            "completed": getattr(remote_batch.file_counts, "completed", 0),
            "failed": getattr(remote_batch.file_counts, "failed", 0),
            "in_progress": getattr(remote_batch.file_counts, "in_progress", 0),
            "cancelled": getattr(remote_batch.file_counts, "cancelled", 0),
        }
    await sync_to_async(batch_obj.save)()

    # Return with Aqueduct ID
    response_data = {
        "id": batch_obj.id,
        "object": "vector_store.file_batch",
        "vector_store_id": vs_obj.id,
        "status": batch_obj.status,
        "file_counts": batch_obj.file_counts,
        "created_at": batch_obj.created_at,
    }
    return JsonResponse(response_data, status=200)
