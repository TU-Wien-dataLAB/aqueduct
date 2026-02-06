from typing import Optional

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai.types.vector_stores.file_create_params import FileCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import FileObject, Token, VectorStore, VectorStoreFile

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
from .errors import error_response


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FileCreateParams))
@log_request
async def vector_store_files(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    pydantic_model: Optional[dict] = None,
    *args,
    **kwargs,
):
    """
    GET /v1/vector_stores/{vector_store_id}/files - List files in vector store
    POST /v1/vector_stores/{vector_store_id}/files - Add file to vector store
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

    if request.method == "GET":
        # List files in vector store from local DB
        files_list = await sync_to_async(list)(
            VectorStoreFile.objects.filter(vector_store=vs_obj)
            .order_by("-created_at")
            .select_related("vector_store", "file_obj")
        )

        return JsonResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": vf.id,
                        "object": "vector_store.file",
                        "vector_store_id": vs_obj.id,
                        "file_id": vf.file_obj.id if vf.file_obj else None,
                        "status": vf.status,
                        "usage_bytes": vf.usage_bytes,
                        "created_at": vf.created_at,
                        "last_error": vf.last_error,
                    }
                    for vf in files_list
                ],
                "has_more": False,
            },
            status=200,
        )

    # POST /v1/vector_stores/{vector_store_id}/files - Add file to vector store
    params = pydantic_model if pydantic_model else {}
    file_id = params.get("file_id")
    if not file_id:
        return error_response("Missing required parameter: file_id", param="file_id", status=400)

    # Check file limit before adding
    max_files = getattr(settings, "MAX_VECTOR_STORE_FILES", 1000)
    current_count = await sync_to_async(VectorStoreFile.objects.filter(vector_store=vs_obj).count)()
    if current_count >= max_files:
        return error_response(f"Vector store file limit reached ({max_files})", status=403)

    # Lookup FileObject by Aqueduct ID
    try:
        if token.service_account:
            file_obj = await FileObject.objects.aget(
                id=file_id, token__service_account__team=token.service_account.team
            )
        else:
            file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return error_response("File not found.", param="file_id", status=404)

    if not file_obj.remote_id:
        return error_response(
            "File has no remote reference.", error_type="server_error", status=500
        )

    # Add file to vector store on upstream using remote file ID
    try:
        create_kwargs = {"vector_store_id": vs_obj.remote_id, "file_id": file_obj.remote_id}
        if params.get("chunking_strategy"):
            create_kwargs["chunking_strategy"] = params["chunking_strategy"]
        if params.get("attributes"):
            create_kwargs["attributes"] = params["attributes"]
        remote_vs_file = await client.vector_stores.files.create(**create_kwargs)
    except Exception as e:
        return error_response(
            f"Failed to add file to vector store on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local record
    now = timezone.now()
    vs_file_obj = VectorStoreFile(
        vector_store=vs_obj,
        file_obj=file_obj,
        remote_id=remote_vs_file.id,
        status=remote_vs_file.status or "in_progress",
        usage_bytes=getattr(remote_vs_file, "usage_bytes", 0),
        created_at=int(now.timestamp()),
    )
    await sync_to_async(vs_file_obj.save)()

    # Return with Aqueduct ID
    response_data = {
        "id": vs_file_obj.id,
        "object": "vector_store.file",
        "vector_store_id": vs_obj.id,
        "file_id": file_obj.id,
        "status": vs_file_obj.status,
        "usage_bytes": vs_file_obj.usage_bytes,
        "created_at": vs_file_obj.created_at,
        "last_error": vs_file_obj.last_error,
    }
    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file(
    request: ASGIRequest, token: Token, vector_store_id: str, file_id: str, *args, **kwargs
):
    """
    GET /v1/vector_stores/{vector_store_id}/files/{file_id} - Retrieve file
    DELETE /v1/vector_stores/{vector_store_id}/files/{file_id} - Delete file
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

    # Get the vector store file
    try:
        vs_file_obj = await VectorStoreFile.objects.select_related("file_obj").aget(
            id=file_id, vector_store=vs_obj
        )
    except VectorStoreFile.DoesNotExist:
        return error_response("Vector store file not found.", param="file_id", status=404)

    if not vs_file_obj.remote_id:
        return error_response(
            "Vector store file has no remote reference.", error_type="server_error", status=500
        )

    if request.method == "GET":
        # Retrieve from upstream and sync status
        try:
            remote_vs_file = await client.vector_stores.files.retrieve(
                vector_store_id=vs_obj.remote_id, file_id=vs_file_obj.remote_id
            )
        except Exception as e:
            return error_response(
                f"Failed to retrieve vector store file from upstream: {str(e)}",
                error_type="server_error",
                status=502,
            )

        # Update local record with latest status
        vs_file_obj.status = remote_vs_file.status or vs_file_obj.status
        vs_file_obj.usage_bytes = getattr(remote_vs_file, "usage_bytes", vs_file_obj.usage_bytes)
        if hasattr(remote_vs_file, "last_error") and remote_vs_file.last_error:
            vs_file_obj.last_error = remote_vs_file.last_error
        await sync_to_async(vs_file_obj.save)()

        # Return with Aqueduct ID
        response_data = {
            "id": vs_file_obj.id,
            "object": "vector_store.file",
            "vector_store_id": vs_obj.id,
            "file_id": vs_file_obj.file_obj.id if vs_file_obj.file_obj else None,
            "status": vs_file_obj.status,
            "usage_bytes": vs_file_obj.usage_bytes,
            "created_at": vs_file_obj.created_at,
            "last_error": vs_file_obj.last_error,
        }
        return JsonResponse(response_data, status=200)

    # DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}
    try:
        await client.vector_stores.files.delete(
            vector_store_id=vs_obj.remote_id, file_id=vs_file_obj.remote_id
        )
    except Exception as e:
        return error_response(
            f"Failed to delete vector store file from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Delete local record
    await sync_to_async(vs_file_obj.delete)()

    # Return with Aqueduct ID
    return JsonResponse(
        {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}, status=200
    )
