from contextlib import suppress
from typing import TypedDict

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db import transaction
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from openai.types.vector_stores.file_create_params import FileCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import FileObject, Token, VectorStore, VectorStoreFile

from .decorators import catch_router_exceptions, log_request, parse_body, token_authenticated, tos_accepted
from .errors import error_response


class FileUpdateBody(TypedDict, total=False):
    """Request body for updating a vector store file."""

    attributes: dict


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FileCreateParams))
@log_request
@catch_router_exceptions
async def vector_store_files(
    request: ASGIRequest, token: Token, vector_store_id: str, pydantic_model: dict | None = None, *args, **kwargs
) -> JsonResponse:
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
        # List files in vector store - return upstream data directly
        remote_files_response = await client.vector_stores.files.list(vector_store_id=vs_obj.id)
        remote_files = remote_files_response.data if hasattr(remote_files_response, "data") else []

        # Return upstream data directly (IDs already match)
        response_files = []
        for remote_file in remote_files:
            file_data = remote_file.model_dump(mode="json")
            response_files.append(file_data)

        return JsonResponse({"object": "list", "data": response_files, "has_more": False}, status=200)

    # POST /v1/vector_stores/{vector_store_id}/files - Add file to vector store
    params = pydantic_model if pydantic_model else {}
    file_id = params.get("file_id")
    if not file_id:
        return error_response("Missing required parameter: file_id", param="file_id", status=400)

    # Lookup FileObject by Aqueduct ID (outside transaction - no lock needed)
    try:
        if token.service_account:
            file_obj = await FileObject.objects.aget(
                id=file_id, token__service_account__team=token.service_account.team
            )
        else:
            file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return error_response("File not found.", param="file_id", status=404)

    max_files = settings.MAX_VECTOR_STORE_FILES

    # Create upstream file first, then atomically check the limit and insert locally.
    # If a concurrent request filled the last slot, we clean up the upstream file.
    create_kwargs = {"vector_store_id": vs_obj.id, "file_id": file_obj.id}
    if params.get("chunking_strategy"):
        create_kwargs["chunking_strategy"] = params["chunking_strategy"]
    if params.get("attributes"):
        create_kwargs["attributes"] = params["attributes"]
    remote_vs_file = await client.vector_stores.files.create(**create_kwargs)

    # Atomically check the file limit and create the local record.
    # If a concurrent request filled the last slot while the upstream call was in flight,
    # we reject and clean up the upstream file below.
    @sync_to_async
    def create_local_file_with_recheck() -> tuple[VectorStoreFile | None, str]:
        with transaction.atomic():
            # Re-acquire lock and re-check limit
            try:
                if token.service_account:
                    vs_locked = VectorStore.objects.select_for_update().get(
                        id=vector_store_id, token__service_account__team=token.service_account.team
                    )
                else:
                    vs_locked = VectorStore.objects.select_for_update().get(id=vector_store_id, token__user=token.user)
            except VectorStore.DoesNotExist:
                return None, "not_found"

            current_count = VectorStoreFile.objects.filter(vector_store=vs_locked).count()
            if current_count >= max_files:
                return None, "limit_reached"

            now = timezone.now()
            vs_file_obj = VectorStoreFile(
                id=remote_vs_file.id,
                vector_store=vs_locked,
                file_obj=file_obj,
                status=remote_vs_file.status or "in_progress",
                usage_bytes=remote_vs_file.usage_bytes,
                created_at=int(now.timestamp()),
            )
            vs_file_obj.save()
            return vs_file_obj, "ok"

    _result, recheck_status = await create_local_file_with_recheck()

    if recheck_status == "not_found":
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    if recheck_status == "limit_reached":
        # Upstream file was already created but limit was exceeded by a concurrent request.
        # Clean up the upstream file.
        with suppress(Exception):
            await client.vector_stores.files.delete(vector_store_id=vs_obj.id, file_id=remote_vs_file.id)
        return error_response(f"Vector store file limit reached ({max_files})", status=403)

    # Return upstream response directly (IDs already match)
    response_data = remote_vs_file.model_dump(mode="json")

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FileUpdateBody))
@log_request
@catch_router_exceptions
async def vector_store_file(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    file_id: str,
    pydantic_model: dict | None = None,
    *args,
    **kwargs,
) -> JsonResponse:
    """
    GET /v1/vector_stores/{vector_store_id}/files/{file_id} - Retrieve file
    POST /v1/vector_stores/{vector_store_id}/files/{file_id} - Update file attributes
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
        vs_file_obj = await VectorStoreFile.objects.select_related("file_obj", "vector_store").aget(
            id=file_id, vector_store=vs_obj
        )
    except VectorStoreFile.DoesNotExist:
        return error_response("Vector store file not found.", param="file_id", status=404)

    if request.method == "GET":
        # Retrieve from upstream and sync status
        remote_vs_file = await vs_file_obj.areload_from_upstream(client)

        # Return upstream response directly (IDs already match)
        response_data = remote_vs_file.model_dump(mode="json")

        return JsonResponse(response_data, status=200)

    if request.method == "POST":
        # Update file attributes
        params = pydantic_model if pydantic_model else {}

        update_kwargs = {"vector_store_id": vs_obj.id, "file_id": vs_file_obj.id}
        if params.get("attributes"):
            update_kwargs["attributes"] = params["attributes"]

        if not params.get("attributes"):
            return error_response("Missing required parameter: attributes", param="attributes", status=400)

        remote_vs_file = await client.vector_stores.files.update(**update_kwargs)

        # Return upstream response directly (IDs already match)
        response_data = remote_vs_file.model_dump(mode="json")

        return JsonResponse(response_data, status=200)

    await vs_file_obj.adelete_upstream(client)

    # Delete local record
    await sync_to_async(vs_file_obj.delete)()

    # Return response with upstream ID
    response_data = {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
@catch_router_exceptions
async def vector_store_file_content(
    request: ASGIRequest, token: Token, vector_store_id: str, file_id: str, *args, **kwargs
) -> JsonResponse:
    """
    GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content - Get file content
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
        vs_file_obj = await VectorStoreFile.objects.select_related("file_obj").aget(id=file_id, vector_store=vs_obj)
    except VectorStoreFile.DoesNotExist:
        return error_response("Vector store file not found.", param="file_id", status=404)

    # Get content from upstream
    content_response = await client.vector_stores.files.content(vector_store_id=vs_obj.id, file_id=vs_file_obj.id)

    # FileContentResponse and AsyncPage[FileContentResponse] are both Pydantic models
    response_data = content_response.model_dump()

    return JsonResponse(response_data)
