from typing import Optional, TypedDict

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from openai.types.vector_stores.file_create_params import FileCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import FileObject, Token, VectorStore, VectorStoreFile

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
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
        # List files in vector store - always refresh from upstream
        try:
            remote_files_response = await client.vector_stores.files.list(
                vector_store_id=vs_obj.remote_id
            )
            remote_files = (
                remote_files_response.data if hasattr(remote_files_response, "data") else []
            )
        except Exception:
            return error_response(
                "Failed to retrieve files from upstream.", error_type="server_error", status=502
            )

        # Get existing local VectorStoreFile records
        existing_local_files = await sync_to_async(list)(
            VectorStoreFile.objects.filter(vector_store=vs_obj).select_related("file_obj")
        )

        # Build map by remote_id for quick lookup
        remote_id_to_local = {f.remote_id: f for f in existing_local_files if f.remote_id}

        # Build map of batch-created files (remote_id=None) by file_obj.remote_id
        # for matching against upstream files. When a batch creates VectorStoreFile
        # records, they start with remote_id=None and need to be linked to their
        # upstream counterpart once processing completes.
        # In the OpenAI API, the VectorStoreFile.id equals the source File.id,
        # so we can match by comparing upstream VectorStoreFile.id to FileObject.remote_id.
        batch_file_by_file_remote_id = {}
        for f in existing_local_files:
            if f.remote_id is None and f.file_obj and f.file_obj.remote_id:
                batch_file_by_file_remote_id[f.file_obj.remote_id] = f

        # Get FileObjects that belong to this user/team, indexed by remote_id
        if token.service_account:
            file_objs = await sync_to_async(list)(
                FileObject.objects.filter(token__service_account__team=token.service_account.team)
            )
        else:
            file_objs = await sync_to_async(list)(FileObject.objects.filter(token=token))
        file_remote_id_to_obj = {f.remote_id: f for f in file_objs if f.remote_id}

        response_files = []
        now = timezone.now()

        # Files that need new local records (collected for batch creation)
        files_to_create = []

        for remote_file in remote_files:
            local_vs_file = remote_id_to_local.get(remote_file.id)

            if local_vs_file:
                # Update existing record with latest upstream status
                local_vs_file.status = remote_file.status or local_vs_file.status
                local_vs_file.usage_bytes = remote_file.usage_bytes
                if hasattr(remote_file, "last_error") and remote_file.last_error:
                    local_vs_file.last_error = remote_file.last_error
                await sync_to_async(local_vs_file.save)()

                file_data = remote_file.model_dump(mode="json")
                file_data["id"] = local_vs_file.id
                file_data["vector_store_id"] = vs_obj.id
                file_data["file_id"] = local_vs_file.file_obj.id if local_vs_file.file_obj else None
                response_files.append(file_data)
            else:
                upstream_file_id = remote_file.id
                file_obj = file_remote_id_to_obj.get(upstream_file_id)
                if file_obj:
                    files_to_create.append((file_obj, remote_file))

        # Link upstream files to existing batch-created records or create new ones.
        # Batch-created VectorStoreFile records have remote_id=None and need to be
        # updated with their upstream remote_id once the batch finishes processing.
        if files_to_create:

            @sync_to_async
            def create_or_link_local_records():
                results = []
                for file_obj, remote_vs_file in files_to_create:
                    # Check if there's a batch-created record for this file_obj
                    # that hasn't been linked to an upstream ID yet
                    existing = batch_file_by_file_remote_id.get(file_obj.remote_id)
                    if existing:
                        # Update the batch-created record with the upstream remote_id
                        existing.remote_id = remote_vs_file.id
                        existing.status = remote_vs_file.status or existing.status
                        existing.usage_bytes = remote_vs_file.usage_bytes
                        if hasattr(remote_vs_file, "last_error") and remote_vs_file.last_error:
                            existing.last_error = remote_vs_file.last_error
                        existing.save()
                        results.append((existing, remote_vs_file, file_obj))
                    else:
                        # No batch-created record found, create a new one
                        vs_file, _ = VectorStoreFile.objects.get_or_create(
                            vector_store=vs_obj,
                            remote_id=remote_vs_file.id,
                            defaults={
                                "file_obj": file_obj,
                                "status": remote_vs_file.status or "in_progress",
                                "usage_bytes": remote_vs_file.usage_bytes,
                                "created_at": int(now.timestamp()),
                            },
                        )
                        results.append((vs_file, remote_vs_file, file_obj))
                return results

            created_records = await create_or_link_local_records()
            for vs_file, remote_file, file_obj in created_records:
                file_data = remote_file.model_dump(mode="json")
                file_data["id"] = vs_file.id
                file_data["vector_store_id"] = vs_obj.id
                file_data["file_id"] = file_obj.id
                response_files.append(file_data)

        return JsonResponse(
            {"object": "list", "data": response_files, "has_more": False}, status=200
        )

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

    if not file_obj.remote_id:
        return error_response(
            "File has no remote reference.", error_type="server_error", status=500
        )

    # Use transaction with select_for_update to prevent race conditions on file limit
    max_files = settings.MAX_VECTOR_STORE_FILES

    # Create upstream file FIRST (outside transaction to minimize lock duration)
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

    # Use sync_to_async wrapper with transaction.atomic for the lock
    @sync_to_async
    def create_file_with_lock():
        with transaction.atomic():
            # Re-fetch vector store with lock
            try:
                if token.service_account:
                    vs_obj_locked = VectorStore.objects.select_for_update().get(
                        id=vector_store_id, token__service_account__team=token.service_account.team
                    )
                else:
                    vs_obj_locked = VectorStore.objects.select_for_update().get(
                        id=vector_store_id, token__user=token.user
                    )
            except VectorStore.DoesNotExist:
                return None, None  # Vector store not found

            # Check limit inside locked transaction
            current_count = VectorStoreFile.objects.filter(vector_store=vs_obj_locked).count()

            if current_count >= max_files:
                return vs_obj_locked, None  # Signal limit reached

            # Create local record
            now = timezone.now()
            vs_file_obj = VectorStoreFile(
                vector_store=vs_obj_locked,
                file_obj=file_obj,
                remote_id=remote_vs_file.id,
                status=remote_vs_file.status or "in_progress",
                usage_bytes=remote_vs_file.usage_bytes,
                created_at=int(now.timestamp()),
            )
            vs_file_obj.save()
            return vs_obj_locked, vs_file_obj

    vs_obj_locked, result = await create_file_with_lock()

    if vs_obj_locked is None:
        # Vector store not found error - clean up upstream file
        try:
            await client.vector_stores.files.delete(
                vector_store_id=vs_obj.remote_id, file_id=remote_vs_file.id
            )
        except Exception:
            pass
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    if result is None:
        # Limit reached - clean up upstream file
        try:
            await client.vector_stores.files.delete(
                vector_store_id=vs_obj.remote_id, file_id=remote_vs_file.id
            )
        except Exception:
            pass
        return error_response(f"Vector store file limit reached ({max_files})", status=403)

    # Return upstream response with ID and vector_store_id replaced
    response_data = remote_vs_file.model_dump(mode="json")
    response_data["id"] = result.id
    response_data["vector_store_id"] = vs_obj_locked.id
    # Map file_id to local Aqueduct file ID
    response_data["file_id"] = file_obj.id

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FileUpdateBody))
@log_request
async def vector_store_file(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    file_id: str,
    pydantic_model: Optional[dict] = None,
    *args,
    **kwargs,
):
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

    if not vs_file_obj.remote_id:
        return error_response(
            "Vector store file has no remote reference.", error_type="server_error", status=500
        )

    if request.method == "GET":
        # Retrieve from upstream and sync status
        try:
            remote_vs_file = await vs_file_obj.areload_from_upstream(client)
        except Exception as e:
            return error_response(
                f"Failed to retrieve vector store file from upstream: {str(e)}",
                error_type="server_error",
                status=502,
            )

        # Return upstream response with ID and vector_store_id replaced
        response_data = remote_vs_file.model_dump(mode="json")
        response_data["id"] = vs_file_obj.id
        response_data["vector_store_id"] = vs_obj.id
        # Map file_id to local Aqueduct file ID
        response_data["file_id"] = vs_file_obj.file_obj.id if vs_file_obj.file_obj else None

        return JsonResponse(response_data, status=200)

    elif request.method == "POST":
        # Update file attributes
        params = pydantic_model if pydantic_model else {}

        update_kwargs = {"vector_store_id": vs_obj.remote_id, "file_id": vs_file_obj.remote_id}
        if params.get("attributes"):
            update_kwargs["attributes"] = params["attributes"]

        if not params.get("attributes"):
            return error_response(
                "Missing required parameter: attributes", param="attributes", status=400
            )

        try:
            remote_vs_file = await client.vector_stores.files.update(**update_kwargs)
        except Exception as e:
            return error_response(
                f"Failed to update file attributes on upstream: {str(e)}",
                error_type="server_error",
                status=502,
            )

        # Return upstream response with ID and vector_store_id replaced
        response_data = remote_vs_file.model_dump(mode="json")
        response_data["id"] = vs_file_obj.id
        response_data["vector_store_id"] = vs_obj.id
        # Map file_id to local Aqueduct file ID
        response_data["file_id"] = vs_file_obj.file_obj.id if vs_file_obj.file_obj else None

        return JsonResponse(response_data, status=200)

    # DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}
    try:
        await vs_file_obj.adelete_upstream(client)
    except Exception as e:
        return error_response(
            f"Failed to delete vector store file from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Delete local record
    await sync_to_async(vs_file_obj.delete)()

    # Return response with ID replaced
    response_data = {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_file_content(
    request: ASGIRequest, token: Token, vector_store_id: str, file_id: str, *args, **kwargs
):
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
        vs_file_obj = await VectorStoreFile.objects.select_related("file_obj").aget(
            id=file_id, vector_store=vs_obj
        )
    except VectorStoreFile.DoesNotExist:
        return error_response("Vector store file not found.", param="file_id", status=404)

    if not vs_file_obj.remote_id:
        return error_response(
            "Vector store file has no remote reference.", error_type="server_error", status=500
        )

    # Get content from upstream
    try:
        content_response = await client.vector_stores.files.content(
            vector_store_id=vs_obj.remote_id, file_id=vs_file_obj.remote_id
        )
    except Exception as e:
        return error_response(
            f"Failed to retrieve file content from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Return content directly as binary response
    # Handle different response types from OpenAI client
    if hasattr(content_response, "content"):
        # It's a response object with content attribute
        content = content_response.content
        content_type = getattr(content_response, "content_type", "application/octet-stream")
    elif hasattr(content_response, "read"):
        # It's a file-like object
        content = await content_response.read()
        content_type = "application/octet-stream"
    elif hasattr(content_response, "__iter__") and not isinstance(content_response, (str, bytes)):
        # It's an async iterator (like AsyncPage), convert to bytes
        chunks = []
        async for chunk in content_response:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
            else:
                chunks.append(chunk.encode("utf-8"))
        content = b"".join(chunks)
        content_type = "application/octet-stream"
    else:
        # Assume it's already bytes or string
        if isinstance(content_response, str):
            content = content_response.encode("utf-8")
        else:
            content = content_response
        content_type = "application/octet-stream"

    return HttpResponse(content, content_type=content_type)
