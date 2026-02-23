from typing import Optional

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Q
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from openai.types import VectorStore as OpenAIVectorStore
from openai.types.vector_store import FileCounts as VectorStoreFileCounts
from openai.types.vector_store_create_params import VectorStoreCreateParams
from openai.types.vector_store_update_params import VectorStoreUpdateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import Token, VectorStore

from .decorators import (
    log_request,
    parse_body,
    require_files_api_client,
    token_authenticated,
    tos_accepted,
)
from .errors import error_response


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(VectorStoreCreateParams))
@log_request
async def vector_stores(
    request: ASGIRequest, token: Token, pydantic_model: Optional[dict] = None, *args, **kwargs
):
    """
    GET /v1/vector_stores - List vector stores
    POST /v1/vector_stores - Create vector store
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    if request.method == "GET":
        # List user's vector stores from local DB
        if token.service_account:
            vector_stores_qs = VectorStore.objects.filter(
                token__service_account__team=token.service_account.team
            )
        else:
            vector_stores_qs = VectorStore.objects.filter(token__user=token.user)

        vector_stores_list = await sync_to_async(list)(
            vector_stores_qs.order_by("-created_at")
            .select_related("token")
            .annotate(
                file_count_total=Count("files"),
                file_count_completed=Count("files", filter=Q(files__status="completed")),
                file_count_failed=Count("files", filter=Q(files__status="failed")),
                file_count_in_progress=Count("files", filter=Q(files__status="in_progress")),
                file_count_cancelled=Count("files", filter=Q(files__status="cancelled")),
            )
        )

        return JsonResponse(
            {
                "object": "list",
                "data": [
                    OpenAIVectorStore(
                        id=vs.id,
                        object="vector_store",
                        name=vs.name,
                        status=vs.status,
                        usage_bytes=vs.usage_bytes,
                        created_at=vs.created_at,
                        metadata=vs.metadata,
                        expires_after=vs.expires_after,
                        file_counts=VectorStoreFileCounts(
                            total=vs.file_count_total,
                            completed=vs.file_count_completed,
                            failed=vs.file_count_failed,
                            in_progress=vs.file_count_in_progress,
                            cancelled=vs.file_count_cancelled,
                        ),
                        last_active_at=vs.last_active_at,
                    ).model_dump(mode="json")
                    for vs in vector_stores_list
                ],
                "has_more": False,
            },
            status=200,
        )

    # POST /v1/vector_stores - Create vector store
    params = pydantic_model if pydantic_model else {}
    name = params.get("name")

    if not name:
        return error_response("Missing required parameter: name", param="name", status=400)

    # Check user/team limits before creating
    if token.service_account:
        limit = settings.MAX_TEAM_VECTOR_STORES
        active_count = await sync_to_async(
            VectorStore.objects.filter(
                token__service_account__team=token.service_account.team
            ).count
        )()
    else:
        limit = settings.MAX_USER_VECTOR_STORES
        active_count = await sync_to_async(
            VectorStore.objects.filter(token__user=token.user).count
        )()

    if active_count >= limit:
        return error_response(f"Vector store limit reached ({limit})", status=403)

    # Create on upstream
    try:
        create_kwargs = {"name": name}
        if params.get("expires_after"):
            create_kwargs["expires_after"] = params["expires_after"]
        if params.get("chunking_strategy"):
            create_kwargs["chunking_strategy"] = params["chunking_strategy"]
        if params.get("metadata"):
            create_kwargs["metadata"] = params["metadata"]

        remote_vs = await client.vector_stores.create(**create_kwargs)
    except Exception as e:
        return error_response(
            f"Failed to create vector store on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local record with upstream ID
    now = timezone.now()
    vs_obj = VectorStore(
        id=remote_vs.id,
        token=token,
        name=name,
        status=remote_vs.status or "completed",
        usage_bytes=getattr(remote_vs, "usage_bytes", 0),
        created_at=int(now.timestamp()),
        expires_after=params.get("expires_after"),
        chunking_strategy=params.get("chunking_strategy"),
        metadata=params.get("metadata"),
        upstream_url=settings.AQUEDUCT_FILES_API_URL,
    )
    await sync_to_async(vs_obj.save)()

    # Return upstream response directly (ID already matches)
    response_data = remote_vs.model_dump(mode="json")

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(VectorStoreUpdateParams))
@log_request
@require_files_api_client
async def vector_store(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    pydantic_model: Optional[dict] = None,
    client=None,
    *args,
    **kwargs,
):
    """
    GET /v1/vector_stores/{vector_store_id} - Retrieve vector store
    POST /v1/vector_stores/{vector_store_id} - Modify vector store
    DELETE /v1/vector_stores/{vector_store_id} - Delete vector store
    """

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
        # Retrieve from upstream and sync status
        try:
            remote_vs = await vs_obj.areload_from_upstream(client)
        except Exception as e:
            return error_response(
                f"Failed to retrieve vector store from upstream: {str(e)}",
                error_type="server_error",
                status=502,
            )

        # Return upstream response directly (ID already matches)
        response_data = remote_vs.model_dump(mode="json")

        return JsonResponse(response_data, status=200)

    elif request.method == "POST":
        # Modify vector store
        params = pydantic_model if pydantic_model else {}

        updatable_fields = ("name", "expires_after", "metadata")
        modify_kwargs = {k: params[k] for k in updatable_fields if params.get(k)}

        if modify_kwargs:
            try:
                remote_vs = await client.vector_stores.update(vs_obj.id, **modify_kwargs)
            except Exception as e:
                return error_response(
                    f"Failed to update vector store on upstream: {str(e)}",
                    error_type="server_error",
                    status=502,
                )

            # Update local record
            for field, value in modify_kwargs.items():
                setattr(vs_obj, field, value)
            vs_obj.status = remote_vs.status
            vs_obj.usage_bytes = remote_vs.usage_bytes
            await sync_to_async(vs_obj.save)()

            # Return upstream response directly (ID already matches)
            response_data = remote_vs.model_dump(mode="json")

            return JsonResponse(response_data, status=200)

        # No changes requested, return current state
        try:
            remote_vs = await vs_obj.areload_from_upstream(client)
            response_data = remote_vs.model_dump(mode="json")
            return JsonResponse(response_data, status=200)
        except Exception as e:
            return error_response(
                f"Failed to retrieve vector store from upstream: {str(e)}",
                error_type="server_error",
                status=502,
            )

    # DELETE /v1/vector_stores/{vector_store_id}
    try:
        await vs_obj.adelete_upstream(client)
    except Exception as e:
        return error_response(
            f"Failed to delete vector store from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Capture ID before delete (Django sets pk to None after delete)
    deleted_id = vs_obj.id

    # Delete local record
    await sync_to_async(vs_obj.delete)()

    # Return with upstream ID
    return JsonResponse(
        {"id": deleted_id, "object": "vector_store.deleted", "deleted": True}, status=200
    )


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def vector_store_search(
    request: ASGIRequest, token: Token, vector_store_id: str, *args, **kwargs
):
    """
    POST /v1/vector_stores/{vector_store_id}/search - Search vector store
    """
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Vector Store API not configured", status=503)

    try:
        if token.service_account:
            vs_obj = await VectorStore.objects.aget(
                id=vector_store_id, token__service_account__team=token.service_account.team
            )
        else:
            vs_obj = await VectorStore.objects.aget(id=vector_store_id, token__user=token.user)
    except VectorStore.DoesNotExist:
        return error_response("Vector store not found.", param="vector_store_id", status=404)

    # Get search parameters from request body
    try:
        import json

        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body", param="body", status=400)

    query = body.get("query")
    if not query:
        return error_response("Missing required parameter: query", param="query", status=400)

    # Prepare search kwargs
    search_kwargs = {"vector_store_id": vs_obj.id, "query": query}

    # Add optional parameters
    if body.get("filters"):
        search_kwargs["filters"] = body["filters"]
    if body.get("max_num_results"):
        search_kwargs["max_num_results"] = body["max_num_results"]
    if body.get("min_score"):
        search_kwargs["min_score"] = body["min_score"]
    if body.get("rewrite_query"):
        search_kwargs["rewrite_query"] = body["rewrite_query"]

    # Search on upstream
    try:
        search_results = await client.vector_stores.search(**search_kwargs)
    except Exception as e:
        return error_response(
            f"Failed to search vector store on upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Return upstream response directly (IDs already match)
    results_data = search_results.model_dump(mode="json")

    return JsonResponse(results_data, status=200)
