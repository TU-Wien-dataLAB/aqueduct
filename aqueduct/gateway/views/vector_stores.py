from typing import Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Q
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from openai import AsyncOpenAI
from openai.types import VectorStore as OpenAIVectorStore
from openai.types import VectorStoreSearchParams
from openai.types.vector_store import FileCounts as VectorStoreFileCounts
from openai.types.vector_store_create_params import VectorStoreCreateParams
from openai.types.vector_store_update_params import VectorStoreUpdateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import Token, VectorStore

from .decorators import (
    catch_router_exceptions,
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
@catch_router_exceptions
async def vector_stores(
    request: ASGIRequest,
    token: Token,
    pydantic_model: VectorStoreCreateParams | None = None,
    *args: Any,
    **kwargs: Any,
) -> JsonResponse:
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

        vector_stores_qs = (
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
                        status=vs.status,  # type: ignore[arg-type]
                        usage_bytes=vs.usage_bytes,
                        created_at=vs.created_at,
                        metadata=vs.metadata,
                        expires_after=vs.expires_after,
                        file_counts=VectorStoreFileCounts(
                            total=getattr(vs, "file_count_total", 0),
                            completed=getattr(vs, "file_count_completed", 0),
                            failed=getattr(vs, "file_count_failed", 0),
                            in_progress=getattr(vs, "file_count_in_progress", 0),
                            cancelled=getattr(vs, "file_count_cancelled", 0),
                        ),
                        last_active_at=vs.last_active_at,
                    ).model_dump(mode="json")
                    async for vs in vector_stores_qs
                ],
                "has_more": False,
            },
            status=200,
        )

    # POST /v1/vector_stores - Create vector store
    if not pydantic_model:
        return error_response("Missing required parameter: name", param="name", status=400)
    name = pydantic_model.get("name")

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
    create_kwargs: dict[str, Any] = {"name": name}
    if pydantic_model.get("expires_after"):
        create_kwargs["expires_after"] = pydantic_model["expires_after"]
    if pydantic_model.get("chunking_strategy"):
        create_kwargs["chunking_strategy"] = pydantic_model["chunking_strategy"]
    if pydantic_model.get("metadata"):
        create_kwargs["metadata"] = pydantic_model["metadata"]

    remote_vs = await client.vector_stores.create(**create_kwargs)

    # Create local record with upstream ID
    now = timezone.now()
    vs_obj = VectorStore(
        id=remote_vs.id,
        token=token,
        name=name,
        status=remote_vs.status or "completed",
        usage_bytes=getattr(remote_vs, "usage_bytes", 0),
        created_at=int(now.timestamp()),
        expires_after=pydantic_model.get("expires_after"),
        chunking_strategy=pydantic_model.get("chunking_strategy"),
        metadata=pydantic_model.get("metadata"),
        upstream_url=settings.AQUEDUCT_FILES_API_URL or "",
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
@catch_router_exceptions
async def vector_store(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    pydantic_model: VectorStoreUpdateParams | None = None,
    client: AsyncOpenAI | None = None,
    *args: Any,
    **kwargs: Any,
) -> JsonResponse:
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
        remote_vs = await vs_obj.areload_from_upstream(client)

        if not remote_vs:
            return error_response("Vector store not found.", param="vector_store_id", status=404)

        # Return upstream response directly (ID already matches)
        response_data = remote_vs.model_dump(mode="json")

        return JsonResponse(response_data, status=200)

    if request.method == "POST":
        # Modify vector store
        if not pydantic_model:
            return error_response("No fields to update", status=400)

        modify_kwargs: dict[str, Any] = {}
        if pydantic_model.get("name"):
            modify_kwargs["name"] = pydantic_model["name"]
        if pydantic_model.get("expires_after"):
            modify_kwargs["expires_after"] = pydantic_model["expires_after"]
        if pydantic_model.get("metadata"):
            modify_kwargs["metadata"] = pydantic_model["metadata"]

        if modify_kwargs:
            if not client:
                return error_response("Vector Store API not configured", status=503)
            remote_vs = await client.vector_stores.update(vs_obj.id, **modify_kwargs)

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
        remote_vs = await vs_obj.areload_from_upstream(client)
        if not remote_vs:
            return error_response("Vector store not found.", param="vector_store_id", status=404)
        response_data = remote_vs.model_dump(mode="json")
        return JsonResponse(response_data, status=200)

    await vs_obj.adelete_upstream(client)

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
@parse_body(model=TypeAdapter(VectorStoreSearchParams))
@log_request
@require_files_api_client
@catch_router_exceptions
async def vector_store_search(
    request: ASGIRequest,
    token: Token,
    vector_store_id: str,
    pydantic_model: VectorStoreSearchParams,
    *args: Any,
    **kwargs: Any,
) -> JsonResponse:
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

    # Search on upstream
    search_results = await client.vector_stores.search(vector_store_id=vs_obj.id, **pydantic_model)

    # Return upstream response directly (IDs already match)
    results_data = search_results.model_dump(mode="json")

    return JsonResponse(results_data, status=200)
