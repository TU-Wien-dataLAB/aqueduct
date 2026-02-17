from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from openai.types.batch_create_params import BatchCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client
from management.models import Batch, BatchStatus, FileObject, Token

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
from .errors import error_response
from .files import sync_batch_file_if_needed


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(TypeAdapter(BatchCreateParams))
@log_request
async def batches(
    request: ASGIRequest,
    token: Token,
    pydantic_model: BatchCreateParams | None = None,
    *args,
    **kwargs,
):
    """
    GET /batches - list user's batches from local DB
    POST /batches - create a new batch on upstream
    """
    if request.method == "GET":
        # Users only see their own batches, not all batches from upstream
        batch_objects = await sync_to_async(list)(
            Batch.objects.filter(token__user=token.user)
            .order_by("-created_at")
            .select_related("input_file")
        )

        return JsonResponse(
            {
                "object": "list",
                "data": [b.model.model_dump() for b in batch_objects],
                "has_more": False,
            },
            status=200,
        )

    # POST /batches

    # Check batch limit before creating batch (for upstream quota management)
    max_batches = getattr(settings, "MAX_USER_BATCHES", 10)
    if token.service_account:
        active_count = await sync_to_async(
            Batch.objects.filter(
                token__service_account__team=token.service_account.team,
                status__in=[
                    BatchStatus.VALIDATING,
                    BatchStatus.IN_PROGRESS,
                    BatchStatus.CANCELLING,
                ],
            ).count
        )()
        limit = getattr(settings, "MAX_TEAM_BATCHES", 50)
    else:
        active_count = await sync_to_async(
            Batch.objects.filter(
                token__user=token.user,
                status__in=[
                    BatchStatus.VALIDATING,
                    BatchStatus.IN_PROGRESS,
                    BatchStatus.CANCELLING,
                ],
            ).count
        )()
        limit = max_batches

    if active_count >= limit:
        return error_response(f"Batch limit reached ({limit})", status=403)

    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    # Get the input file's remote ID
    try:
        file_obj = await FileObject.objects.aget(
            id=pydantic_model["input_file_id"], token__user=token.user
        )
    except FileObject.DoesNotExist:
        return error_response("Input file not found.", param="input_file_id", status=404)

    if not file_obj.remote_id:
        return error_response("File was not uploaded to upstream API.", status=400)

    # Create batch on upstream using remote file ID
    try:
        remote_batch = await client.batches.create(
            input_file_id=file_obj.remote_id,
            endpoint=pydantic_model["endpoint"],
            completion_window=pydantic_model["completion_window"],
            metadata=pydantic_model.get("metadata"),
        )
    except Exception as e:
        return error_response(
            f"Failed to create batch on upstream: {str(e)}", error_type="server_error", status=502
        )

    # Create local tracking record
    now = timezone.now()
    batch_obj = Batch(
        completion_window=pydantic_model["completion_window"],
        created_at=int(now.timestamp()),
        endpoint=pydantic_model["endpoint"],
        input_file=file_obj,
        token=token,
        status=remote_batch.status,
        metadata=pydantic_model.get("metadata"),
        remote_id=remote_batch.id,
        expires_at=remote_batch.expires_at,
        request_counts=remote_batch.request_counts.model_dump()
        if remote_batch.request_counts
        else {},
    )
    await sync_to_async(batch_obj.save)()

    # Return response with Aqueduct IDs, not remote IDs
    response_data = remote_batch.model_dump()
    response_data["id"] = batch_obj.id
    response_data["input_file_id"] = file_obj.id

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET"])
@token_authenticated(token_auth_only=True)
@log_request
async def batch(request: ASGIRequest, token: Token, batch_id: str, *args, **kwargs):
    """
    GET /batches/{batch_id} - retrieve a batch from upstream

    Returns 404 if batch not found or not owned by user - NEVER falls back to upstream.

    When output_file_id or error_file_id exist in the upstream
    response but don't have local FileObject records, create them lazily.
    """
    try:
        batch_obj = await Batch.objects.select_related("input_file").aget(id=batch_id, token=token)
    except Batch.DoesNotExist:
        return error_response("Batch not found.", param="batch_id", status=404)

    remote_id = batch_obj.remote_id
    if not remote_id:
        return error_response(
            "Batch has no remote reference.", error_type="server_error", status=500
        )

    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    try:
        remote_batch = await batch_obj.areload_from_upstream(client)
    except Exception as e:
        return error_response(
            f"Failed to retrieve batch from upstream: {str(e)}",
            error_type="server_error",
            status=502,
        )

    # Create local FileObject records for output/error files if missing (inherit ownership from input_file token)
    output_file_obj = await sync_batch_file_if_needed(
        remote_batch.output_file_id, token, client, batch_obj, "output_file"
    )
    error_file_obj = await sync_batch_file_if_needed(
        remote_batch.error_file_id, token, client, batch_obj, "error_file"
    )

    # Return response with Aqueduct IDs, not remote IDs
    response_data = remote_batch.model_dump()
    response_data["id"] = batch_obj.id
    response_data["input_file_id"] = batch_obj.input_file_id

    # Replace remote file IDs with Aqueduct IDs (if files were synced)
    if output_file_obj:
        response_data["output_file_id"] = output_file_obj.id
    if error_file_obj:
        response_data["error_file_id"] = error_file_obj.id

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@log_request
async def batch_cancel(request: ASGIRequest, token: Token, batch_id: str, *args, **kwargs):
    """
    POST /batches/{batch_id}/cancel - cancel a batch on upstream

    Requires local Batch record with matching ownership.
    Returns 404 if batch not found or not owned by user - NEVER falls back to upstream.
    """
    try:
        batch_obj = await Batch.objects.select_related("input_file").aget(id=batch_id, token=token)
    except Batch.DoesNotExist:
        return error_response("Batch not found.", param="batch_id", status=404)

    remote_id = batch_obj.remote_id
    if not remote_id:
        return error_response(
            "Batch has no remote reference.", error_type="server_error", status=500
        )

    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    try:
        remote_batch = await client.batches.cancel(remote_id)
    except Exception as e:
        return error_response(
            f"Failed to cancel batch on upstream: {str(e)}", error_type="server_error", status=502
        )

    # Update local record
    batch_obj.status = remote_batch.status
    if remote_batch.cancelling_at:
        batch_obj.cancelling_at = remote_batch.cancelling_at
    if remote_batch.cancelled_at:
        batch_obj.cancelled_at = remote_batch.cancelled_at
    await sync_to_async(batch_obj.save)()

    # Return response with Aqueduct IDs
    response_data = remote_batch.model_dump()
    response_data["id"] = batch_obj.id
    response_data["input_file_id"] = batch_obj.input_file_id

    return JsonResponse(response_data, status=200)
