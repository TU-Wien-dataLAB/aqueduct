import json
import uuid

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from asgiref.sync import sync_to_async
from django.utils import timezone

from pydantic import ValidationError, TypeAdapter
from openai.types.batch_create_params import BatchCreateParams

from management.models import Batch, FileObject
from .decorators import token_authenticated, log_request


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated
@log_request
async def batches(request: ASGIRequest, token, *args, **kwargs):
    """
    GET /batches - list batches for this token
    POST /batches - create a new batch from an uploaded JSONL file
    """
    if request.method == "GET":
        batch_objs = await sync_to_async(list)(
            Batch.objects.filter(input_file__token=token)
        )
        openai_batches = [b.model for b in batch_objs]
        return JsonResponse(
            data={
                "data": [
                    b.model_dump(exclude_none=True, exclude_unset=True)
                    for b in openai_batches
                ],
                "object": "list",
            },
            status=200,
        )

    # POST /batches
    try:
        body = request.body.decode('utf-8')
        TypeAdapter(BatchCreateParams).validate_json(body)
        params = json.loads(body)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    # Validate input file
    input_file_id = params.get("input_file_id")
    try:
        file_obj = await FileObject.objects.aget(id=input_file_id, token=token)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "Input file not found."}, status=404)
    if file_obj.purpose != "batch":
        return JsonResponse({"error": "File purpose must be 'batch'."}, status=400)

    # Create batch record
    now = timezone.now()
    created_at = int(now.timestamp())
    batch_obj = Batch(
        id=f"batch-{uuid.uuid4().hex}",
        completion_window=params["completion_window"],
        created_at=created_at,
        endpoint=params["endpoint"],
        input_file=file_obj,
        status="validating",
        metadata=params.get("metadata"),
    )
    await sync_to_async(batch_obj.save)()
    openai_batch = batch_obj.model
    return JsonResponse(
        openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200
    )


@csrf_exempt
@require_http_methods(["GET"])
@token_authenticated
@log_request
async def batch(request: ASGIRequest, token, batch_id: str, *args, **kwargs):
    """GET /batches/{batch_id} - retrieve a batch"""
    try:
        batch_obj = await Batch.objects.aget(
            id=batch_id, input_file__token=token
        )
    except Batch.DoesNotExist:
        return JsonResponse({"error": "Batch not found."}, status=404)
    openai_batch = batch_obj.model
    return JsonResponse(
        openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200
    )


@csrf_exempt
@require_POST
@token_authenticated
@log_request
async def batch_cancel(request: ASGIRequest, token, batch_id: str, *args, **kwargs):
    """POST /batches/{batch_id}/cancel - cancel an in-progress batch"""
    try:
        batch_obj = await Batch.objects.aget(
            id=batch_id, input_file__token=token
        )
    except Batch.DoesNotExist:
        return JsonResponse({"error": "Batch not found."}, status=404)
    # mark cancelling; TODO: implement actual cancellation logic
    now_ts = int(timezone.now().timestamp())
    batch_obj.status = "cancelling"
    batch_obj.cancelling_at = now_ts
    await sync_to_async(batch_obj.save)()
    openai_batch = batch_obj.model
    return JsonResponse(
        openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200
    )
