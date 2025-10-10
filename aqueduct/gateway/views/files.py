import json

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Sum
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from management.models import FileObject

from .decorators import log_request, token_authenticated, tos_accepted


def validate_batch_file(data: bytes):
    lines = data.decode().splitlines()
    custom_ids = set()
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
            custom_id = d.get("custom_id")
            if not custom_id:
                raise ValueError(f"No custom_id found at line {i + 1}")
            elif custom_id in custom_ids:
                raise ValueError(f"Duplicate custom_id found at line {i + 1}")
            else:
                custom_ids.add(custom_id)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Invalid JSON at line {i + 1}")


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def files(request: ASGIRequest, token, *args, **kwargs):
    if request.method == "GET":
        file_objs = await sync_to_async(list)(FileObject.objects.filter(token__user=token.user))
        openai_files = [f.model for f in file_objs]
        return JsonResponse(
            data={
                "data": [f.model_dump(exclude_none=True, exclude_unset=True) for f in openai_files],
                "object": "list",
            },
            status=200,
        )
    # POST /files
    uploaded = request.FILES.get("file")
    purpose = request.POST.get("purpose")
    if not uploaded or not purpose:
        return JsonResponse({"error": "Both 'file' and 'purpose' are required."}, status=400)
    if purpose not in ["batch", "user_data"]:
        return JsonResponse(
            {"error": f"Purpose '{purpose}' is currently not supported."}, status=400
        )
    filename = uploaded.name
    if purpose == "batch" and not filename.endswith(".jsonl"):
        return JsonResponse(
            {"error": "Only .jsonl files are currently supported for purpose 'batch'."}, status=400
        )
    data = uploaded.read()
    # Enforce per-file size limit from settings
    max_file_bytes = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_file_bytes:
        return JsonResponse(
            {
                "error": f"File too large. Individual file must be "
                f"<= {settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB}MB."
            },
            status=400,
        )
    sum_res = await FileObject.objects.filter(token__user=token.user).aaggregate(
        sum_bytes=Sum("bytes")
    )
    current_total = sum_res.get("sum_bytes") or 0
    # Enforce per-token total storage limit from settings
    max_total_bytes = settings.AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB * 1024 * 1024
    if current_total + len(data) > max_total_bytes:
        return JsonResponse(
            {
                "error": f"Total files size exceeds "
                f"{settings.AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB}MB limit."
            },
            status=400,
        )

    if purpose == "batch":
        try:
            validate_batch_file(data)
        except ValueError as e:
            return JsonResponse({"error": f"Batch file validation failed: {str(e)}"}, status=400)

    now = timezone.now()
    created_at = int(now.timestamp())
    # Expire file after 1 week
    # Set expiry based on settings (days)
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    expires_at = int((now + timezone.timedelta(days=expiry_days)).timestamp())
    file_obj = FileObject(
        token=token,
        bytes=len(data),
        filename=filename,
        created_at=created_at,
        purpose=purpose,
        expires_at=expires_at,
    )
    await sync_to_async(file_obj.save)()
    await sync_to_async(file_obj.write)(data)
    openai_file = file_obj.model
    return JsonResponse(openai_file.model_dump(exclude_none=True, exclude_unset=True), status=200)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=False)
@log_request
async def file(request: ASGIRequest, file_id: str, *args, **kwargs):
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token__user=request.user)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "File not found."}, status=404)
    if request.method == "GET":
        openai_file = file_obj.model
        return JsonResponse(
            openai_file.model_dump(exclude_none=True, exclude_unset=True), status=200
        )
    # DELETE /files/{file_id}
    await sync_to_async(file_obj.delete)()
    return JsonResponse({"id": file_id, "deleted": True, "object": "file"}, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@log_request
async def file_content(request: ASGIRequest, token, file_id: str, *args, **kwargs):
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "File not found."}, status=404)
    data = await sync_to_async(file_obj.read)()
    return HttpResponse(data, content_type="application/json", status=200)
