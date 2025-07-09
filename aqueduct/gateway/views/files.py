from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_GET
from asgiref.sync import sync_to_async
from django.db.models import Sum
from django.utils import timezone

from openai.types.file_object import FileObject as OpenAIFileObject
from management.models import FileObject
from openai.types import FileCreateParams
from .decorators import token_authenticated, log_request


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated
@log_request
async def files(request: ASGIRequest, token, *args, **kwargs):
    if request.method == "GET":
        file_objs = await sync_to_async(list)(FileObject.objects.filter(token=token))
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
    if purpose != "batch":
        return JsonResponse({"error": f"Purpose '{purpose}' is currently not supported."}, status=400)
    filename = uploaded.name
    if not filename.endswith(".jsonl"):
        return JsonResponse({"error": "Only .jsonl files are currently supported."}, status=400)
    data = uploaded.read()
    if len(data) > 8 * 1024 * 1024:
        return JsonResponse({"error": "File too large. Individual file must be <= 8MB."}, status=400)
    sum_res = await FileObject.objects.filter(token=token).aaggregate(sum_bytes=Sum("bytes"))
    current_total = sum_res.get("sum_bytes") or 0
    if current_total + len(data) > 1024 * 1024 * 1024:
        return JsonResponse({"error": "Total files size exceeds 1GB limit."}, status=400)
    created_at = int(timezone.now().timestamp())
    file_obj = FileObject(token=token, bytes=len(data), filename=filename, created_at=created_at, purpose=purpose)
    await sync_to_async(file_obj.save)()
    await sync_to_async(file_obj.write)(data)
    openai_file = file_obj.model
    return JsonResponse(openai_file.model_dump(exclude_none=True, exclude_unset=True), status=200)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated
@log_request
async def file(request: ASGIRequest, token, file_id: str, *args, **kwargs):
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token=token)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "File not found."}, status=404)
    if request.method == "GET":
        openai_file = file_obj.model
        return JsonResponse(openai_file.model_dump(exclude_none=True, exclude_unset=True), status=200)
    # DELETE /files/{file_id}
    await sync_to_async(file_obj.delete)()
    return JsonResponse({"id": file_id, "deleted": True, "object": "file"}, status=200)


@csrf_exempt
@require_GET
@token_authenticated
@log_request
async def file_content(request: ASGIRequest, token, file_id: str, *args, **kwargs):
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token=token)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "File not found."}, status=404)
    data = await sync_to_async(file_obj.read)()
    return HttpResponse(data, content_type="application/json", status=200)
