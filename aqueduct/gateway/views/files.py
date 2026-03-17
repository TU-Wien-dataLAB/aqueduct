import io
import json
from typing import Literal, Optional

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Sum
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from openai import AsyncOpenAI
from openai.types.file_create_params import FileCreateParams as OpenAIFileCreateParams
from pydantic import BaseModel, ConfigDict, TypeAdapter

from gateway.config import get_files_api_client
from management.models import Batch, FileObject, Token

from .decorators import (
    catch_router_exceptions,
    log_request,
    parse_body,
    process_batch_file,
    token_authenticated,
    tos_accepted,
)
from .errors import error_response


class FilesCreateParams(BaseModel):
    file: bytes
    purpose: Literal["assistants", "batch", "user_data"]
    # IO[bytes] requires arbitrary_types_allowed for model settings
    model_config = ConfigDict(arbitrary_types_allowed=True)


def calculate_expires_at(remote_expires_at: Optional[int]) -> int:
    """Calculate local expiry timestamp, using earlier of local or upstream expiry."""
    now = timezone.now()
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    local_expires_at = int((now + timezone.timedelta(days=expiry_days)).timestamp())

    if remote_expires_at and remote_expires_at < local_expires_at:
        return remote_expires_at
    return local_expires_at


def validate_batch_file(data: bytes):
    """Validate batch file format: valid JSON lines with unique custom_ids."""
    lines = data.decode("utf-8").splitlines()
    custom_ids = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Invalid JSON at line {i + 1}")
        custom_id = d.get("custom_id")
        if not custom_id:
            raise ValueError(f"No custom_id found at line {i + 1}")
        elif custom_id in custom_ids:
            raise ValueError(f"Duplicate custom_id found at line {i + 1}")
        else:
            custom_ids.add(custom_id)


async def sync_batch_file_if_needed(
    remote_file_id: Optional[str],
    token: Token,
    client: AsyncOpenAI,
    batch_obj: Optional["Batch"] = None,
    field_name: Literal["output_file", "error_file"] = "output_file",
) -> Optional[FileObject]:
    """
    Ensure a local FileObject record exists for a batch output/error file.

    When a batch completes, the upstream provider creates output_file_id and
    error_file_id. These files need local FileObject records for:
    1. Access control (user can only access files they own)
    2. Consistent ID mapping (users use Aqueduct IDs, not remote IDs)

    This function is called when retrieving a batch to lazily create
    FileObject records for any output/error files that don't exist locally.
    """
    if not remote_file_id:
        return None

    # Check if we already have a local record for this upstream file
    try:
        return await FileObject.objects.aget(id=remote_file_id, token=token)
    except FileObject.DoesNotExist:
        pass

    # Fetch file metadata from upstream
    try:
        remote_file = await client.files.retrieve(remote_file_id)
    except Exception:
        # File may have been deleted upstream, skip
        return None

    # Create local record with same ownership as batch.
    # Use get_or_create to prevent duplicate records from concurrent requests
    # (e.g., two GET /batches/{id} requests polling the same completed batch).
    local_expires_at = calculate_expires_at(remote_file.expires_at)

    file_obj, _ = await sync_to_async(FileObject.objects.get_or_create)(
        id=remote_file.id,
        token=token,
        defaults={
            "bytes": remote_file.bytes,
            "filename": remote_file.filename,
            "created_at": remote_file.created_at,
            "purpose": remote_file.purpose,
            "expires_at": local_expires_at,
            "preview": "",
            "upstream_url": settings.AQUEDUCT_FILES_API_URL,
        },
    )

    # Update batch record with linked file if batch_obj provided
    if batch_obj:
        setattr(batch_obj, field_name, file_obj)
        await sync_to_async(batch_obj.save)(update_fields=[field_name])

    return file_obj


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FilesCreateParams))
@process_batch_file
@log_request
@catch_router_exceptions
async def files(
    request: ASGIRequest,
    token: Token,
    pydantic_model: Optional[OpenAIFileCreateParams] = None,
    file_content: Optional[bytes] = None,
    file_preview: Optional[str] = None,
    *args,
    **kwargs,
):
    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    if request.method == "GET":
        if token.service_account:
            file_objects = await sync_to_async(list)(
                FileObject.objects.filter(token__service_account__team=token.service_account.team)
                .order_by("-created_at")
                .select_related("token")
            )
        else:
            file_objects = await sync_to_async(list)(
                FileObject.objects.filter(token__user=token.user)
                .order_by("-created_at")
                .select_related("token")
            )

        return JsonResponse(
            {
                "object": "list",
                "data": [
                    f.model.model_dump(exclude_none=True, exclude_unset=True) for f in file_objects
                ],
                "has_more": False,
            },
            status=200,
        )

    # POST /files
    uploaded = pydantic_model["file"]
    purpose = pydantic_model["purpose"]
    filename = uploaded.name

    # Validate file extension for batch files
    if purpose == "batch" and not filename.endswith(".jsonl"):
        return error_response(
            "Only .jsonl files are currently supported for purpose 'batch'.", status=400
        )

    # Enforce per-token total storage limit
    if token.service_account:
        sum_res = await FileObject.objects.filter(
            token__service_account__team=token.service_account.team
        ).aaggregate(sum_bytes=Sum("bytes"))
    else:
        sum_res = await FileObject.objects.filter(token__user=token.user).aaggregate(
            sum_bytes=Sum("bytes")
        )
    current_total = sum_res.get("sum_bytes") or 0
    max_total_bytes = settings.AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB * 1024 * 1024
    if current_total + len(file_content) > max_total_bytes:
        return error_response(
            f"Total files size exceeds {settings.AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB}MB limit.",
            status=413,
        )

    # Validate batch file format (valid JSON, unique custom_ids)
    if purpose == "batch":
        try:
            validate_batch_file(file_content)
        except ValueError as e:
            return error_response(f"Batch file validation failed: {str(e)}", status=400)

    # file_content is already read and processed by @process_batch_file decorator
    # For batch files: model names are rewritten
    # For other files: content is passed as-is

    # Create a file-like object for the OpenAI client
    file_tuple = (filename, io.BytesIO(file_content))

    # Upload to upstream using OpenAI client
    remote_file = await client.files.create(file=file_tuple, purpose=purpose)

    # Create local tracking record with preview
    local_expires_at = calculate_expires_at(remote_file.expires_at)

    file_obj = FileObject(
        id=remote_file.id,
        token=token,
        bytes=len(file_content),
        filename=filename,
        created_at=int(timezone.now().timestamp()),
        purpose=purpose,
        expires_at=local_expires_at,
        preview=file_preview,
        upstream_url=settings.AQUEDUCT_FILES_API_URL,
    )
    await sync_to_async(file_obj.save)()

    # Return response with upstream ID
    response_data = file_obj.model.model_dump(exclude_none=True, exclude_unset=True)

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=True)
@log_request
@catch_router_exceptions
async def file(request: ASGIRequest, token: Token, file_id: str, *args, **kwargs):
    """
    Retrieve or delete a specific file.

    Requires local FileObject record with matching ownership.
    Returns 404 if file not found or not owned by user - NEVER falls back to upstream.
    """
    try:
        if token.service_account:
            file_obj = await FileObject.objects.aget(
                id=file_id, token__service_account__team=token.service_account.team
            )
        else:
            file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return error_response("File not found.", param="file_id", status=404)

    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    if request.method == "GET":
        # Fetch current status from upstream
        remote_file = await file_obj.areload_from_upstream(client)

        # Return response with upstream ID (same as file_obj.id)
        response_data = remote_file.model_dump()
        return JsonResponse(response_data, status=200)

    # DELETE /files/{file_id}
    await file_obj.adelete_upstream(client)

    # Delete local record
    await sync_to_async(file_obj.delete)()

    # Return response with upstream ID
    response_data = {"id": file_id, "object": "file", "deleted": True}
    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@log_request
@catch_router_exceptions
async def file_content(request: ASGIRequest, token: Token, file_id: str, *args, **kwargs):
    """
    Retrieve the content of a specific file.

    Requires local FileObject record with matching ownership.
    Returns 404 if file not found or not owned by user - NEVER falls back to upstream.
    """
    try:
        if token.service_account:
            file_obj = await FileObject.objects.aget(
                id=file_id, token__service_account__team=token.service_account.team
            )
        else:
            file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return error_response("File not found.", param="file_id", status=404)

    try:
        client = get_files_api_client()
    except ValueError:
        return error_response("Files API not configured", status=503)

    # Returns HttpxBinaryResponseContent, use .content to get bytes
    response = await client.files.content(file_obj.id)

    return HttpResponse(response.content, content_type="application/octet-stream", status=200)
