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
from pydantic import BaseModel, ConfigDict, TypeAdapter

from gateway.config import get_files_api_client
from management.models import FileObject, Token

from .decorators import (
    log_request,
    parse_body,
    process_batch_file,
    token_authenticated,
    tos_accepted,
)


class FilesCreateParams(BaseModel):
    file: bytes
    purpose: Literal["batch", "user_data"]
    model_config = ConfigDict(arbitrary_types_allowed=True)


def validate_batch_file(data: bytes):
    """Validate batch file format: valid JSON lines with unique custom_ids."""
    lines = data.decode().splitlines()
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
    remote_file_id: str | None, token: Token, client
) -> FileObject | None:
    """
    Ensure a local FileObject record exists for a batch output/error file.

    When a batch completes, the upstream provider creates output_file_id and
    error_file_id. These files need local FileObject records for:
    1. Access control (user can only access files they own)
    2. Consistent ID mapping (users use Aqueduct IDs, not remote IDs)

    This function is called when retrieving a batch to lazily create
    FileObject records for any output/error files that don't exist locally.

    Args:
        remote_file_id: The upstream provider's file ID (e.g., "file-abc123")
        token: The token that owns the batch (ownership is inherited)
        client: The OpenAI client for fetching file metadata

    Returns:
        The FileObject record (existing or newly created), or None if remote_file_id is None
    """
    if not remote_file_id:
        return None

    # Check if we already have a local record for this remote_id
    try:
        return await FileObject.objects.aget(remote_id=remote_file_id)
    except FileObject.DoesNotExist:
        pass

    # Fetch file metadata from upstream
    try:
        remote_file = await client.files.retrieve(remote_file_id)
    except Exception:
        # File may have been deleted upstream, skip
        return None

    # Create local record with same ownership as batch
    now = timezone.now()
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    file_obj = FileObject(
        token=token,
        bytes=remote_file.bytes,
        filename=remote_file.filename,
        created_at=remote_file.created_at,
        purpose=remote_file.purpose,  # "batch_output" or similar
        expires_at=int((now + timezone.timedelta(days=expiry_days)).timestamp()),
        remote_id=remote_file.id,
        upstream_url=settings.AQUEDUCT_FILES_API_URL,
    )
    await sync_to_async(file_obj.save)()
    return file_obj


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(FilesCreateParams))
@process_batch_file
@log_request
async def files(
    request: ASGIRequest,
    token: Token,
    pydantic_model: Optional[dict] = None,
    file_content: Optional[bytes] = None,
    file_preview: Optional[str] = None,
    *args,
    **kwargs,
):
    client = get_files_api_client()

    if request.method == "GET":
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
        return JsonResponse(
            {"error": "Only .jsonl files are currently supported for purpose 'batch'."}, status=400
        )

    # Enforce per-token total storage limit
    sum_res = await FileObject.objects.filter(token__user=token.user).aaggregate(
        sum_bytes=Sum("bytes")
    )
    current_total = sum_res.get("sum_bytes") or 0
    max_total_bytes = settings.AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB * 1024 * 1024
    if current_total + len(file_content) > max_total_bytes:
        return JsonResponse(
            {
                "error": f"Total files size exceeds "
                f"{settings.AQUEDUCT_FILES_API_MAX_PER_TOKEN_SIZE_MB}MB limit."
            },
            status=413,
        )

    # Validate batch file format (valid JSON, unique custom_ids)
    if purpose == "batch":
        try:
            validate_batch_file(file_content)
        except ValueError as e:
            return JsonResponse({"error": f"Batch file validation failed: {str(e)}"}, status=400)

    # file_content is already read and processed by @process_batch_file decorator
    # For batch files: model names are rewritten
    # For other files: content is passed as-is

    # Create a file-like object for the OpenAI client
    file_tuple = (filename, io.BytesIO(file_content))

    # Upload to upstream using OpenAI client
    remote_file = await client.files.create(file=file_tuple, purpose=purpose)

    # Create local tracking record with preview
    now = timezone.now()
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    file_obj = FileObject(
        token=token,
        bytes=len(file_content),
        filename=filename,
        created_at=int(now.timestamp()),
        purpose=purpose,
        # Use local expiry for cleanup scheduling (remote may have different expiry)
        expires_at=int((now + timezone.timedelta(days=expiry_days)).timestamp()),
        remote_id=remote_file.id,
        preview=file_preview,
        upstream_url=settings.AQUEDUCT_FILES_API_URL,
    )
    await sync_to_async(file_obj.save)()

    # IMPORTANT: Return response with Aqueduct ID, not remote ID
    response_data = remote_file.model_dump()
    response_data["id"] = file_obj.id  # Replace remote ID with Aqueduct ID

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=True)
@log_request
async def file(request: ASGIRequest, token: Token, file_id: str, *args, **kwargs):
    """
    Retrieve or delete a specific file.

    SECURITY: Requires local FileObject record with matching ownership.
    Returns 404 if file not found or not owned by user - NEVER falls back to upstream.
    """
    # SECURITY: Require local record to exist with correct ownership
    # This is the ONLY way to access a file - no fallback to upstream
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return JsonResponse(
            {
                "error": {
                    "message": "File not found.",
                    "type": "invalid_request_error",
                    "param": "file_id",
                }
            },
            status=404,
        )

    remote_id = file_obj.remote_id
    if not remote_id:
        return JsonResponse(
            {"error": {"message": "File has no remote reference.", "type": "server_error"}},
            status=500,
        )

    client = get_files_api_client()

    if request.method == "GET":
        # Fetch current status from upstream
        try:
            remote_file = await client.files.retrieve(remote_id)
        except Exception as e:
            return JsonResponse(
                {
                    "error": {
                        "message": f"Failed to retrieve file from upstream: {str(e)}",
                        "type": "server_error",
                    }
                },
                status=502,
            )

        # Return response with Aqueduct ID
        response_data = remote_file.model_dump()
        response_data["id"] = file_obj.id  # Replace remote ID with Aqueduct ID
        return JsonResponse(response_data, status=200)

    # DELETE /files/{file_id}
    try:
        delete_response = await client.files.delete(remote_id)
    except Exception as e:
        return JsonResponse(
            {
                "error": {
                    "message": f"Failed to delete file from upstream: {str(e)}",
                    "type": "server_error",
                }
            },
            status=502,
        )

    # Delete local record
    await sync_to_async(file_obj.delete)()

    # Return response with Aqueduct ID
    response_data = delete_response.model_dump()
    response_data["id"] = file_obj.id
    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@log_request
async def file_content(request: ASGIRequest, token: Token, file_id: str, *args, **kwargs):
    """
    Retrieve the content of a specific file.

    SECURITY: Requires local FileObject record with matching ownership.
    Returns 404 if file not found or not owned by user - NEVER falls back to upstream.
    """
    # SECURITY: Require local record to exist with correct ownership
    try:
        file_obj = await FileObject.objects.aget(id=file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return JsonResponse(
            {
                "error": {
                    "message": "File not found.",
                    "type": "invalid_request_error",
                    "param": "file_id",
                }
            },
            status=404,
        )

    remote_id = file_obj.remote_id
    if not remote_id:
        return JsonResponse(
            {"error": {"message": "File has no remote reference.", "type": "server_error"}},
            status=500,
        )

    client = get_files_api_client()

    try:
        # Returns HttpxBinaryResponseContent, use .content to get bytes
        response = await client.files.content(remote_id)
    except Exception as e:
        return JsonResponse(
            {
                "error": {
                    "message": f"Failed to retrieve file content from upstream: {str(e)}",
                    "type": "server_error",
                }
            },
            status=502,
        )

    return HttpResponse(response.content, content_type="application/octet-stream", status=200)
