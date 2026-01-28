import asyncio
import json
import logging
from collections import deque
from datetime import timedelta
from typing import AsyncIterator

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from litellm import Router
from openai.types.batch_create_params import BatchCreateParams
from pydantic import TypeAdapter

from gateway.config import get_files_api_client, get_router
from gateway.views.files import sync_batch_file_if_needed
from management.models import Batch, BatchStatus, FileObject, Token

from .decorators import log_request, parse_body, token_authenticated, tos_accepted
from .utils import cache_lock

log = logging.getLogger("aqueduct")


class BatchService:
    """Business logic for batch operations."""

    @staticmethod
    async def create_batch(user, params: dict) -> Batch:
        """Create a new batch from validated parameters."""
        file_obj = await FileObject.objects.aget(id=params["input_file_id"], token__user=user)

        if file_obj.purpose != "batch":
            raise ValueError("File purpose must be 'batch'")

        now = timezone.now()
        num_lines = await sync_to_async(file_obj.num_lines)()

        batch_obj = Batch(
            completion_window=params["completion_window"],
            created_at=int(now.timestamp()),
            endpoint=params["endpoint"],
            input_file=file_obj,
            status=BatchStatus.VALIDATING,
            metadata=params.get("metadata"),
            expires_at=int(
                (now + timedelta(days=settings.AQUEDUCT_FILES_API_EXPIRY_DAYS)).timestamp()
            ),
            request_counts={"input": num_lines, "total": 0, "completed": 0, "failed": 0},
        )
        await sync_to_async(batch_obj.save)()
        return batch_obj

    @staticmethod
    async def check_batch_limit(token) -> tuple[bool, int]:
        """
        Check if user/team has reached batch creation limit.
        Returns (can_create, limit).
        """
        active_statuses = [BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS]

        if token.service_account:
            count = await sync_to_async(
                Batch.objects.filter(
                    input_file__token__service_account__team=token.service_account.team,
                    status__in=active_statuses,
                ).count
            )()
            limit = settings.MAX_TEAM_BATCHES
        else:
            count = await sync_to_async(
                Batch.objects.filter(
                    input_file__token__user=token.user, status__in=active_statuses
                ).count
            )()
            limit = settings.MAX_USER_BATCHES

        return count < limit, limit


async def _list_batches(token):
    """List all batches for user."""
    batch_objs = await sync_to_async(list)(Batch.objects.filter(input_file__token__user=token.user))
    return JsonResponse(
        {
            "data": [b.model.model_dump(exclude_none=True, exclude_unset=True) for b in batch_objs],
            "object": "list",
        }
    )


async def _create_batch(token: Token, params: BatchCreateParams):
    """Create a new batch."""
    try:
        batch_obj = await BatchService.create_batch(token.user, params)
        return JsonResponse(
            batch_obj.model.model_dump(exclude_none=True, exclude_unset=True), status=200
        )
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "Input file not found."}, status=404)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)


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
    GET /batches - list batches for this token
    POST /batches - create a new batch from an uploaded JSONL file
    """
    if request.method == "GET":
        return await _list_batches(token)
    else:
        can_create, limit = await BatchService.check_batch_limit(token)
        if not can_create:
            return JsonResponse({"error": f"Batch limit reached ({limit})"}, status=403)

        return await _create_batch(token, pydantic_model)


@csrf_exempt
@require_http_methods(["GET"])
@token_authenticated(token_auth_only=True)
@log_request
async def batch(request: ASGIRequest, token: Token, batch_id: str, *args, **kwargs):
    """
    Retrieve a specific batch.

    This endpoint also handles lazy synchronization of batch output/error files:
    When a batch completes, the upstream provider creates output_file_id and error_file_id.
    These files need local FileObject records for access control, so we create them
    on-demand when the batch is queried.
    """
    # Verify batch ownership through input_file relationship
    try:
        batch_obj = await Batch.objects.select_related("input_file__token").aget(
            id=batch_id, input_file__token=token
        )
    except Batch.DoesNotExist:
        return JsonResponse(
            {
                "error": {
                    "message": "Batch not found.",
                    "type": "invalid_request_error",
                    "param": "batch_id",
                }
            },
            status=404,
        )

    client = get_files_api_client()

    # If batch has a remote_id, fetch current status from upstream
    if batch_obj.remote_id:
        try:
            remote_batch = await client.batches.retrieve(batch_obj.remote_id)

            # Sync output/error files if they exist upstream but not locally
            # These files inherit ownership from the batch's input_file token
            output_file_obj = await sync_batch_file_if_needed(
                remote_batch.output_file_id, token, client
            )
            error_file_obj = await sync_batch_file_if_needed(
                remote_batch.error_file_id, token, client
            )

            # Build response with Aqueduct IDs
            response_data = remote_batch.model_dump()
            response_data["id"] = batch_obj.id  # Replace remote batch ID with Aqueduct ID
            response_data["input_file_id"] = batch_obj.input_file_id  # Already an Aqueduct ID

            # Replace remote file IDs with Aqueduct IDs
            if output_file_obj:
                response_data["output_file_id"] = output_file_obj.id
            if error_file_obj:
                response_data["error_file_id"] = error_file_obj.id

            return JsonResponse(response_data, status=200)

        except Exception:
            # Fall back to local data if upstream fetch fails
            pass

    # Return local batch data
    return JsonResponse(
        batch_obj.model.model_dump(exclude_none=True, exclude_unset=True), status=200
    )


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=False)
@log_request
async def batch_cancel(request: ASGIRequest, token, batch_id: str, *args, **kwargs):
    """POST /batches/{batch_id}/cancel - cancel an in-progress batch"""
    try:
        batch_obj = await Batch.objects.aget(id=batch_id, input_file__token=token)
    except Batch.DoesNotExist:
        return JsonResponse({"error": "Batch not found."}, status=404)

    now_ts = int(timezone.now().timestamp())
    if batch_obj.status == BatchStatus.IN_PROGRESS:
        batch_obj.cancelling_at = now_ts
        batch_obj.status = BatchStatus.CANCELLING
        await sync_to_async(batch_obj.save)()
    elif batch_obj.status == BatchStatus.VALIDATING or batch_obj.status == BatchStatus.FINALIZING:
        batch_obj.cancelling_at = now_ts
        batch_obj.cancelled_at = now_ts
        batch_obj.status = BatchStatus.CANCELLED
        await sync_to_async(batch_obj.save)()

    openai_batch = batch_obj.model
    return JsonResponse(openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200)


async def _mark_batch_in_progress(batch: Batch):
    """Transition batch from validating to in_progress."""
    now_ts = int(timezone.now().timestamp())
    batch.status = BatchStatus.IN_PROGRESS
    batch.in_progress_at = now_ts
    await sync_to_async(batch.save)()


async def _mark_batch_cancelled(batch: Batch):
    """Mark batch as cancelled."""
    now_ts = int(timezone.now().timestamp())
    batch.status = BatchStatus.CANCELLED
    batch.cancelled_at = now_ts
    await sync_to_async(batch.save)()


async def _mark_batch_completed(batch: Batch):
    """Mark batch as completed."""
    if batch.status != BatchStatus.COMPLETED:
        now_ts = int(timezone.now().timestamp())
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = now_ts
        await sync_to_async(batch.save)()


async def round_robin(batches: list[Batch]) -> AsyncIterator[tuple[Batch, str]]:
    """
    Round-robin iterator over batch requests.

    Yields (batch, request_line) tuples, automatically handling:
    - Status transitions (validating -> in_progress)
    - Cancellations
    - Completion detection
    """
    iterators = deque()
    for batch in batches:
        lines = await sync_to_async(batch.input_file_lines)()
        if lines:
            iterators.append((batch, lines))
        else:
            await _mark_batch_completed(batch)

    while iterators:
        batch, lines = iterators.popleft()

        await sync_to_async(batch.refresh_from_db)(fields=["status"])

        if batch.status == BatchStatus.CANCELLING:
            await _mark_batch_cancelled(batch)
            continue

        if batch.status == BatchStatus.VALIDATING:
            await _mark_batch_in_progress(batch)

        if not lines:
            continue

        yield batch, lines.popleft()

        if lines:
            iterators.append((batch, lines))


class AsyncBoundedParallelQueue:
    def __init__(self, max_parallel):
        self.semaphore = asyncio.BoundedSemaphore(max_parallel)
        self.tasks = set()

    async def process(self, coro, *args, **kwargs):
        """Add task to queue, blocking if full"""
        await self.semaphore.acquire()

        task = asyncio.create_task(self._worker(coro, *args, **kwargs))
        self.tasks.add(task)
        log.debug(f"Added task to {self.__class__.__name__} (Running {len(self.tasks)} tasks)")
        task.add_done_callback(self._remove_task)
        return task

    async def _worker(self, coro, *args, **kwargs):
        """Execute task and release semaphore"""
        try:
            return await coro(*args, **kwargs)
        finally:
            self.semaphore.release()

    def _remove_task(self, task):
        """Cleanup completed task"""
        try:
            self.tasks.remove(task)
        except KeyError:
            pass

    async def join(self):
        """Wait for all tasks to complete"""
        await asyncio.gather(*self.tasks)


class EndpointDispatcher:
    """Dispatches requests to appropriate router methods based on endpoint."""

    ENDPOINT_MAP = {
        "/chat/completions": "acompletion",
        "/completions": "atext_completion",
        "/embeddings": "aembedding",
    }

    def __init__(self, router: Router):
        self.router = router

    async def dispatch(self, endpoint: str, body: dict):
        """Dispatch request to appropriate router method."""
        for endpoint_suffix, method_name in self.ENDPOINT_MAP.items():
            if endpoint.endswith(endpoint_suffix):
                if endpoint_suffix == "/completions" and "chat" in endpoint:
                    method_name = "acompletion"

                method = getattr(self.router, method_name)
                return await method(**body)

        raise ValueError(f"Endpoint {endpoint} not supported")


class BatchRequestProcessor:
    """Handles processing of individual batch requests."""

    def __init__(self, router: Router, processed_ids: set):
        self.router = router
        self.processed_ids = processed_ids
        self.dispatcher = EndpointDispatcher(router)

    async def process(self, batch: Batch, params_str: str):
        """Process a single batch request and record result."""
        try:
            params = self._parse_params(params_str)
            processing_id = self._get_processing_id(batch, params)
            if processing_id in self.processed_ids:
                return
            else:
                self.processed_ids.add(processing_id)

            log.info(f"Processing request {batch.id}-{params.get('custom_id')}")
            result = await self._execute_request(batch, params)
            await self._record_success(batch, params, result)

        except Exception as e:
            await self._record_error(batch, params_str, e)

    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """Parse and validate request parameters."""
        try:
            return json.loads(params_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON line")

    @staticmethod
    def _get_processing_id(batch: Batch, params: dict) -> str:
        """Check if request was already processed."""
        custom_id = params.get("custom_id")
        if not custom_id:
            raise ValueError("Missing custom_id parameter")

        return f"{batch.id}-{custom_id}"

    @staticmethod
    def _validate_request(batch: Batch, params: dict):
        """Validate request parameters."""
        url = params.get("url")
        if url != batch.endpoint:
            raise RuntimeError(f"Request URL mismatch: {url} != {batch.endpoint}")

        body = params.get("body", {})
        if body.get("stream", False):
            raise ValueError("Streaming not supported in batch API")

    async def _execute_request(self, batch: Batch, params: dict):
        """Execute the AI request via router."""
        self._validate_request(batch, params)

        body = params.get("body", {})
        body["timeout"] = settings.AQUEDUCT_BATCH_PROCESSING_TIMEOUT_SECONDS

        return await self.dispatcher.dispatch(batch.endpoint, body)

    @staticmethod
    async def _record_success(batch: Batch, params: dict, result):
        """Record successful request result."""
        data = result.model_dump(exclude_none=True, exclude_unset=True)
        custom_id = params.get("custom_id")

        response_data = {
            "id": data.get("id", f"batch-{custom_id}"),
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": data.get("id", f"batch-{custom_id}"),
                "body": data,
            },
            "error": None,
        }

        await sync_to_async(batch.append)(response_data)

    @staticmethod
    async def _record_error(batch: Batch, params_str: str, error: Exception):
        """Record request error."""
        try:
            params = json.loads(params_str)
            custom_id = params.get("custom_id")
        except Exception:
            custom_id = None

        error_data = {"id": None, "custom_id": custom_id, "error": {"error": str(error)}}

        await sync_to_async(batch.append)(error_data, error=True)


async def process_batch_request(router: Router, batch: Batch, params: str, processed_ids: set):
    """Process a single batch request line and record its output or error."""
    processor = BatchRequestProcessor(router, processed_ids)
    await processor.process(batch, params)


BATCH_LOCK = "batch-processing-lock"


class BatchProcessingSession:
    """Manages a single batch processing session with timing and state."""

    def __init__(self):
        self.start_time = timezone.now()
        self.runtime_minutes = settings.AQUEDUCT_BATCH_PROCESSING_RUNTIME_MINUTES
        self.reload_interval = settings.AQUEDUCT_BATCH_PROCESSING_RELOAD_INTERVAL_SECONDS
        self.next_reload_time = self.current_timestamp() + self.reload_interval
        self.processed_ids = set()

    @staticmethod
    def current_timestamp() -> float:
        """Get current timestamp as float."""
        return timezone.now().timestamp()

    def current_timestamp_int(self) -> int:
        """Get current timestamp as int."""
        return int(self.current_timestamp())

    def should_stop(self) -> bool:
        """Check if session should stop based on runtime limit."""
        end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
        return timezone.now() >= end_time

    def should_reload_batches(self) -> bool:
        """Check if it's time to reload batches from database."""
        return self.current_timestamp() > self.next_reload_time

    def schedule_next_reload(self):
        """Schedule the next batch reload."""
        self.next_reload_time = self.current_timestamp() + self.reload_interval


class BatchLoader:
    """Handles loading and reloading batches from database."""

    @staticmethod
    async def load_active_batches() -> list[Batch]:
        """Load all active batches from database."""
        return await sync_to_async(list)(
            Batch.objects.filter(
                status__in=[BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.CANCELLING]
            )
        )

    @staticmethod
    async def expire_old_batches(before_timestamp: int):
        """Mark expired batches as expired."""
        await sync_to_async(
            lambda: Batch.objects.filter(
                expires_at__isnull=False, expires_at__lt=before_timestamp
            ).update(status=BatchStatus.EXPIRED, expired_at=before_timestamp)
        )()


async def _try_acquire_lock_and_process(session: BatchProcessingSession) -> bool:
    """Attempt to acquire lock and run processing loop."""
    lock_timeout_seconds = settings.AQUEDUCT_BATCH_PROCESSING_TIMEOUT_SECONDS + 15
    lock_ttl_seconds = session.runtime_minutes * 60

    acquire_deadline = session.start_time + timedelta(seconds=lock_timeout_seconds)

    while timezone.now() < acquire_deadline:
        with cache_lock(BATCH_LOCK, lock_ttl_seconds) as acquired:
            if acquired:
                log.info("Lock acquired! Starting batch processing.")
                await _process_batches(session)
                log.info("Batch processing loop complete.")
                return True
            else:
                await asyncio.sleep(1)

    return False


async def _process_batches(session: BatchProcessingSession):
    """Core processing loop - load batches and process them."""
    max_parallel = settings.AQUEDUCT_BATCH_PROCESSING_CONCURRENCY()
    queue = AsyncBoundedParallelQueue(max_parallel=max_parallel)
    router = get_router()

    batches = None
    rr = None

    while not session.should_stop():
        if batches is None or session.should_reload_batches():
            log.debug("Reloading batches...")
            batches = await BatchLoader.load_active_batches()
            if batches:
                rr = round_robin(batches)
            log.info(f"Loaded {len(batches)} active batches")
            session.schedule_next_reload()
            assert not session.should_reload_batches()

        if not batches:
            await asyncio.sleep(2)
            continue

        try:
            batch, params = await anext(rr)
            await queue.process(process_batch_request, router, batch, params, session.processed_ids)
        except StopAsyncIteration:
            batches = []

    await queue.join()


async def run_batch_processing():
    """
    Main batch processing loop. Runs for configured duration, processing
    all active batches concurrently with periodic reloads.
    """
    session = BatchProcessingSession()

    await BatchLoader.expire_old_batches(session.current_timestamp_int())

    if not await _try_acquire_lock_and_process(session):
        log.warning("Could not acquire batch processing lock")
