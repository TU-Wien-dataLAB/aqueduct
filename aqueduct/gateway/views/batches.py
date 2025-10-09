import asyncio
import json
from collections import deque
from typing import AsyncIterator, Any

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from asgiref.sync import sync_to_async
from django.utils import timezone
from datetime import timedelta
from litellm import Router

from pydantic import ValidationError, TypeAdapter
from openai.types.batch_create_params import BatchCreateParams

from management.models import Batch, FileObject
from django.conf import settings
from .decorators import token_authenticated, log_request, tos_accepted
from .utils import cache_lock
from ..router import get_router


@csrf_exempt
@require_http_methods(["GET", "POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def batches(request: ASGIRequest, token, *args, **kwargs):
    """
    GET /batches - list batches for this token
    POST /batches - create a new batch from an uploaded JSONL file
    """
    if request.method == "GET":
        batch_objs = await sync_to_async(list)(
            Batch.objects.filter(input_file__token__user=token.user)
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
    # Enforce batch creation limits: per-user (user tokens) or per-team (service-account tokens)
    # Only batches that are active (validating, in_progress) count toward limits
    active_statuses = ["validating", "in_progress"]
    if token.service_account:
        qs = Batch.objects.filter(
            input_file__token__service_account__team=token.service_account.team,
            status__in=active_statuses,
        )
        limit = settings.MAX_TEAM_BATCHES
    else:
        qs = Batch.objects.filter(
            input_file__token__user=token.user,
            status__in=active_statuses,
        )
        limit = settings.MAX_USER_BATCHES
    existing = await sync_to_async(qs.count)()
    if existing >= limit:
        return JsonResponse({"error": f"Batch limit reached ({limit})"}, status=403)
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
        file_obj = await FileObject.objects.aget(id=input_file_id, token__user=token.user)
    except FileObject.DoesNotExist:
        return JsonResponse({"error": "Input file not found."}, status=404)
    if file_obj.purpose != "batch":
        return JsonResponse({"error": "File purpose must be 'batch'."}, status=400)

    # Create batch record with expiry
    now = timezone.now()
    created_at = int(now.timestamp())
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    expires_at = int((now + timezone.timedelta(days=expiry_days)).timestamp())
    num_lines = await sync_to_async(file_obj.num_lines)()
    batch_obj = Batch(
        completion_window=params["completion_window"],
        created_at=created_at,
        endpoint=params["endpoint"],
        input_file=file_obj,
        status="validating",
        metadata=params.get("metadata"),
        expires_at=expires_at,
        request_counts={"input": num_lines, "total": 0, "completed": 0, "failed": 0}
    )
    await sync_to_async(batch_obj.save)()
    openai_batch = batch_obj.model
    return JsonResponse(
        openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200
    )


@csrf_exempt
@require_http_methods(["GET"])
@token_authenticated(token_auth_only=True)
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
@token_authenticated(token_auth_only=False)
@log_request
async def batch_cancel(request: ASGIRequest, token, batch_id: str, *args, **kwargs):
    """POST /batches/{batch_id}/cancel - cancel an in-progress batch"""
    try:
        batch_obj = await Batch.objects.aget(
            id=batch_id, input_file__token=token
        )
    except Batch.DoesNotExist:
        return JsonResponse({"error": "Batch not found."}, status=404)

    now_ts = int(timezone.now().timestamp())
    if batch_obj.status == "in_progress":
        batch_obj.cancelling_at = now_ts
        batch_obj.status = "cancelling"
    elif batch_obj.status == "validating" or batch_obj.status == "finalizing":
        batch_obj.cancelling_at = now_ts
        batch_obj.cancelled_at = now_ts
        batch_obj.status = "cancelled"

    await sync_to_async(batch_obj.save)()
    openai_batch = batch_obj.model
    return JsonResponse(
        openai_batch.model_dump(exclude_none=True, exclude_unset=True), status=200
    )


async def _mark_batch_in_progress(batch: Batch):
    """Transition batch from validating to in_progress."""
    now_ts = int(timezone.now().timestamp())
    batch.status = "in_progress"
    batch.in_progress_at = now_ts
    await sync_to_async(batch.save)()


async def _mark_batch_cancelled(batch: Batch):
    """Mark batch as cancelled."""
    now_ts = int(timezone.now().timestamp())
    batch.status = "cancelled"
    batch.cancelled_at = now_ts
    await sync_to_async(batch.save)()


async def _mark_batch_completed(batch: Batch):
    """Mark batch as completed."""
    if batch.status != "completed":
        now_ts = int(timezone.now().timestamp())
        batch.status = "completed"
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
        
        if batch.status == "cancelling":
            await _mark_batch_cancelled(batch)
            continue
        
        if batch.status == "validating":
            await _mark_batch_in_progress(batch)
        
        if not lines:
            continue
        
        yield batch, lines.popleft()
        
        if lines:
            iterators.append((batch, lines))


class AsyncBoundedParallelQueue:
    def __init__(self, max_parallel):
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.tasks = set()
        self.cancelled = False

    async def process(self, coro, *args, **kwargs):
        """Add task to queue, blocking if full"""
        if self.cancelled:
            raise RuntimeError("Queue processing cancelled")

        await self.semaphore.acquire()
        if self.cancelled:
            self.semaphore.release()
            raise RuntimeError("Queue processing cancelled")

        task = asyncio.create_task(self._worker(coro, *args, **kwargs))
        self.tasks.add(task)
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

    def cancel_all(self, cancel_running=False):
        """Cancel pending and optionally running tasks"""
        self.cancelled = True
        for task in self.tasks:
            if cancel_running or not task.done():
                task.cancel()


async def process_batch_request(router: Router, batch: Batch, params: str, processed_ids: set):
    # Process a single batch request line and record its output or error
    # Skip if line parsing yielded None
    try:
        params = json.loads(params)
    except json.JSONDecodeError:
        # invalid input line
        await sync_to_async(batch.append)({"error": "Invalid JSON line"}, error=True)
        return

    data = dict()
    response_data = dict()
    custom_id = params.get("custom_id", None)

    try:
        if not custom_id:
            raise ValueError("Missing custom_id parameter in batch request")
        processing_id = f"{batch.id}-{custom_id}"
        if processing_id in processed_ids:
            return
        else:
            processed_ids.add(processing_id)
        response_data["custom_id"] = custom_id

        method = params.get("method", None)
        url = params.get("url", None)
        if url != batch.endpoint:
            raise RuntimeError(
                f"Request URL mismatch for Batch endpoint! Request URL is {url} while Batch endpoint is set to {batch.endpoint}")
        body = params.get("body", {})

        if body.get("stream", False):
            raise ValueError("Streaming requests are not supported in the /batches API")

        body["timeout"] = settings.AQUEDUCT_BATCH_PROCESSING_TIMEOUT_SECONDS

        # Dispatch based on endpoint
        endpoint = batch.endpoint
        if endpoint.endswith('/completions') and 'chat' in endpoint:
            result = await router.acompletion(**body)
        elif endpoint.endswith('/completions'):
            result = await router.atext_completion(**body)
        elif endpoint.endswith('/embeddings'):
            result = await router.aembedding(**body)
        else:
            raise NotImplementedError(f"Batch endpoint {endpoint} not supported")

        data = result.model_dump(exclude_none=True, exclude_unset=True)
        request_id = data.get("id", f"batch-{custom_id}")
        response_data["response"] = dict(status_code=200, request_id=request_id, body=data)
        response_data["error"] = None
        response_data["id"] = request_id

        await sync_to_async(batch.append)(response_data)
    except Exception as e:
        # Log error and continue
        response_data["error"] = {'error': str(e)}
        response_data["custom_id"] = custom_id
        request_id = data.get("id", None)
        response_data["id"] = request_id
        await sync_to_async(batch.append)(response_data, error=True)


BATCH_LOCK = "batch-processing-lock"


class BatchProcessingSession:
    """Manages a single batch processing session with timing and state."""
    
    def __init__(self):
        self.start_time = timezone.now()
        self.runtime_minutes = settings.AQUEDUCT_BATCH_PROCESSING_RUNTIME_MINUTES
        self.reload_interval = settings.AQUEDUCT_BATCH_PROCESSING_RELOAD_INTERVAL_SECONDS
        self.next_reload_time = self.current_timestamp() + self.reload_interval
        self.processed_ids = set()
        
    def current_timestamp(self) -> float:
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
            Batch.objects.filter(status__in=['validating', 'in_progress', 'cancelling'])
        )
    
    @staticmethod
    async def expire_old_batches(before_timestamp: int):
        """Mark expired batches as expired."""
        await sync_to_async(
            lambda: Batch.objects.filter(
                expires_at__isnull=False,
                expires_at__lt=before_timestamp,
            ).update(status="expired", expired_at=before_timestamp)
        )()


async def _try_acquire_lock_and_process(session: BatchProcessingSession) -> bool:
    """Attempt to acquire lock and run processing loop."""
    lock_timeout_seconds = settings.AQUEDUCT_BATCH_PROCESSING_TIMEOUT_SECONDS + 15
    lock_ttl_seconds = session.runtime_minutes * 60
    
    acquire_deadline = session.start_time + timedelta(seconds=lock_timeout_seconds)
    
    while timezone.now() < acquire_deadline:
        with cache_lock(BATCH_LOCK, lock_ttl_seconds) as acquired:
            if acquired:
                print("Lock acquired! Starting batch processing.")
                await _process_batches(session)
                print("Batch processing loop complete.")
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
            batches = await BatchLoader.load_active_batches()
            if batches:
                print(f"Loaded {len(batches)} active batches")
                rr = round_robin(batches)
                session.schedule_next_reload()
        
        if not batches:
            await asyncio.sleep(2)
            continue
        
        try:
            batch, params = await anext(rr)
            await queue.process(
                process_batch_request, 
                router, 
                batch, 
                params, 
                session.processed_ids
            )
        except StopAsyncIteration:
            batches = None
            rr = None
    
    await queue.join()


async def run_batch_processing():
    """
    Main batch processing loop. Runs for configured duration, processing
    all active batches concurrently with periodic reloads.
    """
    session = BatchProcessingSession()
    
    await BatchLoader.expire_old_batches(session.current_timestamp_int())
    
    if not await _try_acquire_lock_and_process(session):
        print("Could not acquire batch processing lock")
