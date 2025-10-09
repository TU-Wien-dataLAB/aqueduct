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


async def round_robin(batches: list[Batch]) -> AsyncIterator[tuple[Batch, str]]:
    """
    Async round-robin over batches and their input_file_iterator,
    refreshing status between iterations.
    """
    iterators = await sync_to_async(deque)((b, b.input_file_lines()) for b in batches)
    for batch, lines in iterators:
        # This should not happen but do it defensively
        await sync_to_async(batch.refresh_from_db)(fields=["status"])
        if len(lines) == 0 and batch.status != "completed":
            now_ts = int(timezone.now().timestamp())
            batch.status = "completed"
            batch.completed_at = now_ts
            await sync_to_async(batch.save)()

    while iterators:
        batch, lines = iterators.popleft()
        try:
            await sync_to_async(batch.refresh_from_db)(fields=["status"])
            if batch.status == "cancelling":
                now_ts = int(timezone.now().timestamp())
                batch.status = "cancelled"
                batch.cancelled_at = now_ts
                await sync_to_async(batch.save)()
                raise StopAsyncIteration

            if batch.status == "validating":
                now_ts = int(timezone.now().timestamp())
                batch.status = "in_progress"
                batch.in_progress_at = now_ts
                await sync_to_async(batch.save)()

            if len(lines) == 0:
                # cannot raise StopIteration in generator due to PEP 479
                # This
                raise StopAsyncIteration

            yield batch, lines.popleft()

        except StopAsyncIteration:
            # End of this batch's iteration or batch has been canceled
            pass
        else:
            # Re-add active batch to the end
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


async def run_batch_processing():
    start_time = timezone.now()
    start_ts = int(start_time.timestamp())

    runtime_minutes = settings.AQUEDUCT_BATCH_PROCESSING_RUNTIME_MINUTES
    lock_acquire_timeout = settings.AQUEDUCT_BATCH_PROCESSING_TIMEOUT_SECONDS + 15
    curr_time = lambda: timezone.now().timestamp()

    run_until = int((start_time + timedelta(minutes=runtime_minutes)).timestamp())
    acquire_lock_until = int((start_time + timedelta(seconds=lock_acquire_timeout)).timestamp())

    # expire batches whose expires_at timestamp has passed in bulk
    await sync_to_async(
        lambda: Batch.objects.filter(
            expires_at__isnull=False,
            expires_at__lt=start_ts,
        ).update(status="expired", expired_at=start_ts)
    )()

    # begin processing loop
    lock_acquired = False
    while not lock_acquired and curr_time() < acquire_lock_until:
        lock_ttl = (runtime_minutes * 60) - (start_ts - curr_time())
        with cache_lock(BATCH_LOCK, lock_ttl) as acquired:
            if acquired:
                lock_acquired = True
                print("Lock acquired! Starting batch processing.")
                max_parallel: int = settings.AQUEDUCT_BATCH_PROCESSING_CONCURRENCY()

                queue = AsyncBoundedParallelQueue(max_parallel=max_parallel)
                router = get_router()

                reload_interval = settings.AQUEDUCT_BATCH_PROCESSING_RELOAD_INTERVAL_SECONDS
                update_batches_next = curr_time() + reload_interval
                batches, rr = None, None
                processed_ids = set() # keep track of processed ids as requests might be running when we reload batches
                while curr_time() < run_until:
                    if batches is None or rr is None or curr_time() > update_batches_next:
                        update_batches_next = curr_time() + reload_interval
                        batches = await sync_to_async(list)(
                            Batch.objects.filter(status__in=['validating', 'in_progress', 'cancelling'])
                        )
                        if batches:
                            print(f"Updated batches! Processing {len(batches)} batches...")
                            rr = round_robin(batches)

                    if not batches:
                        await asyncio.sleep(2)
                        continue

                    try:
                        batch, params = await anext(rr)
                        await queue.process(process_batch_request, router, batch, params, processed_ids)
                    except StopAsyncIteration:
                        pass

                await queue.join()
                print("Batch processing loop complete.")
            else:
                await asyncio.sleep(1)
                continue
