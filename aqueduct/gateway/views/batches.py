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
from litellm import Router

from pydantic import ValidationError, TypeAdapter
from openai.types.batch_create_params import BatchCreateParams

from management.models import Batch, FileObject
from django.conf import settings
from .decorators import token_authenticated, log_request
from ..router import get_router


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

    # Create batch record with expiry
    now = timezone.now()
    created_at = int(now.timestamp())
    expiry_days = settings.AQUEDUCT_FILES_API_EXPIRY_DAYS
    expires_at = int((now + timezone.timedelta(days=expiry_days)).timestamp())
    batch_obj = Batch(
        completion_window=params["completion_window"],
        created_at=created_at,
        endpoint=params["endpoint"],
        input_file=file_obj,
        status="validating",
        metadata=params.get("metadata"),
        expires_at=expires_at,
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


async def process_batch_request(router: Router, batch: Batch, params: str):
    # Process a single batch request line and record its output or error
    # Skip if line parsing yielded None
    try:
        params = json.loads(params)
    except json.JSONDecodeError:
        # invalid input line
        sync_to_async(batch.append)({"error": "Invalid JSON line"}, error=True)
        return

    try:
        # Dispatch based on endpoint
        endpoint = batch.endpoint
        if endpoint.endswith('/completions') and 'chat' in endpoint:
            result = await router.acompletion(**params)
        elif endpoint.endswith('/completions'):
            result = await router.atext_completion(**params)
        elif endpoint.endswith('/embeddings'):
            result = await router.aembedding(**params)
        else:
            raise NotImplementedError(f"Batch endpoint {endpoint} not supported")

        data = result.model_dump(exclude_none=True, exclude_unset=True)
        await sync_to_async(batch.append)(data)
    except Exception as e:
        # Log error and continue
        err = {'error': str(e)}
        sync_to_async(batch.append)(err, error=True)


async def run_batch_processing():
    start_time = timezone.now()
    runtime_hours = settings.AQUEDUCT_BATCH_PROCESSING_RUNTIME_HOURS
    run_until = int((start_time + timezone.timedelta(hours=runtime_hours)).timestamp())
    curr_time = lambda: timezone.now().timestamp()

    max_parallel = settings.AQUEDUCT_BATCH_PROCESSING_CONCURRENCY
    queue = AsyncBoundedParallelQueue(max_parallel=max_parallel)
    router = get_router()

    # Fetch batches ready to process
    batches = await sync_to_async(list)(
        Batch.objects.filter(status__in=['validating', 'in_progress'])
    )

    async for batch, params in round_robin(batches):
        if curr_time() > run_until:
            break
        await queue.process(process_batch_request, router, batch, params)

    await queue.join()
