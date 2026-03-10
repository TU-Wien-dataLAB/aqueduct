import json
from datetime import datetime
from typing import Any

from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage
from openai.types import (
    Batch,
    BatchRequestCounts,
    Embedding,
    FileObject,
    Image,
    ImagesResponse,
    VectorStore,
    VectorStoreSearchResponse,
)
from openai.types.audio import Transcription
from openai.types.audio.transcription import UsageDuration
from openai.types.images_response import Usage as ImageUsage
from openai.types.images_response import UsageInputTokensDetails
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputMessageItem,
    ResponseInputText,
    ResponseItemList,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from openai.types.vector_store import FileCounts
from openai.types.vector_store_search_response import Content
from openai.types.vector_stores import FileContentResponse, VectorStoreFile, VectorStoreFileBatch
from openai.types.vector_stores.vector_store_file_batch import FileCounts as FileCountsBatch
from pydantic import BaseModel


class MockConfig(BaseModel):
    status_code: int = 200
    response_data: dict[str, Any] | None = {}
    headers: dict[str, str] = {"Content-Type": "application/json"}


class MockStreamingConfig(MockConfig):
    response_data: list[bytes] = []
    headers: dict[str, str] = {"Content-Type": "text/event-stream"}


class MockPlainTextConfig(MockConfig):
    response_data: str = ""
    headers: dict[str, str] = {"Content-Type": "text/plain; charset=utf-8"}


def convert_to_stream_data(data: list[BaseModel]) -> list[bytes]:
    return [b"data: " + json.dumps(item).encode() + b"\n\n" for item in data]


_response_basic_data = {
    "metadata": {},
    "model": "gpt-4.1-nano-2025-04-14",
    "object": "response",
    "parallel_tool_calls": True,
    "temperature": 1.0,
    "max_output_tokens": 50,
    "reasoning": {},
    "store": True,
    "text": {"format": {"type": "text"}, "verbosity": "medium"},
    "tool_choice": "auto",
    "tools": [],
}

default_post_configs = {
    # Note: audio/speech is streaming by default - it cannot be sent as JSON!
    "audio/speech": MockStreamingConfig(response_data=[b"mock", b"audio", b"data"]),
    "audio/transcriptions": MockConfig(
        response_data=Transcription(
            text="This is a mock transcription", usage=UsageDuration(type="duration", seconds=60)
        ).model_dump()
    ),
    "batches": MockConfig(
        response_data=Batch(
            cancelled_at=None,
            cancelling_at=None,
            completed_at=None,
            completion_window="24h",
            created_at=1694268190,
            endpoint="/v1/chat/completions",
            error_file_id=None,
            errors=None,
            expires_at=1773058900,
            failed_at=None,
            finalizing_at=None,
            id="batch_123456789",
            in_progress_at=None,
            input_file_id="file-123456789",
            metadata={"custom_id": "my-batch"},
            object="batch",
            request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
            status="validating",
        ).model_dump()
    ),
    "batches/id/cancel": MockConfig(
        response_data=Batch(
            cancelled_at=1773058900,
            cancelling_at=1773058900,
            completed_at=None,
            completion_window="24h",
            created_at=1694268190,
            endpoint="/v1/chat/completions",
            error_file_id=None,
            errors=None,
            failed_at=None,
            finalizing_at=None,
            id="batch_123456789",
            in_progress_at=None,
            input_file_id="file-123456789",
            metadata={"custom_id": "my-batch"},
            object="batch",
            request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
            status="cancelled",
        ).model_dump()
    ),
    "completions": MockConfig(
        response_data=TextCompletionResponse(
            id="cmpl-123456789",
            object="text_completion",
            created=1694268190,
            model="text-davinci-003",
            choices=[
                {
                    "text": "This is a mock completion response.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
        ).model_dump()
    ),
    "chat/completions": MockConfig(
        response_data=ModelResponse(
            id="chatcmpl-123456789",
            object="chat.completion",
            created=1694268190,
            model="gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock chat completion response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(
                prompt_tokens=15,
                completion_tokens=9,
                total_tokens=24,
                prompt_tokens_details={"cached_tokens": 0, "audio_tokens": 0},
                completion_tokens_details={
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            ),
        ).model_dump()
    ),
    "embeddings": MockConfig(
        response_data=EmbeddingResponse(
            object="list",
            data=[
                Embedding(object="embedding", embedding=[0.1234, -0.5678, 0.9012, -0.3456], index=0)
            ],
            model="text-embedding-ada-002",
            usage=Usage(prompt_tokens=8, total_tokens=8),
        ).model_dump()
    ),
    "files": MockConfig(
        response_data=FileObject(
            id="file-mock-123",
            filename="test.jsonl",
            bytes=100,
            purpose="batch",
            created_at=int(datetime.now().timestamp()),
            expires_at=None,
            status="processed",
            status_details=None,
            object="file",
        ).model_dump()
    ),
    "images/generations": MockConfig(
        response_data=ImagesResponse(
            created=1713833628,
            data=[
                Image(
                    b64_json="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                )
            ],
            usage=ImageUsage(
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                input_tokens_details=UsageInputTokensDetails(text_tokens=10, image_tokens=40),
            ),
        ).model_dump()
    ),
    "responses": MockConfig(
        response_data=Response(
            **_response_basic_data,
            created_at=1741476542,
            id="resp_12345abc",
            output=[
                ResponseOutputMessage(
                    type="message",
                    id="msg_12345abc",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="Hello! I'm doing well, thank you. How can I assist you today?",
                            annotations=[],
                        )
                    ],
                )
            ],
            status="completed",
            usage=ResponseUsage(
                input_tokens=13,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=17,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=30,
            ),
        ).model_dump()
    ),
    "vector_stores": MockConfig(
        response_data=VectorStore(
            id="vs-mock-123",
            name="Test Store",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            expires_after=None,
            metadata=None,
            object="vector_store",
            file_counts=FileCounts(total=0, completed=0, failed=0, in_progress=0, cancelled=0),
            last_active_at=None,
            expires_at=None,
        ).model_dump()
    ),
    "vector_stores/id": MockConfig(
        response_data=VectorStore(
            id="vs-mock-123",
            name="Updated Name",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            expires_after=None,
            metadata=None,
            object="vector_store",
            file_counts=FileCounts(total=0, completed=0, failed=0, in_progress=0, cancelled=0),
            last_active_at=None,
            expires_at=None,
        ).model_dump()
    ),
    "vector_stores/id/file_batches": MockConfig(
        response_data=VectorStoreFileBatch(
            id="vsb-mock-1",
            status="in_progress",
            created_at=1741476542,
            file_counts=FileCountsBatch(total=2, completed=0, failed=0, in_progress=2, cancelled=0),
            object="vector_store.files_batch",
            vector_store_id="vs-mock-123",
        ).model_dump()
    ),
    "vector_stores/id/file_batches/id/cancel": MockConfig(
        response_data=VectorStoreFileBatch(
            id="vsb-mock-1",
            status="cancelled",
            created_at=1741476542,
            file_counts=FileCountsBatch(total=2, completed=0, failed=0, in_progress=0, cancelled=2),
            object="vector_store.files_batch",
            vector_store_id="vs-mock-123",
        ).model_dump()
    ),
    "vector_stores/id/files": MockConfig(
        response_data=VectorStoreFile(
            id="vsf-mock-123",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            object="vector_store.file",
            vector_store_id="vs-mock-123",
        ).model_dump()
    ),
    "vector_stores/id/files/id": MockConfig(
        response_data=VectorStoreFile(
            id="vsf-mock-123",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            object="vector_store.file",
            vector_store_id="vs-mock-123",
            attributes={"key": "value"},
        ).model_dump()
    ),
    "vector_stores/id/search": MockConfig(
        response_data={
            "object": "vector_store.search_results.page",
            "data": [
                VectorStoreSearchResponse(
                    file_id="file-mock-123",
                    filename="test.txt",
                    score=0.95,
                    attributes={},
                    content=[Content(text="Test content", type="text")],
                ).model_dump()
            ],
        }
    ),
}

_chat_completion_stream_data = [
    ModelResponse(
        id="chatcmpl-12345",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Beneath the sky so vast and blue,  \n", "role": "assistant"},
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
    ).model_dump(),
    ModelResponse(
        id="chatcmpl-12345",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        choices=[
            {"index": 0, "delta": {"content": "Whispers of dreams drift softly through,  \n"}}
        ],
        stream=True,
        stream_options={"include_usage": True},
    ).model_dump(),
    ModelResponse(
        id="chatcmpl-12345",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {"content": "A gentle breeze, a song so sweet,  \n"}}],
        stream=True,
        stream_options={"include_usage": True},
    ).model_dump(),
    ModelResponse(
        id="chatcmpl-12345",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {"content": "Moments of magic, softly complete."}}],
        stream=True,
        stream_options={"include_usage": True},
    ).model_dump(),
    ModelResponse(
        id="chatcmpl-12345",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        stream=True,
        stream_options={"include_usage": True},
    ).model_dump(),
]


_responses_stream_data: list[BaseModel] = [
    ResponseCreatedEvent(
        response=Response(
            **_response_basic_data,
            id="resp_12345abc",
            created_at=1769184439.0,
            output=[],
            status="in_progress",
        ),
        sequence_number=0,
        type="response.created",
    ).model_dump(),
    ResponseInProgressEvent(
        response=Response(
            **_response_basic_data,
            id="resp_12345abc",
            created_at=1769184439.0,
            output=[],
            status="in_progress",
        ),
        sequence_number=1,
        type="response.in_progress",
    ).model_dump(),
    ResponseOutputItemAddedEvent(
        item=ResponseOutputMessage(
            id="msg_67890def", content=[], role="assistant", status="in_progress", type="message"
        ),
        output_index=0,
        sequence_number=2,
        type="response.output_item.added",
    ).model_dump(),
    ResponseContentPartAddedEvent(
        content_index=0,
        item_id="msg_67890def",
        output_index=0,
        part=ResponseOutputText(annotations=[], text="", type="output_text"),
        sequence_number=3,
        type="response.content_part.added",
    ).model_dump(),
    ResponseTextDeltaEvent(
        content_index=0,
        delta="Hello, ",
        item_id="msg_67890def",
        logprobs=[],
        output_index=0,
        sequence_number=4,
        type="response.output_text.delta",
    ).model_dump(),
    ResponseTextDeltaEvent(
        content_index=0,
        delta="how are you?",
        item_id="msg_67890def",
        logprobs=[],
        output_index=0,
        sequence_number=5,
        type="response.output_text.delta",
    ).model_dump(),
    ResponseTextDoneEvent(
        content_index=0,
        item_id="msg_67890def",
        logprobs=[],
        output_index=0,
        sequence_number=6,
        text="Hello, how are you?",
        type="response.output_text.done",
    ).model_dump(),
    ResponseContentPartDoneEvent(
        content_index=0,
        item_id="msg_67890def",
        output_index=0,
        part=ResponseOutputText(
            annotations=[], text="Hello, how are you?", type="output_text", logprobs=[]
        ),
        sequence_number=7,
        type="response.content_part.done",
    ).model_dump(),
    ResponseOutputItemDoneEvent(
        item=ResponseOutputMessage(
            id="msg_67890def",
            content=[
                ResponseOutputText(
                    annotations=[], text="Hello, how are you?", type="output_text", logprobs=[]
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
        output_index=0,
        sequence_number=8,
        type="response.output_item.done",
    ).model_dump(),
    ResponseCompletedEvent(
        response=Response(
            **_response_basic_data,
            id="resp_12345abc",
            created_at=1769184439.0,
            output=[
                ResponseOutputMessage(
                    id="msg_67890def",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Hello, how are you?",
                            type="output_text",
                            logprobs=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            status="completed",
            usage=ResponseUsage(
                input_tokens=13,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=7,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=17,
            ),
        ),
        sequence_number=9,
        type="response.completed",
    ).model_dump(),
]

default_post_stream_configs = {
    "chat/completions": MockStreamingConfig(
        response_data=convert_to_stream_data(_chat_completion_stream_data)
    ),
    "responses": MockStreamingConfig(response_data=convert_to_stream_data(_responses_stream_data)),
}

default_get_configs = {
    "batches/id": MockConfig(
        response_data=Batch(
            cancelled_at=None,
            cancelling_at=None,
            completed_at=None,
            completion_window="24h",
            created_at=1694268190,
            endpoint="/v1/chat/completions",
            error_file_id=None,
            errors=None,
            expires_at=1773058900,
            failed_at=None,
            finalizing_at=None,
            id="batch_123456789",
            in_progress_at=None,
            input_file_id="file-123456789",
            metadata={"custom_id": "my-batch"},
            object="batch",
            request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
            status="validating",
        ).model_dump()
    ),
    "responses/id": MockConfig(
        response_data=Response(
            **_response_basic_data,
            created_at=1769125418.0,
            id="resp_12345abc",
            output=[
                ResponseOutputMessage(
                    content=[
                        ResponseOutputText(
                            annotations=[], text="Hello, how are you?", type="output_text"
                        )
                    ],
                    id="msg_12345abc",
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            status="completed",
            usage=ResponseUsage(
                input_tokens=13,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=21,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=34,
            ),
        ).model_dump()
    ),
    "responses/id/input_items": MockConfig(
        response_data=ResponseItemList(
            data=[
                ResponseInputMessageItem(
                    content=[ResponseInputText(text="Hello, how are you?", type="input_text")],
                    id="msg_12345abc",
                    role="user",
                    status="completed",
                    type="message",
                )
            ],
            first_id="msg_12345abc",
            has_more=False,
            last_id="msg_12345abc",
            object="list",
        ).model_dump()
    ),
    "vector_stores/id": MockConfig(
        response_data=VectorStore(
            id="vs-mock-123",
            name="Test Store",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            expires_after=None,
            metadata=None,
            object="vector_store",
            file_counts=FileCounts(total=0, completed=0, failed=0, in_progress=0, cancelled=0),
            last_active_at=None,
            expires_at=None,
        ).model_dump()
    ),
    "vector_stores/id/file_batches/id": MockConfig(
        response_data=VectorStoreFileBatch(
            id="vsb-mock-1",
            status="in_progress",
            created_at=1741476542,
            file_counts=FileCountsBatch(total=2, completed=0, failed=0, in_progress=2, cancelled=0),
            object="vector_store.files_batch",
            vector_store_id="vs-mock-123",
        ).model_dump()
    ),
    "vector_stores/id/files": MockConfig(
        response_data={
            "data": [
                VectorStoreFile(
                    id="vsf-mock-123",
                    status="completed",
                    usage_bytes=100,
                    created_at=1741476542,
                    last_error=None,
                    object="vector_store.file",
                    vector_store_id="vs-mock-123",
                ).model_dump()
            ],
            "has_more": False,
        }
    ),
    "vector_stores/id/files/id": MockConfig(
        response_data=VectorStoreFile(
            id="vsf-mock-123",
            status="completed",
            usage_bytes=0,
            created_at=1741476542,
            object="vector_store.file",
            vector_store_id="vs-mock-123",
        ).model_dump()
    ),
    "vector_stores/id/files/id/content": MockConfig(  # TODO!
        response_data={
            "data": [
                FileContentResponse(text="Test ").model_dump(),
                FileContentResponse(text="file ").model_dump(),
                FileContentResponse(text="content").model_dump(),
            ],
            "has_more": False,
            "object": "vector_store.file_content.page",
        }
    ),
}

default_delete_configs = {
    "responses/id": MockConfig(response_data=None),
    "vector_stores/id": MockConfig(
        response_data={"id": "vs-mock-123", "object": "vector_store.deleted", "deleted": True}
    ),
}

special_configs: dict[str, MockConfig] = {}
