import json
from typing import Any, Dict

from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage
from openai.types import Image, ImagesResponse
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
from pydantic import BaseModel


class MockConfig(BaseModel):
    status_code: int = 200
    response_data: Dict[str, Any] | None = {}
    headers: Dict[str, str] = {"Content-Type": "application/json"}


class MockStreamingConfig(MockConfig):
    response_data: list[bytes] = []
    headers: Dict[str, str] = {"Content-Type": "text/event-stream"}


class MockPlainTextConfig(MockConfig):
    response_data: str = ""
    headers: Dict[str, str] = {"Content-Type": "text/plain; charset=utf-8"}


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
    "audio/speech": MockStreamingConfig(
        response_data=[b"mock", b"audio", b"data"]
    ),  # TODO: use a model
    "audio/transcriptions": MockConfig(
        response_data=Transcription(
            text="This is a mock transcription", usage=UsageDuration(type="duration", seconds=60)
        ).model_dump()
    ),
    # Batches are already mocked with a mock_router; TODO: check the response data
    "batches": MockConfig(
        response_data={  # TODO: use a model?
            "id": "batch_123456789",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "errors": None,
            "input_file_id": "file-123456789",
            "completion_window": "24h",
            "status": "validating",
            "created_at": 1694268190,
            "in_progress_at": None,
            "expires_at": 1694354590,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "error_file_id": None,
            "results_file_id": None,
            "metadata": {"custom_id": "my-batch"},
        }
    ),
    # TODO: check this!
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
            usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
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
                {"object": "embedding", "embedding": [0.1234, -0.5678, 0.9012, -0.3456], "index": 0}
            ],
            model="text-embedding-ada-002",
            usage=Usage(prompt_tokens=8, total_tokens=8),
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
}

# TODO: use openai/lllm models, dump them and encode
_chat_completion_stream_data = [
    b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Beneath the sky so vast and blue,  \\n","role":"assistant"}}],"stream_options":{"include_usage":true}}\n\n',
    b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Whispers of dreams drift softly through,  \\n"}}],"stream_options":{"include_usage":true}}\n\n',
    b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"A gentle breeze, a song so sweet,  \\n"}}],"stream_options":{"include_usage":true}}\n\n',
    b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Moments of magic, softly complete."}}],"stream_options":{"include_usage":true}}\n\n',
]

_responses_stream_data: list[dict] = [
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
    "chat/completions": MockStreamingConfig(response_data=_chat_completion_stream_data),
    "responses": MockStreamingConfig(
        response_data=[
            b"data: " + json.dumps(item).encode() + b"\n\n" for item in _responses_stream_data
        ]
    ),
}

default_get_configs = {
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
}

default_delete_configs = {"responses/id": MockConfig(response_data=None)}

special_configs: dict[str, MockConfig] = {}
