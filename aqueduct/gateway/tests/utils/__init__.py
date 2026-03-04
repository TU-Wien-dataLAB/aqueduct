import json
import warnings
from typing import List

from openai.types.chat import ChatCompletionChunk


def _build_chat_headers(access_token):
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}


def _build_chat_payload(model, messages, max_completion_tokens=50, stream=False):
    payload = {"model": model, "messages": messages, "max_completion_tokens": max_completion_tokens}
    if stream:
        payload["stream"] = True
    return payload


async def _read_streaming_response_lines(response) -> List[str]:
    """
    Collect all streamed lines (each line is a data: ... event) from a StreamingHttpResponse.
    """
    streamed_lines = []
    async for chunk in response.streaming_content:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.strip().splitlines():
            line = line.strip()
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                streamed_lines.append(data)
    return streamed_lines


def _parse_streamed_content_pieces(streamed_lines: List[str]) -> List[str]:
    """
    Parse each chunk as JSON and collect content pieces from OpenAI streaming response.
    """
    content_pieces = []
    for data in streamed_lines:
        try:
            chunk = ChatCompletionChunk.model_validate(json.loads(data))
        except Exception:
            warnings.warn("Chat completion request returned invalid JSON data!")
            continue
        choices = chunk.choices
        if choices:
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content_pieces.append(piece)
    return content_pieces
