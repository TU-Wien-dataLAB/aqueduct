import json

from django.urls import reverse
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    TextChoices,
    TextCompletionResponse,
    Usage,
)

from gateway.tests.utils import _read_streaming_response_lines
from gateway.tests.utils.base import ChatCompletionsBase
from management.models import Request
from management.models import Usage as AqueductUsage
from mock_api.mock_configs import (
    MockConfig,
    MockStreamingConfig,
    convert_to_stream_data,
    responses_detail_response,
)


class TokenUsageDetailsTest(ChatCompletionsBase):
    @staticmethod
    def _chat_completion_payload(
        prompt_tokens, completion_tokens, cached_tokens=None, reasoning_tokens=None
    ) -> dict:
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            **(
                {"prompt_tokens_details": {"cached_tokens": cached_tokens}}
                if cached_tokens is not None
                else {}
            ),
            **(
                {"completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
                if reasoning_tokens is not None
                else {}
            ),
        )
        return ModelResponse(
            id="chatcmpl-cached-tokens",
            object="chat.completion",
            created=1768398242,
            model="gpt-4.1-nano",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="Cached response."),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ).model_dump()

    def _run_chat_completion_and_get_request(self, response_data):
        with self.mock_server.patch_external_api(
            "chat/completions", MockConfig(response_data=response_data)
        ):
            response = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.content}")
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "Expected one logged request.")
        return requests[0]

    def test_chat_completion_records_token_details(self):
        req = self._run_chat_completion_and_get_request(
            self._chat_completion_payload(1000, 10, cached_tokens=800, reasoning_tokens=6)
        )
        self.assertEqual(req.input_tokens, 1000, "Input tokens should include cached tokens.")
        self.assertEqual(req.cached_input_tokens, 800)
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 6)
        self.assertEqual(req.token_usage.total_tokens, 1010)

    def test_chat_completion_without_cache_details_stores_zero(self):
        req = self._run_chat_completion_and_get_request(
            self._chat_completion_payload(20, 10, cached_tokens=None)
        )
        self.assertEqual(req.input_tokens, 20)
        self.assertEqual(req.cached_input_tokens, 0)
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 0)
        self.assertEqual(req.token_usage.total_tokens, 30)

    def test_response_without_usage_stores_zero(self):
        data = self._chat_completion_payload(20, 10, cached_tokens=5)
        del data["usage"]
        req = self._run_chat_completion_and_get_request(data)
        self.assertEqual(req.input_tokens, 0)
        self.assertEqual(req.cached_input_tokens, 0)
        self.assertEqual(req.output_tokens, 0)
        self.assertEqual(req.reasoning_tokens, 0)

    def test_cached_input_tokens_cannot_exceed_input_tokens(self):
        req = self._run_chat_completion_and_get_request(
            self._chat_completion_payload(1000, 10, cached_tokens=5000)
        )
        self.assertEqual(req.input_tokens, 1000)
        self.assertEqual(
            req.cached_input_tokens, 1000, "Cached input tokens should not exceed input tokens."
        )

    async def test_chat_completion_streaming_records_token_details(self):
        stream_data = [
            ModelResponse(
                id="chatcmpl-cached-stream",
                created=1768398242,
                model="gpt-4.1-nano",
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"role": "assistant", "content": "Cached stream."}}],
            ).model_dump(),
            ModelResponse(
                id="chatcmpl-cached-stream",
                created=1768398242,
                model="gpt-4.1-nano",
                object="chat.completion.chunk",
                choices=[],
                usage=Usage(
                    prompt_tokens=1000,
                    completion_tokens=10,
                    total_tokens=1010,
                    prompt_tokens_details={"cached_tokens": 800},
                    completion_tokens_details={"reasoning_tokens": 6},
                ),
            ).model_dump(),
        ]
        mock_resp = MockStreamingConfig(response_data=convert_to_stream_data(stream_data))
        with self.mock_server.patch_external_api("chat/completions", mock_resp):
            request = self._build_chat_completion_request(self.MESSAGES, stream=True)
            response = await self.async_client.post(**request, content_type="application/json")

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        await _read_streaming_response_lines(response)

        requests = [r async for r in Request.objects.all()]
        self.assertEqual(len(requests), 1, "Expected one logged request.")
        req = requests[0]
        self.assertEqual(req.input_tokens, 1000, "Input tokens should include cached tokens.")
        self.assertEqual(
            req.cached_input_tokens, 800, "Cached input tokens should come from the final chunk."
        )
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 6)
        self.assertEqual(req.token_usage.total_tokens, 1010)

    def test_completions_records_token_details(self):
        # LiteLLM sends text completion requests to chat/completions for chat-only models.
        chat_data = self._chat_completion_payload(1000, 10, cached_tokens=800, reasoning_tokens=6)
        text_data = TextCompletionResponse(
            id="cmpl-cached-tokens",
            object="text_completion",
            created=1768398242,
            model="gpt-4.1-nano",
            choices=[
                TextChoices(
                    text="This is a mock completion response.",
                    index=0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=1000, completion_tokens=10, total_tokens=1010),
        ).model_dump()
        text_data["usage"]["prompt_tokens_details"] = {"cached_tokens": 800}
        text_data["usage"]["completion_tokens_details"] = {"reasoning_tokens": 6}

        with (
            self.mock_server.patch_external_api(
                "chat/completions", MockConfig(response_data=chat_data)
            ),
            self.mock_server.patch_external_api("completions", MockConfig(response_data=text_data)),
        ):
            response = self.client.post(
                reverse("gateway:completions"),
                data=json.dumps({"model": self.model, "prompt": "Hello"}),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.content}")

        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "Expected one logged request.")
        req = requests[0]
        self.assertEqual(req.input_tokens, 1000)
        self.assertEqual(req.cached_input_tokens, 800)
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 6)
        self.assertEqual(req.token_usage.total_tokens, 1010)

    async def test_responses_api_records_token_details(self):
        response_data = responses_detail_response(())
        response_data["usage"]["input_tokens"] = 1000
        response_data["usage"]["input_tokens_details"] = {"cached_tokens": 800}
        response_data["usage"]["output_tokens"] = 10
        response_data["usage"]["output_tokens_details"] = {"reasoning_tokens": 6}
        response_data["usage"]["total_tokens"] = 1010

        with self.mock_server.patch_external_api(
            "responses", MockConfig(response_data=response_data)
        ):
            response = await self.async_client.post(
                reverse("gateway:v1_responses"),
                data=json.dumps(
                    {
                        "model": self.model,
                        "input": [{"role": "user", "content": "Hello, how are you?"}],
                        "max_output_tokens": 50,
                    }
                ),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.content}")

        requests = [r async for r in Request.objects.all()]
        self.assertEqual(len(requests), 1, "Expected one logged request.")
        req = requests[0]
        self.assertEqual(req.input_tokens, 1000)
        self.assertEqual(req.cached_input_tokens, 800)
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 6)
        self.assertEqual(req.token_usage.total_tokens, 1010)

    def test_token_usage_setter_limits_cached_input_tokens(self):
        req = Request()
        req.token_usage = AqueductUsage(input_tokens=100, output_tokens=5, cached_input_tokens=500)
        self.assertEqual(req.cached_input_tokens, 100)

        req.token_usage = AqueductUsage(input_tokens=100, output_tokens=5, cached_input_tokens=-5)
        self.assertEqual(req.cached_input_tokens, 0)

        req.token_usage = AqueductUsage(input_tokens=100, output_tokens=5, cached_input_tokens=40)
        self.assertEqual(req.cached_input_tokens, 40)
        self.assertEqual(req.token_usage, AqueductUsage(100, 5, 40))

    def test_token_usage_setter_limits_reasoning_tokens(self):
        for reported, expected in [(-5, 0), (50, 20), (12, 12)]:
            with self.subTest(reported=reported):
                req = Request()
                req.token_usage = AqueductUsage(
                    input_tokens=10, output_tokens=20, reasoning_tokens=reported
                )
                self.assertEqual(req.reasoning_tokens, expected)
                self.assertEqual(req.token_usage.reasoning_tokens, expected)
                self.assertEqual(req.token_usage.total_tokens, 30)

    def test_reasoning_tokens_cannot_exceed_output_tokens(self):
        req = self._run_chat_completion_and_get_request(
            self._chat_completion_payload(20, 10, reasoning_tokens=50)
        )
        self.assertEqual(req.output_tokens, 10)
        self.assertEqual(req.reasoning_tokens, 10)
        self.assertEqual(req.token_usage.total_tokens, 30)
