import io
import json
from pathlib import Path

from asgiref.sync import sync_to_async
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings

from gateway.tests.utils.base import INTEGRATION_TEST_BACKEND, GatewayTTSSTTestCase
from management.models import Request


class SpeechEndpointTest(GatewayTTSSTTestCase):
    """Test the speech (text-to-speech) endpoint."""

    async def test_speech_endpoint_basic(self):
        """Test basic speech generation with valid parameters."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS tests require OpenAI backend")

        payload = {
            "model": self.tts_model,
            "input": "Hello, this is a test of the text-to-speech system.",
            "voice": "alloy",
            "response_format": "mp3",
        }

        response = await self.async_client.post(
            self.url_tts,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Check that response is audio data (streaming response)
        self.assertEqual(response.get("Content-Type"), "text/event-stream")

        # Collect the streamed content
        audio_data = b""
        async for chunk in response.streaming_content:
            audio_data += chunk

        self.assertGreater(len(audio_data), 0, "Should receive audio data")

        # Check that the database contains one request
        requests = await sync_to_async(list)(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after speech generation."
        )
        req = requests[0]
        self.assertIn("speech", req.path, "Request endpoint should be for speech.")
        self.assertIsNotNone(req.token_usage)

    def test_speech_endpoint_invalid_model(self):
        """Test speech endpoint with invalid model."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS tests require OpenAI backend")

        payload = {
            "model": "invalid-tts-model",
            "input": "Hello, this is a test.",
            "voice": "alloy",
        }

        response = self.client.post(
            self.url_tts,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code, 400, f"Expected 400 Bad Request, got {response.status_code}"
        )

    def test_speech_endpoint_missing_required_fields(self):
        """Test speech endpoint with missing required fields."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS tests require OpenAI backend")

        payload = {
            "model": self.tts_model,
            # Missing 'input' field
            "voice": "alloy",
        }

        response = self.client.post(
            self.url_tts,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code, 400, f"Expected 400 Bad Request, got {response.status_code}"
        )

    def test_speech_endpoint_non_tts_model(self):
        """Test speech endpoint with a model that doesn't support TTS."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS tests require OpenAI backend")

        payload = {
            "model": self.model,  # This is a chat model, not TTS
            "input": "Hello, this is a test.",
            "voice": "alloy",
        }

        response = self.client.post(
            self.url_tts,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code, 400, f"Expected 400 Bad Request, got {response.status_code}"
        )

    @override_settings(RELAY_REQUEST_TIMEOUT=0.1)
    def test_speech_endpoint_timeout(self):
        """Test speech endpoint timeout."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS tests require OpenAI backend")

        payload = {"model": self.tts_model, "input": "Hello, this is a test.", "voice": "alloy"}

        response = self.client.post(
            self.url_tts,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code, 504, f"Expected 504 Gateway Timeout, got {response.status_code}"
        )


class TranscriptionsEndpointTest(GatewayTTSSTTestCase):
    """Test the transcriptions (speech-to-text) endpoint."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(Path(__file__).parent / "resources" / "Eraserhead.mp3", "rb") as f:
            cls.test_audio_content = f.read()

    def setUp(self):
        super().setUp()
        self.test_audio_file = SimpleUploadedFile(
            "Eraserhead.mp3", self.test_audio_content, content_type="audio/mp3"
        )

    def test_transcriptions_endpoint_basic(self):
        """Test basic transcription with valid audio file."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"file": self.test_audio_file, "model": self.stt_model},
            headers=self.multipart_headers,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        self.assertIn("text", response_json, "Response should contain 'text' field")

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after transcription."
        )
        req = requests[0]
        self.assertIn("transcriptions", req.path, "Request endpoint should be for transcriptions.")
        self.assertIsNotNone(req.token_usage)

    def test_transcriptions_endpoint_invalid_model(self):
        """Test transcriptions endpoint with invalid model."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"file": self.test_audio_file, "model": "invalid-stt-model"},
            headers=self.multipart_headers,
        )

        self.assertEqual(
            response.status_code, 404, f"Expected 404 Not Found, got {response.status_code}"
        )
        self.assertIn("Incompatible model", response.json()["error"])

    def test_transcriptions_endpoint_missing_file(self):
        """Test transcriptions endpoint with missing file."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"model": self.stt_model},  # Missing file
            headers=self.multipart_headers,
        )

        self.assertEqual(
            response.status_code, 400, f"Expected 400 Bad Request, got {response.status_code}"
        )

    def test_transcriptions_endpoint_non_stt_model(self):
        """Test transcriptions endpoint with a model that doesn't support STT."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"file": self.test_audio_file, "model": self.model},  # This is a chat model, not STT
            headers=self.multipart_headers,
        )

        self.assertEqual(
            response.status_code, 404, f"Expected 404 Not Found, got {response.status_code}"
        )

    def test_transcriptions_endpoint_with_language(self):
        """Test transcriptions endpoint with language parameter."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"file": self.test_audio_file, "model": self.stt_model, "language": "en"},
            headers=self.multipart_headers,
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        response_json = response.json()
        self.assertIn("text", response_json, "Response should contain 'text' field")

    def test_transcriptions_endpoint_with_response_format(self):
        """Test transcriptions endpoint with response_format parameter."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {
                "file": self.test_audio_file,
                "model": self.stt_model,
                "response_format": "verbose_json",
            },
            headers=self.multipart_headers,
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        response_json = response.json()
        self.assertIn("text", response_json, "Response should contain 'text' field")

    @override_settings(RELAY_REQUEST_TIMEOUT=0.1)
    def test_transcriptions_endpoint_timeout(self):
        """Test transcriptions endpoint timeout."""
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("STT tests require OpenAI backend")

        response = self.client.post(
            self.url_stt,
            {"file": self.test_audio_file, "model": self.stt_model},
            headers=self.multipart_headers,
        )

        self.assertEqual(
            response.status_code, 504, f"Expected 504 Gateway Timeout, got {response.status_code}"
        )


class TTSTSTLifecycleTest(GatewayTTSSTTestCase):
    """Test the complete TTS -> STT lifecycle."""

    async def test_tts_stt_lifecycle(self):
        """
        Lifecycle test: TTS model creates an audio clip, then STT model decodes it.
        This tests the complete round-trip functionality.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS/STT lifecycle tests require OpenAI backend")

        original_text = (
            "Hello, this is a test of the complete text-to-speech and speech-to-text lifecycle."
        )

        # Generate speech using TTS endpoint
        tts_payload = {
            "model": self.tts_model,
            "input": original_text,
            "voice": "alloy",
            "response_format": "mp3",
        }

        tts_response = await self.async_client.post(
            self.url_tts,
            data=json.dumps(tts_payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(
            tts_response.status_code, 200, f"TTS request failed: {tts_response.status_code}"
        )

        # Collect the audio data from streaming response
        audio_data = b""
        async for chunk in tts_response.streaming_content:
            audio_data += chunk

        self.assertGreater(len(audio_data), 0, "TTS should generate audio data")

        # Check TTS request was logged
        tts_requests = await sync_to_async(list)(Request.objects.filter(path__contains="speech"))
        self.assertEqual(len(tts_requests), 1, "There should be exactly one TTS request.")
        tts_request = tts_requests[0]
        self.assertIsNotNone(tts_request.token_usage)

        buffer = io.BytesIO(audio_data)
        buffer.name = "file.mp3"

        stt_response = await self.async_client.post(
            self.url_stt, {"file": buffer, "model": self.stt_model}, headers=self.multipart_headers
        )

        self.assertEqual(
            stt_response.status_code, 200, f"STT request failed: {stt_response.status_code}"
        )

        stt_response_json = stt_response.json()
        self.assertIn("text", stt_response_json, "STT response should contain 'text' field")

        transcribed_text = stt_response_json["text"]
        self.assertIsInstance(transcribed_text, str, "Transcribed text should be a string")
        self.assertGreater(len(transcribed_text), 0, "Transcribed text should not be empty")

        # Check STT request was logged
        stt_requests = await sync_to_async(list)(
            Request.objects.filter(path__contains="transcriptions")
        )
        self.assertEqual(len(stt_requests), 1, "There should be exactly one STT request.")
        stt_request = stt_requests[0]
        self.assertIsNotNone(stt_request.token_usage)

        # Verify the complete lifecycle: total requests should be 2 (TTS + STT)
        total_requests = await sync_to_async(list)(Request.objects.all())
        self.assertEqual(
            len(total_requests), 2, "There should be exactly two requests in the lifecycle."
        )

        # The transcribed text should contain some of the original content
        # (exact match may not be possible due to audio compression/transcription accuracy)
        original_words = set(original_text.lower().split())
        transcribed_words = set(transcribed_text.lower().split())

        # Check that at least some words from the original text appear in the transcription
        common_words = original_words.intersection(transcribed_words)
        self.assertGreater(
            len(common_words),
            0,
            f"Transcription should contain some words from original text. "
            f"Original: {original_words}, Transcribed: {transcribed_words}, Common: {common_words}",
        )

    async def test_tts_stt_lifecycle_with_different_voices(self):
        """
        Test TTS -> STT lifecycle with different voice options.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest("TTS/STT lifecycle tests require OpenAI backend")

        original_text = "Testing different voice options for text-to-speech conversion."
        voices = ["alloy", "echo", "fable"]

        for voice in voices:
            with self.subTest(voice=voice):
                # Clear previous requests
                await Request.objects.all().adelete()

                # Step 1: Generate speech with specific voice
                tts_payload = {
                    "model": self.tts_model,
                    "input": original_text,
                    "voice": voice,
                    "response_format": "mp3",
                }

                tts_response = await self.async_client.post(
                    self.url_tts,
                    data=json.dumps(tts_payload),
                    headers=self.headers,
                    content_type="application/json",
                )

                self.assertEqual(
                    tts_response.status_code,
                    200,
                    f"TTS request failed for voice {voice}: {tts_response.status_code}",
                )

                # Collect audio data
                audio_data = b""
                async for chunk in tts_response.streaming_content:
                    audio_data += chunk

                self.assertGreater(
                    len(audio_data), 0, f"TTS should generate audio data for voice {voice}"
                )

                # Step 2: Transcribe the audio
                audio_file = SimpleUploadedFile(
                    f"generated_speech_{voice}.mp3", audio_data, content_type="audio/mpeg"
                )

                stt_response = await self.async_client.post(
                    self.url_stt,
                    {"file": audio_file, "model": self.stt_model},
                    headers=self.multipart_headers,
                )

                self.assertEqual(
                    stt_response.status_code,
                    200,
                    f"STT request failed for voice {voice}: {stt_response.status_code}",
                )

                stt_response_json = stt_response.json()
                self.assertIn(
                    "text",
                    stt_response_json,
                    f"STT response should contain 'text' field for voice {voice}",
                )

                transcribed_text = stt_response_json["text"]
                self.assertGreater(
                    len(transcribed_text),
                    0,
                    f"Transcribed text should not be empty for voice {voice}",
                )

                # Verify requests were logged
                tts_requests = await sync_to_async(list)(
                    Request.objects.filter(path__contains="speech")
                )
                stt_requests = await sync_to_async(list)(
                    Request.objects.filter(path__contains="transcriptions")
                )
                self.assertEqual(
                    len(tts_requests), 1, f"There should be one TTS request for voice {voice}"
                )
                self.assertEqual(
                    len(stt_requests), 1, f"There should be one STT request for voice {voice}"
                )
