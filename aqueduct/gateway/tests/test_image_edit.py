from http import HTTPStatus

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings

from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request, Usage


class ImageEditEndpointTest(GatewayIntegrationTestCase):
    def test_edit_image(self):
        response = self.client.post(
            "/v1/images/edits",
            {
                "model": "dall-e-2",
                "prompt": "Add a tree",
                "image": SimpleUploadedFile("image.png", b"image bytes", "image/png"),
            },
            headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"},
        )
        self.assertEqual(response.status_code, HTTPStatus.OK, response.content)
        self.assertIn("b64_json", response.json()["data"][0])
        self.assertIsInstance(Request.objects.get().token_usage, Usage)
        sent = self.mock_server.get_last_request("images/edits")
        self.assertEqual(
            sent["image"], [{"filename": "image.png", "content": b"image bytes".hex()}]
        )
        self.assertEqual(sent["response_format"], ["b64_json"])

    def test_multiple_images_and_mask(self):
        response = self.client.post(
            "/images/edits",
            {
                "model": "gpt-image-1",
                "prompt": "Combine these",
                "image[]": [
                    SimpleUploadedFile("one.png", b"first", "image/png"),
                    SimpleUploadedFile("two.png", b"second", "image/png"),
                ],
                "mask": SimpleUploadedFile("mask.png", b"mask", "image/png"),
            },
            headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"},
        )
        self.assertEqual(response.status_code, HTTPStatus.OK, response.content)
        sent = self.mock_server.get_last_request("images/edits")
        self.assertEqual(
            sent["image[]"],
            [
                {"filename": "one.png", "content": b"first".hex()},
                {"filename": "two.png", "content": b"second".hex()},
            ],
        )
        self.assertEqual(sent["mask"], [{"filename": "mask.png", "content": b"mask".hex()}])
        self.assertNotIn("response_format", sent)

    def test_numeric_prompt_is_preserved(self):
        response = self.client.post(
            "/images/edits",
            {"model": "dall-e-2", "prompt": "123", "image": SimpleUploadedFile("a.png", b"a")},
            headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"},
        )
        self.assertEqual(response.status_code, HTTPStatus.OK, response.content)
        self.assertEqual(self.mock_server.get_last_request("images/edits")["prompt"], ["123"])

    def test_conflicting_image_fields_are_rejected(self):
        response = self.client.post(
            "/images/edits",
            {
                "model": "dall-e-2",
                "prompt": "Edit",
                "image": SimpleUploadedFile("a.png", b"a"),
                "image[]": "text",
            },
            headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"},
        )
        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)

    def test_validation_and_authentication(self):
        headers = {"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"}
        for extra in ({"stream": "true"}, {"response_format": "url"}, {"unexpected": "value"}):
            with self.subTest(extra=extra):
                response = self.client.post(
                    "/images/edits",
                    {
                        "model": "dall-e-2",
                        "prompt": "Edit",
                        "image": SimpleUploadedFile("a.png", b"a"),
                        **extra,
                    },
                    headers=headers,
                )
                self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST, response.content)
        response = self.client.post(
            "/images/edits", {"model": "dall-e-2", "prompt": "Edit"}, headers=headers
        )
        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        response = self.client.post("/images/edits", {"model": "dall-e-2", "prompt": "Edit"})
        self.assertEqual(response.status_code, HTTPStatus.UNAUTHORIZED)

    @override_settings(AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB=0.00001)
    def test_total_limit_counts_all_images(self):
        response = self.client.post(
            "/images/edits",
            {
                "model": "gpt-image-1",
                "prompt": "Edit",
                "image[]": [
                    SimpleUploadedFile("a.png", b"123456"),
                    SimpleUploadedFile("b.png", b"123456"),
                ],
            },
            headers={"Authorization": f"Bearer {self.AQUEDUCT_ACCESS_TOKEN}"},
        )
        self.assertEqual(response.status_code, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
