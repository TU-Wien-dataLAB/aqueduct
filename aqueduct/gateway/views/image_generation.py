import litellm
import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm import BadRequestError
from openai.types import ImagesResponse
from pydantic import TypeAdapter

from gateway.config import get_openai_client, get_router
from management.models import Request

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
)
from .utils import _get_token_usage


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@check_limits
@parse_body(model=TypeAdapter(openai.types.ImageGenerateParams))
@log_request
@resolve_alias
@check_model_availability
@catch_router_exceptions
async def image_generation(
    request: ASGIRequest,
    pydantic_model: openai.types.ImageGenerateParams,
    request_log: Request,
    *args,
    **kwargs,
):
    if pydantic_model.get("stream"):
        # LiteLLM cannot parse a Stream response, so we don't support streaming for now
        raise BadRequestError(
            "Aqueduct does not support image streaming.",
            pydantic_model.get("model"),
            llm_provider=None,
        )

    response_format = pydantic_model.get("response_format", "b64_json")
    if response_format != "b64_json":
        raise BadRequestError(
            "Only b64_json response format is supported for image generation.",
            pydantic_model.get("model"),
            llm_provider=None,
        )
    # update if it was None
    pydantic_model["response_format"] = response_format

    model: str = pydantic_model.get("model", "unknown")
    try:
        client: openai.AsyncClient = get_openai_client(model)
    except ValueError:
        return JsonResponse({"error": f"Incompatible model '{model}'!"}, status=400)

    router = get_router()
    deployment: litellm.Deployment = router.get_deployment(model_id=model)

    model_relay, provider, _, _ = litellm.get_llm_provider(deployment.litellm_params.model)
    pydantic_model["model"] = model_relay

    resp: ImagesResponse = await client.images.generate(**pydantic_model)
    data = resp.model_dump(exclude_unset=True)
    request_log.token_usage = _get_token_usage(data)

    return JsonResponse(data)
