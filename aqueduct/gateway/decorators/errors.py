import logging
import re
from functools import wraps
from typing import Any

import litellm
import openai
from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import RawJsonResponse, error_response

log = logging.getLogger("aqueduct")


def catch_router_exceptions(view_func: AsyncView) -> AsyncView:
    def _r(e: Exception) -> str:
        s = str(e)
        s = re.sub(r"Lite-?[lL][lL][mM]", "Aqueduct", s)  # uppercase
        return re.sub(r"lite-?[lL][lL][mM]", "aqueduct", s)  # lowercase

    def _exception_response(e: Exception, status: int) -> RawJsonResponse:
        """Convert an openai/litellm exception to an OpenAI-compatible JsonResponse.

        The openai SDK parses ``code``, ``param``, and ``type`` from the
        response body when available (see ``openai.APIError.__init__``).
        LiteLLM exceptions inherit from the corresponding openai classes
        but typically pass ``body=None``, so these will be ``None`` for
        most litellm errors.  We forward whatever is available.
        """
        code = getattr(e, "code", None)
        return error_response(
            message=_r(e),
            error_type=getattr(e, "type", None),
            param=getattr(e, "param", None),
            code=str(code) if code is not None else None,
            status=status,
        )

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # https://docs.litellm.ai/docs/exception_mapping#litellm-exceptions
        # also except equivalent openai exceptions
        try:
            return await view_func(request, *args, **kwargs)
        except (litellm.BadRequestError, openai.BadRequestError) as e:
            log.exception("Bad request - %s", _r(e))
            return _exception_response(e, status=400)
        except (litellm.AuthenticationError, openai.AuthenticationError) as e:
            log.exception("Authentication error - %s", _r(e))
            return _exception_response(e, status=401)
        except (litellm.exceptions.PermissionDeniedError, openai.PermissionDeniedError) as e:
            log.exception("Permission denied - %s", _r(e))
            return _exception_response(e, status=403)
        except (litellm.NotFoundError, openai.NotFoundError) as e:
            log.exception("Not found - %s", _r(e))
            return _exception_response(e, status=404)
        except (litellm.UnprocessableEntityError, openai.UnprocessableEntityError) as e:
            log.exception("Unprocessable entity - %s", _r(e))
            return _exception_response(e, status=422)
        except (litellm.RateLimitError, openai.RateLimitError) as e:
            log.exception("Rate limit exceeded - %s", _r(e))
            return _exception_response(e, status=429)
        except (litellm.Timeout, openai.APITimeoutError) as e:
            log.exception("Timeout - %s", _r(e))
            return _exception_response(e, status=504)
        except (
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            openai.APIConnectionError,
        ) as e:
            log.exception("Service unavailable - %s", _r(e))
            return _exception_response(e, status=503)
        except (litellm.InternalServerError, openai.InternalServerError) as e:
            log.exception("Internal server error - %s", _r(e))
            return _exception_response(e, status=500)
        except (litellm.APIError, openai.APIError) as e:
            # APIError is raised e.g. when user sends extra kwargs in the request body,
            # so we return a 400 Bad request.
            log.exception("API error - %s", _r(e))
            return _exception_response(e, status=400)
        except Exception as e:
            log.exception("Unexpected error - %s", _r(e))
            return error_response(_r(e), error_type="server_error", status=502)

    return wrapper
