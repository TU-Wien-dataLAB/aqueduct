import logging
from typing import Any

from django.conf import settings
from django.core.cache import caches

log = logging.getLogger("aqueduct")


def register_response_in_cache(response_id: str | None, model: str, email: str) -> None:
    """Registers a response in the cache for later retrieval."""
    if not response_id:
        log.warning("Missing response data: id=%s, model=%s", response_id, model)
        raise ValueError("Missing response_id")

    cache_key = f"response:{response_id}"
    cache_value = {"model": model, "email": email}

    response_cache = caches["default"]
    response_cache.set(cache_key, cache_value, timeout=settings.RESPONSES_API_TTL_SECONDS)
    log.debug("Registered response %s for user %s with model %s", response_id, email, model)


def get_response_from_cache(response_id: str) -> dict[str, Any] | None:
    """Retrieves a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    result: dict[str, Any] | None = response_cache.get(cache_key)
    return result


def delete_response_from_cache(response_id: str) -> None:
    """Deletes a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    response_cache.delete(cache_key)
