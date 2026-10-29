"""Public-model RPM admission using Django's shared cache, independent of routing SDKs."""

import hashlib
import math

from django.core.cache import cache
from django.utils import timezone


def check_model_limit(model: str, rpm: int) -> tuple[bool, int]:
    """Reserve one request in a fixed UTC minute; return (allowed, retry_after).

    Production requires a shared cache with atomic add/incr (Redis/Valkey).
    Count admissions, including upstream failures, not tokens or retry attempts.
    Cache failures propagate to the gateway: do not silently bypass a global quota.
    """
    now = timezone.now().timestamp()
    digest = hashlib.sha256(model.encode()).hexdigest()
    key = f"rl:model:{digest}:{int(now // 60)}"
    cache.add(key, 0, timeout=120)
    count = cache.incr(key)
    return count <= rpm, math.ceil(60 - now % 60)
