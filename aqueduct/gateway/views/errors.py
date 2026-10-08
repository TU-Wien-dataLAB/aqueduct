"""OpenAI-compatible error responses.

The implementation now lives in ``gateway.response_error`` so it can be shared with the
``gateway.decorators`` package without a circular import. Re-exported here to
keep existing ``gateway.views.errors`` imports working.
"""

from gateway.response_error import error_response

__all__ = ["error_response"]
