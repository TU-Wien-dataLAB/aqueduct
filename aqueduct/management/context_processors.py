from django.http import HttpRequest
from django.utils.functional import SimpleLazyObject

from management.nav import build_shell


def shell(request: HttpRequest) -> dict[str, object]:
    return {"shell": SimpleLazyObject(lambda: build_shell(request))}
