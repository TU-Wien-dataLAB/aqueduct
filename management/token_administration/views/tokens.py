from django.http import HttpResponse

from .base import (BaseAqueductView)


class TokensView(BaseAqueductView):
    def get(self, request, *args, **kwargs):
        return HttpResponse("Hello, world. You're at the tokens index.")