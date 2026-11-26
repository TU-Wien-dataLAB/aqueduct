from datetime import timedelta
from functools import cached_property
from typing import Any

from django.core.paginator import Paginator
from django.db.models import Count, Q, QuerySet
from django.utils import timezone
from django.views.generic import ListView

from gateway.config import get_router_config
from management.models import Request
from management.views.base import BaseAqueductView

STATUSES = {
    "ok": Q(status_code__lt=400),
    "rate_limited": Q(status_code=429),
    "failed": Q(status_code__gte=400) & ~Q(status_code=429),
    "client_error": Q(status_code__gte=400, status_code__lt=500) & ~Q(status_code=429),
    "server_error": Q(status_code__gte=500),
}
"""The status filters; the log's tabs show the first three, the Usage page links to all."""
SPANS = {
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "1w": timedelta(days=7),
    "1m": timedelta(days=30),
}
"""The time frames of the log and the Usage page, and their lengths."""
SORTS = {
    "time": "timestamp",
    "model": "model",
    "status": "status_code",
    "input": "input_tokens",
    "output": "output_tokens",
    "response": "response_time_ms",
}
"""The columns the log sorts by, and their fields."""


class RequestLogView(BaseAqueductView, ListView):
    """The requests the viewer may see, filtered by key, model, time, text and status."""

    template_name = "management/requests.html"
    context_object_name = "requests"
    paginate_by = 50

    def narrowed(self) -> QuerySet[Request]:
        """The visible requests under every filter except status, which the tabs count."""
        params = self.request.GET
        requests = self.profile.visible_requests(self.org)
        if key := params.get("key", ""):
            # A key that is not an id, like "²" which isdigit() accepts, matches nothing.
            numeric = key.isascii() and key.isdigit()
            requests = requests.filter(token_id=key) if numeric else requests.none()
        if params.get("model"):
            requests = requests.filter(model=params["model"])
        if params.get("span") in SPANS:
            requests = requests.filter(timestamp__gte=timezone.now() - SPANS[params["span"]])
        if query := params.get("q", "").strip():
            requests = requests.filter(Q(model__icontains=query) | Q(path__icontains=query))
        return requests

    @property
    def sort(self) -> str:
        sort = self.request.GET.get("sort", "")
        return sort if sort.removeprefix("-") in SORTS else "-time"

    @property
    def status(self) -> str:
        status = self.request.GET.get("status", "")
        return status if status in STATUSES else ""

    @cached_property
    def counts(self) -> dict[str, int]:
        """The rows of each status tab, and of all of them."""
        return self.narrowed().aggregate(
            all=Count("id"), **{name: Count("id", filter=q) for name, q in STATUSES.items()}
        )

    def get_queryset(self) -> QuerySet[Request]:
        requests = self.narrowed().select_related("token")
        if self.status:
            requests = requests.filter(STATUSES[self.status])
        field = SORTS[self.sort.removeprefix("-")]
        return requests.order_by(f"-{field}" if self.sort.startswith("-") else field, "-id")

    def get_paginator(self, *args: Any, **kwargs: Any) -> Paginator:
        """A paginator that takes its count from the status tab shown, which counted the rows."""
        paginator = super().get_paginator(*args, **kwargs)
        paginator.count = self.counts[self.status or "all"]
        return paginator

    def get_context_data(self, **kwargs: Any) -> dict[str, object]:
        params = self.request.GET
        # The router's models, as on the Usage page: reading them from the requests is a pass
        # over all of them. A model gone from the router stays listed while a link selects it.
        models = [model["model_name"] for model in get_router_config().get("model_list", [])]
        if params.get("model") and params["model"] not in models:
            models.append(params["model"])
        return super().get_context_data(
            keys=self.profile.visible_tokens(self.org).order_by("name"),
            models=models,
            counts=self.counts,
            key=params.get("key", ""),
            status=self.status,
            model=params.get("model", ""),
            span=params.get("span", ""),
            query=params.get("q", "").strip(),
            sort=self.sort,
            **kwargs,
        )
