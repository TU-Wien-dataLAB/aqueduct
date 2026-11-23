from datetime import timedelta
from typing import Any

from django.db.models import Count, Q, QuerySet
from django.utils import timezone
from django.views.generic import ListView

from management.models import Request
from management.views.base import BaseAqueductView

STATUSES = {
    "ok": Q(status_code__lt=400),
    "rate_limited": Q(status_code=429),
    "failed": Q(status_code__gte=400) & ~Q(status_code=429),
}
SPANS = {"1h": timedelta(hours=1), "1d": timedelta(days=1), "7d": timedelta(days=7)}
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
        if params.get("key", "").isdigit():
            requests = requests.filter(token_id=params["key"])
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

    def get_queryset(self) -> QuerySet[Request]:
        requests = self.narrowed().select_related("token")
        status = STATUSES.get(self.request.GET.get("status", ""))
        if status is not None:
            requests = requests.filter(status)
        field = SORTS[self.sort.removeprefix("-")]
        return requests.order_by(f"-{field}" if self.sort.startswith("-") else field, "-id")

    def get_context_data(self, **kwargs: Any) -> dict[str, object]:
        params = self.request.GET
        return super().get_context_data(
            keys=self.profile.visible_tokens(self.org).order_by("name"),
            models=self.profile.visible_requests(self.org)
            .order_by("model")
            .values_list("model", flat=True)
            .distinct(),
            counts=self.narrowed().aggregate(
                all=Count("id"), **{name: Count("id", filter=q) for name, q in STATUSES.items()}
            ),
            key=params.get("key", ""),
            status=params["status"] if params.get("status") in STATUSES else "",
            model=params.get("model", ""),
            span=params.get("span", ""),
            query=params.get("q", "").strip(),
            sort=self.sort,
            **kwargs,
        )
