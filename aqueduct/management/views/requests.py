from django.db.models import Q, QuerySet
from django.views.generic import ListView

from management.models import Request
from management.views.base import BaseAqueductView

STATUSES = {
    "ok": Q(status_code__lt=400),
    "rate_limited": Q(status_code=429),
    "failed": Q(status_code__gte=400) & ~Q(status_code=429),
}


class RequestLogView(BaseAqueductView, ListView):
    """The requests the viewer may see, newest first, filtered by key and status."""

    template_name = "management/requests.html"
    context_object_name = "requests"
    paginate_by = 50

    def get_queryset(self) -> QuerySet[Request]:
        requests = self.profile.visible_requests(self.org).select_related("token")
        key = self.request.GET.get("key", "")
        if key.isdigit():
            requests = requests.filter(token_id=key)
        status = STATUSES.get(self.request.GET.get("status", ""))
        if status is not None:
            requests = requests.filter(status)
        return requests.order_by("-timestamp", "-id")

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(
            keys=self.profile.visible_tokens(self.org).order_by("name"),
            key=self.request.GET.get("key", ""),
            status=self.request.GET.get("status", ""),
            **kwargs,
        )
