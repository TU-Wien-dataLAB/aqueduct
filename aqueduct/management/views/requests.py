from typing import Any

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils import timezone
from django.views.generic import TemplateView

from management.models import UserProfile
from management.org_choice import viewer_of
from management.periods import Span
from management.request_filters import read_filters
from management.request_log import request_log


class RequestLogView(LoginRequiredMixin, TemplateView):
    template_name = "management/requests.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        viewer = viewer_of(self.request)
        if viewer is None:
            raise UserProfile.DoesNotExist
        params = self.request.GET
        filters = read_filters(viewer, params, default_span=Span.DAY, now=timezone.now())
        log = request_log(filters, params)
        context.update(
            log=log,
            filters=filters,
            status=log.status,
            narrowed=log.status is not None
            or bool(log.search)
            or filters.span is not Span.DAY
            or bool(filters.mine or filters.key or filters.model),
            retention_days=int(settings.REQUEST_RETENTION_DAYS),
        )
        return context
