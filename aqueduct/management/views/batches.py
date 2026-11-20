from typing import ClassVar

from asgiref.sync import async_to_sync
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone
from django.utils.translation import gettext
from django.views import View
from django.views.generic import TemplateView
from openai import OpenAIError

from gateway.config import get_files_api_client
from management.access import KeyAct
from management.models import Batch
from management.resources import BATCH_STATUSES, CANCELLABLE, Status, batch_list, list_url
from management.views.base import BaseAqueductView


class UserBatchesView(BaseAqueductView, TemplateView):
    template_name = "management/batches.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["listing"] = batch_list(self.viewer, timezone.now(), self.request.GET.get("page"))
        return context


class BatchCancelView(BaseAqueductView, View):
    http_method_names: ClassVar[list] = ["post"]

    def post(self, request, *args, **kwargs) -> HttpResponse:
        batch = get_object_or_404(
            Batch.objects.filter(token__in=self.viewer.keys(KeyAct.OWN)), pk=kwargs["id"]
        )
        back = list_url("batches", request.POST.get("page"))
        if batch.status not in CANCELLABLE:
            messages.error(
                request,
                gettext("Batch %(id)s is %(status)s and cannot be cancelled.")
                % {
                    "id": batch.id,
                    "status": str(Status.of(batch.status, BATCH_STATUSES).label).lower(),
                },
            )
            return redirect(back)
        try:
            async_to_sync(batch.acancel_upstream)(get_files_api_client())
        except (ValueError, OpenAIError):
            messages.error(
                request,
                gettext("Could not cancel batch %(id)s. Try again later.") % {"id": batch.id},
            )
        else:
            messages.success(
                request, gettext("Batch %(id)s is being cancelled.") % {"id": batch.id}
            )
        return redirect(back)
