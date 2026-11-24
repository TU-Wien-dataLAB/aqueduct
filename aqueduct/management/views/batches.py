import logging
from datetime import UTC, datetime

import openai
from asgiref.sync import async_to_sync
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.views.generic import TemplateView, View

from management.models import Batch
from management.views.base import BaseAqueductView

INPUT_FILE_ID_PREVIEW_THRESHOLD = 12

log = logging.getLogger("aqueduct")


class UserBatchesView(BaseAqueductView, TemplateView):
    """
    Displays a list of Batch instances accessible by the current user.
    """

    template_name = "management/batches.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile

        batches = Batch.objects.filter(token__in=profile.my_tokens()).order_by("-created_at")

        # Convert Unix timestamp to datetime for template date filter
        for b in batches:
            b.created_dt = datetime.fromtimestamp(b.created_at, tz=UTC)
            # input file preview (first lines)
            b.input_file_preview = b.input_file.preview or ""
            # short preview of file id for display
            raw_id = b.input_file.id
            if len(raw_id) > INPUT_FILE_ID_PREVIEW_THRESHOLD:
                b.input_file_id_preview = f"{raw_id[:7]}...{raw_id[-5:]}"
            else:
                b.input_file_id_preview = raw_id
            # output file preview
            if b.output_file:
                b.output_file_preview = b.output_file.preview or ""
                b.output_file_has_preview = bool(b.output_file_preview)
                b.output_file_id_preview = b.output_file.id
            else:
                b.output_file_preview = ""
                b.output_file_has_preview = False
                b.output_file_id_preview = "-"
            # error file preview
            if b.error_file:
                b.error_file_preview = b.error_file.preview or ""
                b.error_file_has_preview = bool(b.error_file_preview)
                b.error_file_id_preview = b.error_file.id
            else:
                b.error_file_preview = ""
                b.error_file_has_preview = False
                b.error_file_id_preview = "-"
            b.expires_dt = b.expires_at and datetime.fromtimestamp(b.expires_at, tz=UTC)

        context["batches"] = batches
        context["manageable_token_ids"] = set(
            profile.manageable_tokens().values_list("id", flat=True)
        )
        return context


class BatchCancelView(BaseAqueductView, View):
    """Cancels a running batch the page lists, upstream, and keeps the status it reports."""

    def post(self, request, *args, **kwargs) -> HttpResponse:
        batch = get_object_or_404(
            Batch.objects.filter(
                token__in=self.profile.manageable_tokens(), status__in=Batch.RUNNING
            ),
            pk=kwargs["id"],
        )
        try:
            async_to_sync(batch.acancel_upstream)()
        except (ValueError, openai.OpenAIError):
            log.exception("Could not cancel batch %s upstream", batch.id)
            messages.error(request, f"Could not cancel batch {batch.id}. Try again later.")
            return redirect("batches")
        messages.success(request, f"Batch {batch.id} is being cancelled.")
        return redirect("batches")
