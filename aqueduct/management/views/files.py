import logging
from datetime import UTC, datetime

import openai
from asgiref.sync import async_to_sync
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.views.generic import TemplateView, View

from gateway.config import get_files_api_client
from management.models import Batch, FileObject
from management.views.base import BaseAqueductView

log = logging.getLogger("aqueduct")


class UserFilesView(BaseAqueductView, TemplateView):
    """
    Displays a list of FileObject instances accessible by the current user.
    """

    template_name = "management/files.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile

        files = FileObject.objects.filter(token__in=profile.my_tokens()).order_by("-created_at")

        for f in files:
            f.created_dt = datetime.fromtimestamp(f.created_at, tz=UTC)
            f.expires_dt = f.expires_at and datetime.fromtimestamp(f.expires_at, tz=UTC)

        context["files"] = files
        context["manageable_token_ids"] = set(
            profile.manageable_tokens().values_list("id", flat=True)
        )
        return context


class FileDeleteView(BaseAqueductView, View):
    """Deletes a file the page lists, upstream first and then the local record."""

    def post(self, request, *args, **kwargs) -> HttpResponse:
        file = get_object_or_404(
            FileObject.objects.filter(token__in=self.profile.manageable_tokens()), pk=kwargs["id"]
        )
        running = ("validating", "in_progress", "finalizing")
        if Batch.objects.filter(input_file=file, status__in=running).exists():
            messages.error(request, f"{file.filename} feeds a running batch. Cancel it first.")
            return redirect("files")
        try:
            async_to_sync(file.adelete_upstream)(get_files_api_client())
        except openai.NotFoundError:
            pass  # Already gone upstream: drop the local record too.
        except (ValueError, openai.OpenAIError):
            log.exception("Could not delete file %s upstream", file.id)
            messages.error(request, f"Could not delete {file.filename}. Try again later.")
            return redirect("files")
        file.delete()
        messages.success(request, f"{file.filename} deleted.")
        return redirect("files")
