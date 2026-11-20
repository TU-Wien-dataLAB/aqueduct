import openai
from asgiref.sync import async_to_sync
from django.contrib import messages
from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.http import urlencode
from django.utils.translation import gettext
from django.views.generic import DeleteView, TemplateView

from gateway.config import get_files_api_client
from management.access import KeyAct
from management.deletion import Crumb, deletion
from management.models import BatchStatus, FileObject
from management.resources import file_list, list_url
from management.views.base import BaseAqueductView


class UserFilesView(BaseAqueductView, TemplateView):
    template_name = "management/files.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["listing"] = file_list(
            self.viewer,
            timezone.now(),
            page=self.request.GET.get("page"),
            file_id=self.request.GET.get("file"),
        )
        return context


RUNNING = (
    BatchStatus.VALIDATING,
    BatchStatus.IN_PROGRESS,
    BatchStatus.FINALIZING,
    BatchStatus.CANCELLING,
)


class FileDeleteView(BaseAqueductView, DeleteView):
    pk_url_kwarg = "id"
    template_name = "management/common/confirm_delete.html"

    def get_success_url(self) -> str:
        return list_url("files", self.request.GET.get("page"))

    def get_queryset(self) -> QuerySet[FileObject]:
        return FileObject.objects.filter(token__in=self.viewer.keys(KeyAct.OWN))

    def form_valid(self, form) -> HttpResponse:
        file = self.object
        try:
            client = get_files_api_client()
        except ValueError:
            messages.error(self.request, gettext("The Files API is not configured."))
            return redirect(self.get_success_url())
        if file.batches.filter(status__in=RUNNING).exists():
            messages.error(
                self.request,
                gettext("%(name)s feeds a running batch. Cancel the batch first.")
                % {"name": file.filename},
            )
            return redirect(self.get_success_url())
        try:
            async_to_sync(file.adelete_upstream)(client)
        except openai.NotFoundError:
            pass
        except Exception:
            messages.error(
                self.request,
                gettext("Could not delete %(name)s. Try again later.") % {"name": file.filename},
            )
            return redirect(self.get_success_url())
        file.delete()
        messages.success(self.request, gettext("File %(name)s deleted.") % {"name": file.filename})
        return redirect(self.get_success_url())

    def get_context_data(self, **kwargs) -> dict[str, object]:
        return super().get_context_data(
            confirmation=deletion(self.object),
            cancel_url=self.get_success_url(),
            trail=[
                Crumb(gettext("Files"), self.get_success_url()),
                Crumb(
                    self.object.filename,
                    f"{reverse('files')}?{urlencode({'file': self.object.id})}#{self.object.id}",
                ),
            ],
            **kwargs,
        )
