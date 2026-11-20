import logging

from asgiref.sync import async_to_sync
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext
from django.views import View
from django.views.generic import TemplateView

from management.access import KeyAct
from management.models import VectorStore
from management.resources import store_detail, store_list
from management.views.base import BaseAqueductView

log = logging.getLogger(__name__)


class UserVectorStoresView(BaseAqueductView, TemplateView):
    template_name = "management/vector_stores.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["listing"] = store_list(self.viewer, timezone.now(), self.request.GET.get("page"))
        return context


class VectorStoreDetailView(BaseAqueductView, TemplateView):
    template_name = "management/vector_store_detail.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["detail"] = store_detail(
            self.kwargs["id"], self.viewer, timezone.now(), self.request.GET.get("page")
        )
        return context


def _refresh_vector_store(vs) -> bool:
    """
    Refresh a single VectorStore from upstream.
    Calls areload_from_upstream() with raise_on_error=False.
    Returns True if the refresh succeeded, False otherwise.
    """
    try:
        result = async_to_sync(vs.areload_from_upstream)(raise_on_error=False)
    except Exception:
        log.exception("Failed to refresh vector store %s from upstream", vs.id)
        return False
    else:
        return result is not None


class VectorStoreCardRefreshView(BaseAqueductView, View):
    """
    POST-only view that refreshes a single vector store from the upstream API,
    then redirects back to the vector stores list page.
    """

    def post(self, request, *args, **kwargs) -> HttpResponse:
        vector_store_id = self.kwargs.get("id")
        vs = get_object_or_404(
            VectorStore.objects.filter(token__in=self.viewer.keys(KeyAct.SYNC)), id=vector_store_id
        )

        if _refresh_vector_store(vs):
            messages.success(request, gettext("Vector store %(name)s synced.") % {"name": vs.name})
        else:
            messages.warning(
                request,
                gettext("Could not sync vector store %(name)s. Try again later.")
                % {"name": vs.name},
            )

        return redirect(reverse("vector_stores"))


class VectorStoreDetailRefreshView(BaseAqueductView):
    """
    POST-only view that refreshes a specific vector store and all its files
    from the upstream API, then redirects back to the detail page.
    """

    def post(self, request, *args, **kwargs) -> HttpResponse:
        vector_store_id = self.kwargs.get("id")
        vs = get_object_or_404(
            VectorStore.objects.filter(token__in=self.viewer.keys(KeyAct.SYNC)), id=vector_store_id
        )

        if _refresh_vector_store(vs):
            messages.success(request, gettext("Vector store %(name)s synced.") % {"name": vs.name})
        else:
            messages.warning(
                request,
                gettext("Could not sync vector store %(name)s. Try again later.")
                % {"name": vs.name},
            )

        return redirect(reverse("vector_store_detail", kwargs={"id": vector_store_id}))
