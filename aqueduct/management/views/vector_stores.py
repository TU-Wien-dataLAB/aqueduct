import logging
from datetime import datetime

from asgiref.sync import async_to_sync
from django.conf import settings
from django.contrib import messages
from django.db.models import Q
from django.shortcuts import redirect
from django.urls import reverse
from django.views import View
from django.views.generic import TemplateView

from ..models import VectorStore
from .base import BaseAqueductView

log = logging.getLogger(__name__)


class UserVectorStoresView(BaseAqueductView, TemplateView):
    """
    Displays a list of VectorStore instances accessible by the current user.
    Shows vector stores from user's personal tokens and team service accounts.
    """

    template_name = "management/vector_stores.html"

    def _annotate_vector_stores(self, vector_stores):
        """Add computed fields to vector store instances for template rendering."""
        result = []
        for vs in vector_stores:
            vs.created_dt = datetime.fromtimestamp(vs.created_at)
            vs.files_count = vs.files.count()
            vs.batches_count = vs.file_batches.count()
            vs.size_bytes = sum(
                vf.file_obj.bytes for vf in vs.files.select_related("file_obj").all() if vf.file_obj
            )
            result.append(vs)
        return result

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = self.get_teams_for_user()

        base_qs = VectorStore.objects.prefetch_related("files", "file_batches")

        # Personal vector stores (created via user tokens)
        user_vector_stores = base_qs.filter(token__user=user).order_by("-created_at")
        user_vector_stores = self._annotate_vector_stores(user_vector_stores)

        # Per-team vector stores (created via service account tokens)
        team_sections = []
        for team in teams:
            team_vs = base_qs.filter(token__service_account__team=team).order_by("-created_at")
            team_vs = self._annotate_vector_stores(team_vs)
            team_sections.append(
                {
                    "team": team,
                    "vector_stores": team_vs,
                    "count": len(team_vs),
                    "limit": settings.MAX_TEAM_VECTOR_STORES,
                }
            )

        context["user_vector_stores"] = user_vector_stores
        context["user_vector_store_count"] = len(user_vector_stores)
        context["user_vector_store_limit"] = settings.MAX_USER_VECTOR_STORES
        context["team_sections"] = team_sections
        context["files_limit"] = settings.MAX_VECTOR_STORE_FILES
        return context


class VectorStoreDetailView(BaseAqueductView, TemplateView):
    """
    Displays detail view for a specific VectorStore instance.
    Shows files table and batch cards with progress information.
    """

    template_name = "management/vector_store_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = self.get_teams_for_user()
        vector_store_id = self.kwargs.get("id")

        # Get vector store accessible by user
        try:
            vector_store = VectorStore.objects.prefetch_related(
                "files__file_obj", "file_batches"
            ).get(
                Q(id=vector_store_id)
                & (Q(token__user=user) | Q(token__service_account__team__in=teams))
            )
        except VectorStore.DoesNotExist:
            vector_store = None

        if vector_store:
            # Convert timestamps to datetime
            vector_store.created_dt = datetime.fromtimestamp(vector_store.created_at)
            if vector_store.last_active_at:
                vector_store.last_active_dt = datetime.fromtimestamp(vector_store.last_active_at)

            # Prepare files with datetime conversion and calculate total size
            files = []
            total_size = 0
            for vf in vector_store.files.all():
                vf.created_dt = datetime.fromtimestamp(vf.created_at)
                if vf.file_obj:
                    total_size += vf.file_obj.bytes
                files.append(vf)
            vector_store.size_bytes = total_size
            files = []
            for vf in vector_store.files.all():
                vf.created_dt = datetime.fromtimestamp(vf.created_at)
                files.append(vf)

            # Prepare batches with progress calculation
            batches = []
            for batch in vector_store.file_batches.all():
                batch.created_dt = datetime.fromtimestamp(batch.created_at)
                file_counts = batch.file_counts or {}
                total = file_counts.get("total", 0)
                completed = file_counts.get("completed", 0)
                failed = file_counts.get("failed", 0)
                in_progress = file_counts.get("in_progress", 0)

                batch.total_files = total
                batch.completed_files = completed
                batch.failed_files = failed
                batch.in_progress_files = in_progress

                # Calculate progress percentage
                if total > 0:
                    batch.progress_pct = int(((completed + failed) / total) * 100)
                else:
                    batch.progress_pct = 0

                batches.append(batch)

            context["vector_store"] = vector_store
            context["files"] = files
            context["batches"] = batches
            context["files_limit"] = settings.MAX_VECTOR_STORE_FILES
        else:
            context["vector_store"] = None

        return context


def _refresh_vector_store(vs):
    """
    Refresh a single VectorStore from upstream.
    Calls areload_from_upstream() with raise_on_error=False.
    Returns True if the refresh succeeded, False otherwise.
    """
    try:
        result = async_to_sync(vs.areload_from_upstream)(raise_on_error=False)
        return result is not None
    except Exception:
        log.exception(f"Failed to refresh vector store {vs.id} from upstream")
        return False


class VectorStoreCardRefreshView(BaseAqueductView, View):
    """
    POST-only view that refreshes a single vector store from the upstream API,
    then redirects back to the vector stores list page.
    """

    def post(self, request, *args, **kwargs):
        user = self.profile.user
        teams = self.get_teams_for_user()
        vector_store_id = self.kwargs.get("id")

        try:
            vs = VectorStore.objects.get(
                Q(id=vector_store_id)
                & (Q(token__user=user) | Q(token__service_account__team__in=teams))
            )
        except VectorStore.DoesNotExist:
            messages.error(request, "Vector store not found or you do not have access.")
            return redirect(reverse("vector_stores"))

        if _refresh_vector_store(vs):
            messages.success(request, f'Synced "{vs.name}" from upstream.')
        else:
            messages.warning(request, f'Failed to sync "{vs.name}" from upstream.')

        return redirect(reverse("vector_stores"))


class VectorStoreDetailRefreshView(BaseAqueductView, View):
    """
    POST-only view that refreshes a specific vector store and all its files
    from the upstream API, then redirects back to the detail page.
    """

    def post(self, request, *args, **kwargs):
        user = self.profile.user
        teams = self.get_teams_for_user()
        vector_store_id = self.kwargs.get("id")

        try:
            vs = VectorStore.objects.get(
                Q(id=vector_store_id)
                & (Q(token__user=user) | Q(token__service_account__team__in=teams))
            )
        except VectorStore.DoesNotExist:
            messages.error(request, "Vector store not found or you do not have access.")
            return redirect(reverse("vector_stores"))

        if _refresh_vector_store(vs):
            messages.success(request, f'Synced "{vs.name}" from upstream.')
        else:
            messages.warning(request, f'Failed to sync "{vs.name}" from upstream.')

        return redirect(reverse("vector_store_detail", kwargs={"id": vector_store_id}))
