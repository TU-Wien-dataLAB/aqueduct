from datetime import datetime

from django.db.models import Q
from django.views.generic import TemplateView

from ..models import VectorStore
from .base import BaseAqueductView


class UserVectorStoresView(BaseAqueductView, TemplateView):
    """
    Displays a list of VectorStore instances accessible by the current user.
    Shows vector stores from user's personal tokens and team service accounts.
    """

    template_name = "management/vector_stores.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        user = profile.user
        teams = profile.teams.all()

        # Query vector stores accessible by user (personal tokens + team service accounts)
        vector_stores = (
            VectorStore.objects.filter(
                Q(token__user=user) | Q(token__service_account__team__in=teams)
            )
            .prefetch_related("files", "file_batches")
            .order_by("-created_at")
        )

        # Convert Unix timestamps to datetime for template date filter
        # and calculate counts
        for vs in vector_stores:
            vs.created_dt = datetime.fromtimestamp(vs.created_at)
            vs.files_count = vs.files.count()
            vs.batches_count = vs.file_batches.count()

        context["vector_stores"] = vector_stores
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
        teams = profile.teams.all()
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

            # Prepare files with datetime conversion
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
        else:
            context["vector_store"] = None

        return context
