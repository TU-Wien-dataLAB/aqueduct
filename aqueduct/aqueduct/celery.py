import asyncio
import logging
import os
from datetime import timedelta

from celery import Celery
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("aqueduct")

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aqueduct.settings")

app = Celery("aqueduct")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def delete_old_requests(self: object) -> None:
    """
    Deletes Request objects older than REQUEST_RETENTION_DAYS.
    """
    from management.models import Request  # Import here to avoid issues at startup

    retention_days = getattr(settings, "REQUEST_RETENTION_DAYS", 30)
    cutoff = timezone.now() - timedelta(days=retention_days)
    old_requests = Request.objects.filter(timestamp__lt=cutoff)
    count = old_requests.count()
    old_requests.delete()
    logger.info("Deleted %s requests older than %s days (before %s)", count, retention_days, cutoff)


@app.task(bind=True, ignore_result=True)
def delete_silk_models(self: object) -> None:
    """
    Clears Silk's profiling logs.
    """
    import silk.models  # type: ignore[import-untyped]
    from silk.utils.data_deletion import delete_model  # type: ignore[import-untyped]

    delete_model(silk.models.Profile)
    delete_model(silk.models.SQLQuery)
    delete_model(silk.models.Response)
    delete_model(silk.models.Request)
    logger.info("Cleared Silk profiling logs.")


@app.task(bind=True, ignore_result=True)
def delete_expired_files_and_batches(self: object) -> None:
    """
    Deletes expired FileObject and Batch records.

    For files with a remote_id, attempts to delete the file from the upstream
    provider before deleting the local database record. If upstream deletion
    fails (e.g., file already deleted, network error), the local record is
    still deleted to prevent infinite retry loops.
    """
    from gateway.config import get_files_api_client
    from management.models import Batch, FileObject

    now_ts = int(timezone.now().timestamp())
    try:
        client = get_files_api_client()
        files_api_configured = True
    except ValueError:
        client = None
        files_api_configured = False
        logger.warning("Files API not configured - will skip upstream deletion")

    # Expired files
    files_qs = FileObject.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    files_list = list(files_qs)
    files_count = len(files_list)
    upstream_deleted = 0

    for file_obj in files_list:
        try:
            # Attempt upstream deletion first
            if files_api_configured and client:
                success = asyncio.run(file_obj.adelete_upstream(client, raise_on_error=False))
                if success:
                    upstream_deleted += 1
            # Always delete local record
            file_obj.delete()
        except Exception as e:
            logger.warning("Failed to delete expired file %s: %s", file_obj.id, e)

    logger.info("Deleted %s expired files (%s from upstream).", files_count, upstream_deleted)

    # Expired batches - just delete local records
    # (batch resources are managed by the upstream provider)
    batches_qs = Batch.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    batches_count, _ = batches_qs.delete()
    logger.info("Deleted %s expired batches.", batches_count)
