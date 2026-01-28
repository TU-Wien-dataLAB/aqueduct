import asyncio
import os

from celery import Celery
from django.conf import settings
from django.utils import timezone

from gateway.config import get_files_api_client
from management.models import Batch, FileObject

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
def delete_old_requests(self):
    """
    Deletes Request objects older than REQUEST_RETENTION_DAYS.
    """
    from management.models import Request  # Import here to avoid issues at startup

    retention_days = getattr(settings, "REQUEST_RETENTION_DAYS", 30)
    cutoff = timezone.now() - timezone.timedelta(days=retention_days)
    old_requests = Request.objects.filter(timestamp__lt=cutoff)
    count = old_requests.count()
    old_requests.delete()
    print(f"Deleted {count} requests older than {retention_days} days (before {cutoff})")


@app.task(bind=True, ignore_result=True)
def delete_silk_models(self):
    """
    Clears Silk's profiling logs.
    """
    import silk.models
    from silk.utils.data_deletion import delete_model

    delete_model(silk.models.Profile)
    delete_model(silk.models.SQLQuery)
    delete_model(silk.models.Response)
    delete_model(silk.models.Request)
    print("Cleared Silk profiling logs.")


@app.task(bind=True, ignore_result=True)
def delete_expired_files_and_batches(self):
    """
    Deletes expired FileObject and Batch records.

    For files with a remote_id, attempts to delete the file from the upstream
    provider before deleting the local database record. If upstream deletion
    fails (e.g., file already deleted, network error), the local record is
    still deleted to prevent infinite retry loops.
    """
    now_ts = int(timezone.now().timestamp())
    client = get_files_api_client()

    async def delete_upstream_file(remote_id: str) -> bool:
        """Attempt to delete a file from upstream. Returns True on success."""
        if not client or not remote_id:
            return True  # Nothing to delete upstream
        try:
            await client.files.delete(remote_id)
            return True
        except Exception as e:
            # Log but don't fail - file may already be deleted upstream
            print(f"Failed to delete upstream file {remote_id}: {e}")
            return False

    # Expired files
    files_qs = FileObject.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    files_list = list(files_qs)
    files_count = len(files_list)
    upstream_deleted = 0

    for file_obj in files_list:
        try:
            # Attempt upstream deletion first
            if file_obj.remote_id:
                success = asyncio.run(delete_upstream_file(file_obj.remote_id))
                if success:
                    upstream_deleted += 1
            # Always delete local record
            file_obj.delete()
        except Exception as e:
            print(f"Failed to delete expired file {file_obj.id}: {e}")

    print(f"Deleted {files_count} expired files ({upstream_deleted} from upstream).")

    # Expired batches - just delete local records
    # (batch resources are managed by the upstream provider)
    batches_qs = Batch.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    batches_count = batches_qs.count()
    batches_qs.delete()
    print(f"Deleted {batches_count} expired batches.")
