import os

from celery import Celery
from celery.schedules import crontab
from django.conf import settings
from django.utils import timezone

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aqueduct.settings')

app = Celery('aqueduct')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

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
    Deletes expired FileObject and Batch records (and associated on-disk files) whose expires_at timestamp is past.
    """
    from django.utils import timezone
    from management.models import FileObject, Batch

    now_ts = int(timezone.now().timestamp())
    # Expired files
    files_qs = FileObject.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    files_list = list(files_qs)
    files_count = len(files_list)
    for file_obj in files_list:
        try:
            file_obj.delete()
        except Exception:
            print(f"Failed to delete expired file {file_obj.id}")
            pass
    print(f"Deleted {files_count} expired files (expires before {now_ts}).")

    # Expired batches
    batches_qs = Batch.objects.filter(expires_at__isnull=False, expires_at__lt=now_ts)
    batches_list = list(batches_qs)
    batches_count = len(batches_list)
    for batch_obj in batches_list:
        try:
            batch_obj.delete()
        except Exception:
            print(f"Failed to delete expired batch {batch_obj.id}")
            pass
    print(f"Deleted {batches_count} expired batches (expires before {now_ts}).")


@app.task(bind=True, ignore_result=True)
def process_batches(self):
    """
    Periodic task to process pending batches via run_batch_processing.
    """
    # Import here to avoid startup-time issues
    import asyncio
    from gateway.views.batches import run_batch_processing

    # Execute the async batch runner
    asyncio.run(run_batch_processing())
