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