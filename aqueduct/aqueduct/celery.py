import os

from celery import Celery
from celery.schedules import crontab
from django.conf import settings

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


@app.on_after_configure.connect
def setup_periodic_tasks(sender: Celery, **kwargs):
    # Executes on the specified time in settings
    sender.add_periodic_task(
        crontab.from_string(settings.REQUEST_RETENTION_SCHEDULE),
        debug_task.s(),
    )


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Deleting requests older than {settings.REQUEST_RETENTION_DAYS} days!')