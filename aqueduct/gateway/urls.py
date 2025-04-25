from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = 'gateway'

urlpatterns = [
    # Redirect to the management views if no direct path is given
    path('', RedirectView.as_view(url='/aqueduct/management/tokens/', permanent=True)),

    # Use endpoint slug to determine the target endpoint
    path(
        '<slug:endpoint_slug>/<path:remaining_path>', # Capture endpoint slug and remaining path
        views.ai_gateway_view, # Point to the base view
        name='api_gateway' # More generic name
    ),
]
