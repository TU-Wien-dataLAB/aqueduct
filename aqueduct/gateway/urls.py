from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = 'gateway'

urlpatterns = [
    # Redirect to the management views if no direct path is given
    path('', RedirectView.as_view(url='/aqueduct/management/tokens/', permanent=True)),

    path(
        'v1/<path:remaining_path>',
        views.V1OpenAIGateway.as_view(),  # This now points to the subclass
        name='openai_v1_gateway'
    ),
]
