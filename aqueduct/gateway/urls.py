from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = 'gateway'

urlpatterns = [
    # Redirect to the management views if no direct path is given
    path('', RedirectView.as_view(url='/aqueduct/management/tokens/', permanent=True)),

    # Completions endpoints
    path('completions', views.completions, name='completions'),
    path('v1/completions', views.completions, name='v1_completions'),

    # Chat completions endpoints
    path('chat/completions', views.chat_completions, name='chat_completions'),
    path('v1/chat/completions', views.chat_completions, name='v1_chat_completions'),

    # Embeddings endpoints
    path('embeddings', views.embeddings, name='embeddings'),
    path('v1/embeddings', views.embeddings, name='v1_embeddings'),

    # Models endpoints
    path('models', views.models, name='models'),
    path('v1/models', views.models, name='v1_models'),

    # Files endpoints
    path('files', views.files, name='files'),
    path('v1/files', views.files, name='v1_files'),
    path('files/<str:file_id>', views.file, name='file'),
    path('v1/files/<str:file_id>', views.file, name='v1_file'),
    path('files/<str:file_id>/content', views.file_content, name='file_content'),
    path('v1/files/<str:file_id>/content', views.file_content, name='v1_file_content'),
]
