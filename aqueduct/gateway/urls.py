from django.urls import path
from django.views.generic import RedirectView

from . import views
from .views import mcp as mcp_views

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

    # Speech endpoint
    path('audio/speech', views.speech, name='speech'),
    path('v1/audio/speech', views.speech, name='v1_speech'),

    # Transcriptions endpoint
    path('audio/transcriptions', views.transcriptions, name='transcriptions'),
    path('v1/audio/transcriptions', views.transcriptions, name='v1_transcriptions'),

    # Files endpoints
    path('files', views.files, name='files'),
    path('v1/files', views.files, name='v1_files'),
    path('files/<str:file_id>', views.file, name='file'),
    path('v1/files/<str:file_id>', views.file, name='v1_file'),
    path('files/<str:file_id>/content', views.file_content, name='file_content'),
    path('v1/files/<str:file_id>/content', views.file_content, name='v1_file_content'),
    # Batches endpoints
    path('batches', views.batches, name='batches'),
    path('v1/batches', views.batches, name='v1_batches'),
    path('batches/<str:batch_id>', views.batch, name='batch'),
    path('v1/batches/<str:batch_id>', views.batch, name='v1_batch'),
    path('batches/<str:batch_id>/cancel', views.batch_cancel, name='batch_cancel'),
    path('v1/batches/<str:batch_id>/cancel', views.batch_cancel, name='v1_batch_cancel'),
    
    # MCP server endpoints
    path('mcp-servers/<str:name>/mcp', mcp_views.mcp_server, name='mcp_server'),
]
