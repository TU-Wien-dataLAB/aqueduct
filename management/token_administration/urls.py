from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    # SSO view
    path(r'login/', views.SSOTemplateView.as_view(), name='sso'),
    path(r'admin/login/?', views.SSOTemplateView.as_view(), name='admin_sso'),

    path('', RedirectView.as_view(url='/tokens/', permanent=True)),
    path("tokens/", views.index, name="tokens"),
]