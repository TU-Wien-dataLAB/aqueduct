from django.urls import path

from . import views

urlpatterns = [
    # SSO view
    path('', views.SSOTemplateView.as_view(), name='sso'),
    path('admin/login', views.SSOTemplateView.as_view(), name='admin_sso'),

    path("tokens/", views.index, name="tokens"),
]