from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    # SSO view
    path(r'login/', views.SSOTemplateView.as_view(), name='sso'),
    path(r'admin/login/', views.SSOTemplateView.as_view(), name='admin_sso'),

    path('', RedirectView.as_view(url='/tokens/', permanent=True)),
    path("tokens/", views.tokens, name="tokens"),

    path(r'org/', views.org, name="org"),
    path('team/create/', views.team_create, name='team_create'),
    path('team/<int:id>', views.team, name='team'),

]