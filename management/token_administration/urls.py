from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    # SSO view
    path(r'login/', views.SSOTemplateView.as_view(), name='sso'),
    path(r'admin/login/', views.SSOTemplateView.as_view(), name='admin_sso'),

    path('', RedirectView.as_view(url='/tokens/', permanent=True)),
    path("tokens/", views.TokensView.as_view(), name="tokens"),

    path(r'org/', views.OrgView.as_view(), name="org"),
    path('team/create/', views.TeamCreateView.as_view(), name='team_create'),
    path('team/<int:id>/delete/', views.TeamDeleteView.as_view(), name='team_delete'),
    path('team/<int:id>', views.TeamDetailView.as_view(), name='team'),

    path('team/<int:id>/service-accounts/add/',
         views.ServiceAccountCreateView.as_view(),
         name='service_account_create'),
    path('service-accounts/<int:service_account_id>/delete/',
         views.ServiceAccountDeleteView.as_view(),
         name='service_account_delete'),
    path('service-accounts/<int:service_account_id>/transfer/',
         views.service_account_transfer,
         name='service_account_transfer'),
]
