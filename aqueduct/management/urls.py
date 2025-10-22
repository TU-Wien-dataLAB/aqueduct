from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.urls import path

from . import views


@login_required
def mcp_servers(request):
    """MCP Servers overview page"""
    return render(request, "management/mcp_servers.html")


urlpatterns = [
    path("tokens/", views.UserTokensView.as_view(), name="tokens"),
    path("tokens/create/", views.TokenCreateView.as_view(), name="token_create"),
    path("tokens/<int:id>/delete/", views.TokenDeleteView.as_view(), name="token_delete"),
    path("tokens/<int:id>/edit/", views.TokenEditView.as_view(), name="token_edit"),
    path(
        "tokens/<int:id>/regenerate/", views.TokenRegenerateView.as_view(), name="token_regenerate"
    ),
    path(r"org/", views.OrgView.as_view(), name="org"),
    path("org/switch/", views.OrgSwitchView.as_view(), name="org_switch"),
    path("team/create/", views.TeamCreateView.as_view(), name="team_create"),
    path("team/<int:id>/delete/", views.TeamDeleteView.as_view(), name="team_delete"),
    path("team/<int:id>/edit/", views.TeamUpdateView.as_view(), name="team_edit"),
    path("team/<int:id>", views.TeamDetailView.as_view(), name="team"),
    path(
        "team/<int:id>/service-accounts/add/",
        views.ServiceAccountCreateView.as_view(),
        name="service_account_create",
    ),
    path(
        "service-accounts/<int:service_account_id>/delete/",
        views.ServiceAccountDeleteView.as_view(),
        name="service_account_delete",
    ),
    path(
        "service-accounts/<int:service_account_id>/edit/",
        views.ServiceAccountUpdateView.as_view(),
        name="service_account_edit",
    ),
    path(
        "service-accounts/<int:service_account_id>/transfer/",
        views.ServiceAccountTransferOwnershipView.as_view(),
        name="service_account_transfer",
    ),
    path(
        "team/<int:id>/admins/", views.TeamAdminManagementView.as_view(), name="team_admins_manage"
    ),
    # New URL for Model/Endpoint list
    path("models/", views.ModelListView.as_view(), name="models"),
    path("usage/", views.UsageDashboardView.as_view(), name="usage"),
    path("files/", views.UserFilesView.as_view(), name="files"),
    path("batches/", views.UserBatchesView.as_view(), name="batches"),
    path("mcp-servers/", mcp_servers, name="mcp_servers"),
]
