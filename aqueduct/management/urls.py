from django.conf import settings
from django.urls import path

from . import views

urlpatterns = [
    path("tokens/", views.UserTokensView.as_view(), name="tokens"),
    path("tokens/create/", views.TokenCreateView.as_view(), name="token_create"),
    path("tokens/<int:id>/", views.TokenDetailView.as_view(), name="token_detail"),
    path("tokens/<int:id>/delete/", views.TokenDeleteView.as_view(), name="token_delete"),
    path("tokens/<int:id>/edit/", views.TokenEditView.as_view(), name="token_edit"),
    path(
        "tokens/<int:id>/regenerate/", views.TokenRegenerateView.as_view(), name="token_regenerate"
    ),
    path(r"org/", views.OrgView.as_view(), name="org"),
    path("org/switch/", views.OrgSwitchView.as_view(), name="org_switch"),
    path("organization/", views.OrganizationView.as_view(), name="organization"),
    path(
        "organization/<int:id>/limits/",
        views.OrgLimitsEditView.as_view(),
        name="organization_limits",
    ),
    path(
        "organization/<int:id>/availability/",
        views.OrgAvailabilityEditView.as_view(),
        name="organization_availability",
    ),
    path("team/create/", views.TeamCreateView.as_view(), name="team_create"),
    path("team/<int:id>/delete/", views.TeamDeleteView.as_view(), name="team_delete"),
    path("team/<int:id>/edit/", views.TeamUpdateView.as_view(), name="team_edit"),
    path(
        "team/<int:id>/availability/",
        views.TeamAvailabilityEditView.as_view(),
        name="team_availability",
    ),
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
    path("team/<int:id>/members/", views.TeamMembersView.as_view(), name="team_members"),
    # New URL for Model/Endpoint list
    path("models/", views.ModelListView.as_view(), name="models"),
    path("usage/", views.UsageDashboardView.as_view(), name="usage"),
    path("usage/keys/", views.KeyUsageView.as_view(), name="usage_keys"),
    path("requests/", views.RequestLogView.as_view(), name="requests"),
    path("files/", views.UserFilesView.as_view(), name="files"),
    path("files/<str:id>/delete/", views.FileDeleteView.as_view(), name="file_delete"),
    path("batches/", views.UserBatchesView.as_view(), name="batches"),
    path("batches/<str:id>/cancel/", views.BatchCancelView.as_view(), name="batch_cancel"),
    path("mcp-servers/", views.MCPServersView.as_view(), name="mcp_servers"),
    path("vector-stores/", views.UserVectorStoresView.as_view(), name="vector_stores"),
    path(
        "vector-stores/<str:id>/", views.VectorStoreDetailView.as_view(), name="vector_store_detail"
    ),
    path(
        "vector-stores/<str:id>/refresh/",
        views.VectorStoreDetailRefreshView.as_view(),
        name="vector_store_detail_refresh",
    ),
    path(
        "vector-stores/<str:id>/card-refresh/",
        views.VectorStoreCardRefreshView.as_view(),
        name="vector_store_card_refresh",
    ),
]

# Test-only endpoints (only active during load testing)
if getattr(settings, "LOAD_TESTING", False):
    from .views import test_auth

    urlpatterns += [
        path(
            "test-auth/generate-token/", test_auth.generate_test_token, name="generate-test-token"
        ),
        path("test-auth/cleanup-token/", test_auth.cleanup_test_token, name="cleanup-test-token"),
    ]
