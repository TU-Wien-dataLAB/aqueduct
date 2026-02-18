import asyncio

from django import forms
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Group, User
from django.urls import reverse
from django.utils.html import format_html

from gateway.config import get_files_api_client, get_router_config

from .models import (
    Batch,
    FileObject,
    Org,
    Request,
    ServiceAccount,
    Team,
    TeamMembership,
    Token,
    UserProfile,
    VectorStore,
    VectorStoreFile,
    VectorStoreFileBatch,
)


def get_model_choices() -> list[str]:
    config = get_router_config()
    return [m["model_name"] for m in config.get("model_list", [])]


def get_mcp_server_choices() -> list[str]:
    from gateway.config import get_mcp_config

    try:
        config = get_mcp_config()
        return list(config.keys())
    except RuntimeError:
        return []


class ExcludedModelsAdminForm(forms.ModelForm):
    class Meta:
        model = None  # To be specified by subclass
        fields = "__all__"

    excluded_models = forms.MultipleChoiceField(
        choices=[],
        required=False,
        widget=forms.CheckboxSelectMultiple(),
        label="Excluded Models",
        help_text="Choose models to be excluded.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically provide choices using get_model_choices
        from .admin import get_model_choices

        self.fields["excluded_models"].choices = [(m, m) for m in get_model_choices()]
        if self.instance and getattr(self.instance, "excluded_models", None):
            self.initial["excluded_models"] = self.instance.excluded_models

    def clean_excluded_models(self):
        # Store as list/JSON, not string
        return self.cleaned_data["excluded_models"]


class ExcludedMCPServersAdminForm(forms.ModelForm):
    class Meta:
        model = None  # To be specified by subclass
        fields = "__all__"

    excluded_mcp_servers = forms.MultipleChoiceField(
        choices=[],
        required=False,
        widget=forms.CheckboxSelectMultiple(),
        label="Excluded MCP Servers",
        help_text="Choose MCP servers to be excluded.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["excluded_mcp_servers"].choices = [(s, s) for s in get_mcp_server_choices()]
        if self.instance and getattr(self.instance, "excluded_mcp_servers", None):
            self.initial["excluded_mcp_servers"] = self.instance.excluded_mcp_servers

    def clean_excluded_mcp_servers(self):
        # Store as list/JSON, not string
        return self.cleaned_data["excluded_mcp_servers"]


class UserProfileAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = UserProfile
        fields = "__all__"


# Define an inline admin descriptor for UserProfile model
# which acts a bit like a singleton
class ProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = "Profiles"
    fk_name = "user"
    form = UserProfileAdminForm


def set_user_group(queryset, group_name):
    group, _ = Group.objects.get_or_create(name=group_name)
    is_admin = group_name == "admin"
    for user in queryset:
        user.groups.clear()
        user.groups.add(group)
        if is_admin:
            user.is_staff = True
            user.is_superuser = True
        else:
            user.is_staff = False
            user.is_superuser = False
        user.save()


@admin.action(description="Make Admin")
def make_admin(modeladmin, request, queryset):
    set_user_group(queryset, "admin")


@admin.action(description="Make Org-Admin")
def make_org_admin(modeladmin, request, queryset):
    set_user_group(queryset, "org-admin")


@admin.action(description="Make User")
def make_user(modeladmin, request, queryset):
    set_user_group(queryset, "user")


@admin.action(description="Delete Terms of Service cache")
def delete_tos_cache(modeladmin, request, queryset):
    from tos.middleware import cache

    key_version = cache.get("django:tos:key_version")

    for user in queryset:
        cache.delete(f"django:tos:skip_tos_check:{user.id}", version=key_version)
        cache.delete(f"django:tos:agreed:{user.id}", version=key_version)


# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = (ProfileInline,)
    list_display = (
        "email",
        "is_staff",
        "get_groups",
        "request_limit",
        "input_limit",
        "output_limit",
    )
    list_select_related = ["profile"]
    actions = [make_admin, make_org_admin, make_user, delete_tos_cache]

    def get_groups(self, obj):
        return ", ".join([g.name for g in obj.groups.all()])

    get_groups.short_description = "Groups"

    def request_limit(self, obj) -> int:
        return obj.profile.requests_per_minute

    request_limit.short_description = "Requests per minute"

    def input_limit(self, obj) -> int:
        return obj.profile.input_tokens_per_minute

    input_limit.short_description = "Input tokens per minute"

    def output_limit(self, obj) -> int:
        return obj.profile.output_tokens_per_minute

    output_limit.short_description = "Output tokens per minute"


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


# Inline class for Membership
class TeamMembershipInline(admin.TabularInline):  # or admin.StackedInline for a different layout
    model = TeamMembership
    extra = 1  # How many empty rows to show for adding new members
    # autocomplete_fields = ('user_profile',)  # Easier selection of users
    fields = ("user_profile", "is_admin", "date_added")  # Fields to display in the inline
    readonly_fields = ("date_added",)  # Make date_joined read-only as it's auto-set


class TeamAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = Team
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Customize Team Admin
@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "org_link",
        "requests_per_minute",
        "input_tokens_per_minute",
        "output_tokens_per_minute",
    )
    list_select_related = ["org"]
    search_fields = ("name",)
    inlines = [TeamMembershipInline]  # Add the inline here
    list_filter = ["org__name"]
    form = TeamAdminForm

    def org_link(self, obj):
        link = reverse("admin:management_org_change", args=[obj.org.id])
        return format_html('<a href="{}">{}</a>', link, obj.org.name)

    org_link.short_description = "Org"


class OrgAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = Org
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@admin.register(Org)
class OrgAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "requests_per_minute",
        "input_tokens_per_minute",
        "output_tokens_per_minute",
    )
    form = OrgAdminForm


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    list_select_related = ["token"]
    list_display = (
        "id",
        "input_tokens",
        "output_tokens",
        "status_code",
        "response_time_ms",
        "model",
        "token__name",
        "user_id",
    )
    list_filter = ["status_code", "model", "token"]


@admin.register(ServiceAccount)
class ServiceAccountAdmin(admin.ModelAdmin):
    list_display = ("name", "description", "team_link")
    list_select_related = ["team"]
    list_filter = ["team__name"]

    def team_link(self, obj):
        link = reverse("admin:management_team_change", args=[obj.team.id])
        return format_html('<a href="{}">{}</a>', link, obj.team.name)

    team_link.short_description = "Team"


@admin.register(Token)
class TokenAdmin(admin.ModelAdmin):
    list_display = ("name", "expires_at", "user_link", "sa_link")
    list_select_related = ["user", "service_account", "service_account__team"]
    list_filter = ["service_account__team__name"]

    def sa_link(self, obj):
        if obj.service_account is None:
            return "-"
        link = reverse("admin:management_serviceaccount_change", args=[obj.service_account.id])
        return format_html(
            '<a href="{}">{}</a>',
            link,
            f"{obj.service_account.name} ({obj.service_account.team.name})",
        )

    sa_link.short_description = "Service Account"

    def user_link(self, obj):
        link = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', link, obj.user.email)

    user_link.short_description = "User"


@admin.register(FileObject)
class FileObjectAdmin(admin.ModelAdmin):
    """Admin panel registration for FileObject model."""

    list_display = ("id", "filename", "purpose", "bytes_formatted", "created_at", "expires_at")
    list_filter = ("purpose", "token__user__email")
    search_fields = ("id", "filename")

    def bytes_formatted(self, obj):
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.bytes)

    bytes_formatted.short_description = "Size"
    bytes_formatted.admin_order_field = "bytes"


@admin.register(Batch)
class BatchAdmin(admin.ModelAdmin):
    """Admin panel registration for Batch model."""

    list_display = ("id", "status", "created_at", "completion_window", "endpoint", "input_file")
    list_filter = ("status", "token__user__email")
    search_fields = ("id",)


@admin.register(VectorStore)
class VectorStoreAdmin(admin.ModelAdmin):
    """Admin panel registration for VectorStore model."""

    list_display = (
        "id",
        "name",
        "status",
        "usage_bytes_formatted",
        "file_count",
        "token_link",
        "created_at",
    )
    list_filter = ("status", "token__user__email")
    search_fields = ("id", "name", "remote_id")
    list_select_related = ["token", "token__user"]
    readonly_fields = ("id", "created_at")

    def get_queryset(self, request):
        from django.db.models import Count

        return super().get_queryset(request).annotate(file_count=Count("files"))

    def file_count(self, obj):
        return obj.file_count

    file_count.short_description = "Files"
    file_count.admin_order_field = "file_count"

    def usage_bytes_formatted(self, obj):
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.usage_bytes)

    usage_bytes_formatted.short_description = "Usage"
    usage_bytes_formatted.admin_order_field = "usage_bytes"

    def token_link(self, obj):
        link = reverse("admin:management_token_change", args=[obj.token.id])
        if obj.token.service_account:
            return format_html(
                '<a href="{}">{} ({})</a>', link, obj.token.name, obj.token.service_account.name
            )
        return format_html('<a href="{}">{}</a>', link, obj.token.name)

    token_link.short_description = "Token"

    def delete_model(self, request, obj):
        """Delete from upstream API before deleting local record."""
        # First delete all files from upstream
        for vs_file in obj.files.all():
            if vs_file.remote_id:
                try:
                    client = get_files_api_client()
                    asyncio.run(vs_file.adelete_upstream(client, raise_on_error=False))
                except Exception:
                    pass  # Continue even if upstream deletion fails

        # Delete vector store from upstream
        if obj.remote_id:
            try:
                client = get_files_api_client()
                asyncio.run(obj.adelete_upstream(client, raise_on_error=False))
            except Exception:
                pass  # Continue even if upstream deletion fails

        # Now delete the local record (this will cascade to files)
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        """Delete from upstream API before deleting local records."""
        for obj in queryset:
            # First delete all files from upstream
            for vs_file in obj.files.all():
                if vs_file.remote_id:
                    try:
                        client = get_files_api_client()
                        asyncio.run(vs_file.adelete_upstream(client, raise_on_error=False))
                    except Exception:
                        pass

            # Delete vector store from upstream
            if obj.remote_id:
                try:
                    client = get_files_api_client()
                    asyncio.run(obj.adelete_upstream(client, raise_on_error=False))
                except Exception:
                    pass

        # Now delete local records
        super().delete_queryset(request, queryset)


@admin.register(VectorStoreFile)
class VectorStoreFileAdmin(admin.ModelAdmin):
    """Admin panel registration for VectorStoreFile model."""

    list_display = (
        "id",
        "vector_store_link",
        "file_obj_link",
        "status",
        "usage_bytes_formatted",
        "created_at",
    )
    list_filter = ("status", "vector_store__name")
    search_fields = ("id", "remote_id")
    list_select_related = ["vector_store", "file_obj"]
    readonly_fields = ("id", "created_at")

    def usage_bytes_formatted(self, obj):
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.usage_bytes)

    usage_bytes_formatted.short_description = "Size"
    usage_bytes_formatted.admin_order_field = "usage_bytes"

    def vector_store_link(self, obj):
        link = reverse("admin:management_vectorstore_change", args=[obj.vector_store.id])
        return format_html('<a href="{}">{}</a>', link, obj.vector_store.name)

    vector_store_link.short_description = "Vector Store"

    def file_obj_link(self, obj):
        link = reverse("admin:management_fileobject_change", args=[obj.file_obj.id])
        return format_html('<a href="{}">{}</a>', link, obj.file_obj.filename)

    file_obj_link.short_description = "File"

    def delete_model(self, request, obj):
        """Delete from upstream API before deleting local record."""
        if obj.remote_id:
            try:
                client = get_files_api_client()
                asyncio.run(obj.adelete_upstream(client, raise_on_error=False))
            except Exception:
                pass  # Continue even if upstream deletion fails

        # Now delete the local record
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        """Delete from upstream API before deleting local records."""
        for obj in queryset:
            if obj.remote_id:
                try:
                    client = get_files_api_client()
                    asyncio.run(obj.adelete_upstream(client, raise_on_error=False))
                except Exception:
                    pass  # Continue even if upstream deletion fails

        # Now delete local records
        super().delete_queryset(request, queryset)


@admin.register(VectorStoreFileBatch)
class VectorStoreFileBatchAdmin(admin.ModelAdmin):
    """Admin panel registration for VectorStoreFileBatch model."""

    list_display = ("id", "vector_store_link", "status", "file_counts_formatted", "created_at")
    list_filter = ("status", "vector_store__name")
    search_fields = ("id", "remote_id")
    list_select_related = ["vector_store"]
    readonly_fields = ("id", "created_at")

    def file_counts_formatted(self, obj):
        counts = obj.file_counts or {}
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        return f"{completed}/{total} (failed: {failed})"

    file_counts_formatted.short_description = "Progress"

    def vector_store_link(self, obj):
        link = reverse("admin:management_vectorstore_change", args=[obj.vector_store.id])
        return format_html('<a href="{}">{}</a>', link, obj.vector_store.name)

    vector_store_link.short_description = "Vector Store"
