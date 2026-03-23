import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import ClassVar

from django import forms
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Group, User
from django.db.models import QuerySet
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


def format_unix_timestamp(timestamp: int | None) -> str:
    """Convert a Unix timestamp (seconds) to a readable datetime string."""
    if timestamp is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(timestamp, tz=UTC)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return str(timestamp)


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
        fields: ClassVar[list[str]] = []

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

    def clean_excluded_models(self) -> list[str]:
        # Store as list/JSON, not string
        return self.cleaned_data["excluded_models"]


class ExcludedMCPServersAdminForm(forms.ModelForm):
    class Meta:
        model = None  # To be specified by subclass
        fields: ClassVar[list[str]] = []

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

    def clean_excluded_mcp_servers(self) -> list[str]:
        # Store as list/JSON, not string
        return self.cleaned_data["excluded_mcp_servers"]


class UserProfileAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = UserProfile
        fields: ClassVar[list[str]] = [
            "user",
            "org",
            "teams",
            "requests_per_minute",
            "input_tokens_per_minute",
            "output_tokens_per_minute",
            "excluded_models",
            "merge_exclusion_lists",
            "excluded_mcp_servers",
            "merge_mcp_server_exclusion_lists",
        ]


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


@admin.action(description="Reload from upstream")
def reload_from_upstream(modeladmin, request, queryset):
    objects = list(queryset)
    errors = []

    async def reload_all():
        async def reload_obj(obj) -> None:
            try:
                await obj.areload_from_upstream(raise_on_error=True)
            except Exception as e:
                return str(e)
            else:
                return None

        results = await asyncio.gather(*[reload_obj(obj) for obj in objects])
        errors.extend(r for r in results if r)

    asyncio.run(reload_all())

    if errors:
        for error in errors:
            messages.error(request, f"Reload failed: {error}")
    else:
        messages.success(request, f"Successfully reloaded {len(objects)} object(s).")


# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines: ClassVar[tuple] = (ProfileInline,)
    list_display: ClassVar[tuple] = ("email", "is_staff", "get_groups", "request_limit", "input_limit", "output_limit")
    list_select_related: ClassVar[list] = ["profile"]
    actions: ClassVar[list] = [make_admin, make_org_admin, make_user, delete_tos_cache]

    def get_groups(self, obj) -> str:
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
class TeamMembershipInline(admin.TabularInline):
    model = TeamMembership
    extra = 1
    fields = ("user_profile", "is_admin", "date_added")
    readonly_fields = ("date_added",)  # Make date_joined read-only as it's auto-set


class TeamAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = Team
        fields: ClassVar[list[str]] = [
            "name",
            "description",
            "org",
            "requests_per_minute",
            "input_tokens_per_minute",
            "output_tokens_per_minute",
            "excluded_models",
            "merge_exclusion_lists",
            "excluded_mcp_servers",
            "merge_mcp_server_exclusion_lists",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Customize Team Admin
@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display: ClassVar[tuple] = (
        "name",
        "org_link",
        "requests_per_minute",
        "input_tokens_per_minute",
        "output_tokens_per_minute",
    )
    list_select_related: ClassVar[list] = ["org"]
    search_fields: ClassVar[tuple] = ("name",)
    inlines: ClassVar[list] = [TeamMembershipInline]
    list_filter: ClassVar[list] = ["org__name"]
    form = TeamAdminForm

    def org_link(self, obj) -> str:
        link = reverse("admin:management_org_change", args=[obj.org.id])
        return format_html('<a href="{}">{}</a>', link, obj.org.name)

    org_link.short_description = "Org"


class OrgAdminForm(ExcludedModelsAdminForm, ExcludedMCPServersAdminForm):
    class Meta(ExcludedModelsAdminForm.Meta):
        model = Org
        fields: ClassVar[list[str]] = [
            "name",
            "requests_per_minute",
            "input_tokens_per_minute",
            "output_tokens_per_minute",
            "excluded_models",
            "merge_exclusion_lists",
            "excluded_mcp_servers",
            "merge_mcp_server_exclusion_lists",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@admin.register(Org)
class OrgAdmin(admin.ModelAdmin):
    list_display = ("name", "requests_per_minute", "input_tokens_per_minute", "output_tokens_per_minute")
    form = OrgAdminForm


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    list_select_related: ClassVar[list] = ["token"]
    list_display: ClassVar[tuple] = (
        "id",
        "input_tokens",
        "output_tokens",
        "status_code",
        "response_time_ms",
        "model",
        "token__name",
        "user_id",
    )
    list_filter: ClassVar[list] = ["status_code", "model", "token"]


@admin.register(ServiceAccount)
class ServiceAccountAdmin(admin.ModelAdmin):
    list_display: ClassVar[tuple] = ("name", "description", "team_link")
    list_select_related: ClassVar[list] = ["team"]
    list_filter: ClassVar[list] = ["team__name"]

    def team_link(self, obj) -> str:
        link = reverse("admin:management_team_change", args=[obj.team.id])
        return format_html('<a href="{}">{}</a>', link, obj.team.name)

    team_link.short_description = "Team"


@admin.register(Token)
class TokenAdmin(admin.ModelAdmin):
    list_display: ClassVar[tuple] = ("name", "expires_at", "user_link", "sa_link")
    list_select_related: ClassVar[list] = ["user", "service_account", "service_account__team"]
    list_filter: ClassVar[list] = ["service_account__team__name"]

    def sa_link(self, obj) -> str:
        if obj.service_account is None:
            return "-"
        link = reverse("admin:management_serviceaccount_change", args=[obj.service_account.id])
        return format_html('<a href="{}">{}</a>', link, f"{obj.service_account.name} ({obj.service_account.team.name})")

    sa_link.short_description = "Service Account"

    def user_link(self, obj) -> str:
        link = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', link, obj.user.email)

    user_link.short_description = "User"


@admin.register(FileObject)
class FileObjectAdmin(admin.ModelAdmin):
    """Admin panel registration for FileObject model."""

    list_display: ClassVar[tuple] = (
        "id",
        "filename",
        "purpose",
        "bytes_formatted",
        "created_at_formatted",
        "expires_at_formatted",
    )
    list_filter: ClassVar[tuple] = ("purpose", "token__user__email")
    search_fields: ClassVar[tuple] = ("id", "filename")
    actions: ClassVar[list] = [reload_from_upstream]

    def bytes_formatted(self, obj) -> str:
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.bytes)

    bytes_formatted.short_description = "Size"
    bytes_formatted.admin_order_field = "bytes"

    def created_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.created_at)

    created_at_formatted.short_description = "Created At"
    created_at_formatted.admin_order_field = "created_at"

    def expires_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.expires_at)

    expires_at_formatted.short_description = "Expires At"
    expires_at_formatted.admin_order_field = "expires_at"


@admin.register(Batch)
class BatchAdmin(admin.ModelAdmin):
    """Admin panel registration for Batch model."""

    list_display: ClassVar[tuple] = (
        "id",
        "status",
        "created_at_formatted",
        "completion_window",
        "endpoint",
        "input_file",
    )
    list_filter: ClassVar[tuple] = ("status", "token__user__email")
    search_fields: ClassVar[tuple] = ("id",)
    actions: ClassVar[list] = [reload_from_upstream]

    def created_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.created_at)

    created_at_formatted.short_description = "Created At"
    created_at_formatted.admin_order_field = "created_at"


@admin.register(VectorStore)
class VectorStoreAdmin(admin.ModelAdmin):
    """Admin panel registration for VectorStore model."""

    list_display: ClassVar[tuple] = (
        "id",
        "name",
        "status",
        "usage_bytes_formatted",
        "file_count",
        "token_link",
        "created_at_formatted",
    )
    list_filter: ClassVar[tuple] = ("status", "token__user__email")
    search_fields: ClassVar[tuple] = ("id", "name", "remote_id")
    list_select_related: ClassVar[list] = ["token", "token__user"]
    readonly_fields: ClassVar[tuple] = ("id", "created_at_formatted")
    actions: ClassVar[list] = [reload_from_upstream]

    def get_queryset(self, request) -> QuerySet:
        from django.db.models import Count

        return super().get_queryset(request).annotate(file_count=Count("files"))

    def file_count(self, obj) -> int:
        return obj.file_count

    file_count.short_description = "Files"
    file_count.admin_order_field = "file_count"

    def usage_bytes_formatted(self, obj) -> str:
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.usage_bytes)

    usage_bytes_formatted.short_description = "Usage"
    usage_bytes_formatted.admin_order_field = "usage_bytes"

    def created_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.created_at)

    created_at_formatted.short_description = "Created At"
    created_at_formatted.admin_order_field = "created_at"

    def token_link(self, obj) -> str:
        link = reverse("admin:management_token_change", args=[obj.token.id])
        if obj.token.service_account:
            return format_html('<a href="{}">{} ({})</a>', link, obj.token.name, obj.token.service_account.name)
        return format_html('<a href="{}">{}</a>', link, obj.token.name)

    token_link.short_description = "Token"

    def delete_model(self, request, obj):
        """Delete from upstream API before deleting local record."""
        try:
            client = get_files_api_client()
        except ValueError:
            client = None

        if client:
            # First delete all files from upstream
            for vs_file in obj.files.all():
                with suppress(Exception):
                    asyncio.run(vs_file.adelete_upstream(client, raise_on_error=False))

            # Delete vector store from upstream
            with suppress(Exception):
                asyncio.run(obj.adelete_upstream(client, raise_on_error=False))

        # Now delete the local record (this will cascade to files)
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        """Delete from upstream API before deleting local records."""
        try:
            client = get_files_api_client()
        except ValueError:
            client = None

        if client:
            for obj in queryset:
                # First delete all files from upstream
                for vs_file in obj.files.all():
                    with suppress(Exception):
                        asyncio.run(vs_file.adelete_upstream(client, raise_on_error=False))

                # Delete vector store from upstream
                with suppress(Exception):
                    asyncio.run(obj.adelete_upstream(client, raise_on_error=False))

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
        "created_at_formatted",
    )
    list_filter: ClassVar[tuple] = ("status", "vector_store__name")
    search_fields: ClassVar[tuple] = ("id", "remote_id")
    list_select_related: ClassVar[list] = ["vector_store", "file_obj"]
    readonly_fields: ClassVar[tuple] = ("id", "created_at_formatted")
    actions: ClassVar[list] = [reload_from_upstream]

    def usage_bytes_formatted(self, obj) -> str:
        from django.template.defaultfilters import filesizeformat

        return filesizeformat(obj.usage_bytes)

    usage_bytes_formatted.short_description = "Size"
    usage_bytes_formatted.admin_order_field = "usage_bytes"

    def created_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.created_at)

    created_at_formatted.short_description = "Created At"
    created_at_formatted.admin_order_field = "created_at"

    def vector_store_link(self, obj) -> str:
        link = reverse("admin:management_vectorstore_change", args=[obj.vector_store.id])
        return format_html('<a href="{}">{}</a>', link, obj.vector_store.name)

    vector_store_link.short_description = "Vector Store"

    def file_obj_link(self, obj) -> str:
        link = reverse("admin:management_fileobject_change", args=[obj.file_obj.id])
        return format_html('<a href="{}">{}</a>', link, obj.file_obj.filename)

    file_obj_link.short_description = "File"

    def delete_model(self, request, obj):
        """Delete from upstream API before deleting local record."""
        with suppress(Exception):
            client = get_files_api_client()
            asyncio.run(obj.adelete_upstream(client, raise_on_error=False))

        # Now delete the local record
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        """Delete from upstream API before deleting local records."""
        client = None
        with suppress(ValueError):
            client = get_files_api_client()

        if client:
            for obj in queryset:
                with suppress(Exception):
                    asyncio.run(obj.adelete_upstream(client, raise_on_error=False))

        # Now delete local records
        super().delete_queryset(request, queryset)


@admin.register(VectorStoreFileBatch)
class VectorStoreFileBatchAdmin(admin.ModelAdmin):
    """Admin panel registration for VectorStoreFileBatch model."""

    list_display: ClassVar[tuple] = (
        "id",
        "vector_store_link",
        "status",
        "file_counts_formatted",
        "created_at_formatted",
    )
    list_filter: ClassVar[tuple] = ("status", "vector_store__name")
    search_fields: ClassVar[tuple] = ("id", "remote_id")
    list_select_related: ClassVar[list] = ["vector_store"]
    readonly_fields: ClassVar[tuple] = ("id", "created_at_formatted")
    actions: ClassVar[list] = [reload_from_upstream]

    def file_counts_formatted(self, obj) -> str:
        counts = obj.file_counts or {}
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        return f"{completed}/{total} (failed: {failed})"

    file_counts_formatted.short_description = "Progress"

    def created_at_formatted(self, obj) -> str:
        return format_unix_timestamp(obj.created_at)

    created_at_formatted.short_description = "Created At"
    created_at_formatted.admin_order_field = "created_at"

    def vector_store_link(self, obj) -> str:
        link = reverse("admin:management_vectorstore_change", args=[obj.vector_store.id])
        return format_html('<a href="{}">{}</a>', link, obj.vector_store.name)

    vector_store_link.short_description = "Vector Store"
