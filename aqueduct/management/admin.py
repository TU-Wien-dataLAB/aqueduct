from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User, Group
from django.urls import reverse
from django.utils.html import format_html

from .models import Org, Team, UserProfile, ServiceAccount, Token, Request, TeamMembership


# Define an inline admin descriptor for UserProfile model
# which acts a bit like a singleton
class ProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profiles'
    fk_name = 'user'


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


# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = (ProfileInline,)
    list_display = ('email', 'is_staff', 'get_groups', 'request_limit', 'input_limit', 'output_limit')
    list_select_related = ['profile']
    actions = [make_admin, make_org_admin, make_user]

    def get_groups(self, obj):
        return ", ".join([g.name for g in obj.groups.all()])

    get_groups.short_description = 'Groups'

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
    fields = ('user_profile', 'is_admin', 'date_added')  # Fields to display in the inline
    readonly_fields = ('date_added',)  # Make date_joined read-only as it's auto-set


# Customize Team Admin
@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ('name', 'org_link', 'requests_per_minute', 'input_tokens_per_minute', 'output_tokens_per_minute')
    list_select_related = ["org"]
    search_fields = ('name',)
    inlines = [TeamMembershipInline]  # Add the inline here
    list_filter = ['org__name']

    def org_link(self, obj):
        link = reverse("admin:management_org_change", args=[obj.org.id])
        return format_html('<a href="{}">{}</a>', link, obj.org.name)

    org_link.short_description = "Org"


@admin.register(Org)
class OrgAdmin(admin.ModelAdmin):
    list_display = ('name', 'requests_per_minute', 'input_tokens_per_minute', 'output_tokens_per_minute')


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'input_tokens', 'output_tokens', 'status_code', 'response_time_ms', 'model')
    list_filter = ['status_code', 'model']

@admin.register(ServiceAccount)
class ServiceAccountAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'team_link')
    list_select_related = ['team']
    list_filter = ['team__name']

    def team_link(self, obj):
        link = reverse("admin:management_team_change", args=[obj.team.id])
        return format_html('<a href="{}">{}</a>', link, obj.team.name)

    team_link.short_description = "Team"


@admin.register(Token)
class TokenAdmin(admin.ModelAdmin):
    list_display = ('name', 'expires_at', 'user_link', 'sa_link')
    list_select_related = ['user', 'service_account', 'service_account__team']
    list_filter = ['service_account__team__name']

    def sa_link(self, obj):
        if obj.service_account is None:
            return "-"
        link = reverse("admin:management_serviceaccount_change", args=[obj.service_account.id])
        return format_html('<a href="{}">{}</a>', link, f"{obj.service_account.name} ({obj.service_account.team.name})")

    sa_link.short_description = "Service Account"

    def user_link(self, obj):
        link = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', link, obj.user.email)

    user_link.short_description = "User"
