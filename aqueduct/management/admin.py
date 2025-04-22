from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import Org, Team, UserProfile, ServiceAccount, Token, Request, Model, Endpoint, TeamMembership


# Define an inline admin descriptor for UserProfile model
# which acts a bit like a singleton
class ProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profiles'
    fk_name = 'user'


# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = (ProfileInline,)


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
    list_display = ('name',)
    search_fields = ('name',)
    inlines = [TeamMembershipInline]  # Add the inline here


admin.site.register(Org)
# admin.site.register(Team)
admin.site.register(ServiceAccount)
admin.site.register(Token)
admin.site.register(Request)
admin.site.register(Model)
admin.site.register(Endpoint)
