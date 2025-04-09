from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import Org, Team, UserProfile, ServiceAccount, Token, Request, Model, Endpoint


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

admin.site.register(Org)
admin.site.register(Team)
admin.site.register(ServiceAccount)
admin.site.register(Token)
admin.site.register(Request)
admin.site.register(Model)
admin.site.register(Endpoint)
