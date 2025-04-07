from mozilla_django_oidc.auth import OIDCAuthenticationBackend
from django.conf import settings
from .models import Org, UserProfile


def default_org_name_from_groups(groups: list[str]) -> str | None:
    """
    Default implementation to extract organization name, which returns the first group in the list.
    Override this or set ORG_NAME_FROM_GROUPS_FUNCTION in settings.
    """
    if not groups:
        return None
    return groups[0]


def get_org_name_from_groups(groups):
    """
    Extracts the organization name from the user's groups.
    """
    if hasattr(settings, 'ORG_NAME_FROM_GROUPS_FUNCTION'):
        return settings.ORG_NAME_FROM_GROUPS_FUNCTION(groups)
    else:
        return default_org_name_from_groups(groups)


class OIDCBackend(OIDCAuthenticationBackend):
    def create_user(self, claims):
        groups = claims.get('groups', [])

        org_name = get_org_name_from_groups(groups)
        if not org_name:
            return None  # Authentication fails if no org can be determined

        org, created = Org.objects.get_or_create(name=org_name)

        user = super(OIDCBackend, self).create_user(claims)
        UserProfile.objects.create(user=user, org=org)

        # Check if user is admin
        if hasattr(settings, 'ADMIN_GROUP'):
            is_admin = settings.ADMIN_GROUP in groups
        else:
            is_admin = False
        user.is_staff = is_admin
        user.is_superuser = is_admin

        user.save()

        return user
