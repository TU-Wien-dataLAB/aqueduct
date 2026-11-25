from django.conf import settings as django_settings
from django.http import HttpRequest
from django.utils.functional import SimpleLazyObject

from management.models import UserProfile
from management.themes import custom_themes, default_themes


def settings(request: HttpRequest) -> dict:
    """Settings (EXTRA_NAV_LINKS, TOS_ENABLED, SILKY_ENABLED), the viewer, the custom themes
    and the default theme of each scheme, for the shell."""

    def viewer() -> UserProfile | None:
        if not request.user.is_authenticated:
            return None
        profile: UserProfile | None = getattr(request.user, "profile", None)
        return profile

    lazy_viewer = SimpleLazyObject(viewer)
    return {
        "settings": django_settings,
        "viewer": lazy_viewer,
        "viewer_is_org_admin": SimpleLazyObject(
            lambda: bool(lazy_viewer) and lazy_viewer.is_org_admin(lazy_viewer.org)
        ),
        "custom_themes": custom_themes(),
        "default_themes": default_themes(),
    }
