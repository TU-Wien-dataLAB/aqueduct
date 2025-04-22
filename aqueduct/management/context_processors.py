from django.conf import settings as django_settings


def settings(_):
    """
    Adds EXTRA_NAV_LINKS from settings to the template context.
    """
    return {
        'settings': django_settings,
    }
