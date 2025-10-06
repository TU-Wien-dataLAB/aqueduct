import logging

from django.apps import AppConfig, apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import caches
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.signals import post_save
from django.dispatch import receiver

log = logging.getLogger('aqueduct')


class AqueductManagementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "management"

    def ready(self):
        if settings.TOS_ENABLED:
            log.info('Setting up TOS dispatch functions...')
            cache = caches[getattr(settings, 'TOS_CACHE_NAME', 'default')]
            tos_app = apps.get_app_config('tos')
            TermsOfService = tos_app.get_model('TermsOfService')

            @receiver(post_save, sender=get_user_model(), dispatch_uid='set_staff_in_cache_for_tos', weak=False)
            def set_staff_in_cache_for_tos(instance, **kwargs):
                """ Skips TOS check for admin users. Is called after login (because of user.save()) so that cache is updated at each login."""
                if kwargs.get('raw', False):
                    return

                # Get the cache prefix
                key_version = cache.get('django:tos:key_version')

                # If the user is admin allow them to skip the TOS agreement check
                try:
                    group = instance.profile.group
                except ObjectDoesNotExist:
                    return

                if group == "admin":
                    cache.set('django:tos:skip_tos_check:{}'.format(instance.id), True, version=key_version)
                    log.info(f"Added admin user '{instance.email}' to TOS cache")

                # But if they aren't make sure we invalidate them from the cache
                elif cache.get('django:tos:skip_tos_check:{}'.format(instance.id), False):
                    cache.delete('django:tos:skip_tos_check:{}'.format(instance.id), version=key_version)

            @receiver(post_save, sender=TermsOfService, dispatch_uid='add_staff_users_to_tos_cache', weak=False)
            def add_staff_users_to_tos_cache(*args, **kwargs):
                if kwargs.get('raw', False):
                    return

                # Get the cache prefix
                key_version = cache.get('django:tos:key_version')

                # Efficiently cache all of the users who are allowed to skip the TOS
                # agreement check
                cache.set_many({
                    'django:tos:skip_tos_check:{}'.format(user.id): True
                    for user in get_user_model().objects.filter(
                        groups__name='admin')
                }, version=key_version)
                log.info("Added admin users to TOS cache")
