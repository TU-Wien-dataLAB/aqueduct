from django.apps import AppConfig, apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import caches
from django.db.models import Q
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver


class AqueductManagementConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'management'

    def ready(self):
        if 'tos' in settings.INSTALLED_APPS:
            cache = caches[getattr(settings, 'TOS_CACHE_NAME', 'default')]
            tos_app = apps.get_app_config('tos')
            TermsOfService = tos_app.get_model('TermsOfService')

            @receiver(post_save, sender=get_user_model(), dispatch_uid='set_staff_in_cache_for_tos')
            def set_staff_in_cache_for_tos(instance, **kwargs):
                if kwargs.get('raw', False):
                    return

                # Get the cache prefix
                key_version = cache.get('django:tos:key_version')

                # If the user is staff allow them to skip the TOS agreement check
                if instance.is_staff or instance.is_superuser:
                    cache.set('django:tos:skip_tos_check:{}'.format(instance.id), True, version=key_version)

                # But if they aren't make sure we invalidate them from the cache
                elif cache.get('django:tos:skip_tos_check:{}'.format(instance.id), False):
                    cache.delete('django:tos:skip_tos_check:{}'.format(instance.id), version=key_version)

            @receiver(post_save, sender=TermsOfService, dispatch_uid='add_staff_users_to_tos_cache')
            def add_staff_users_to_tos_cache(*args, **kwargs):
                if kwargs.get('raw', False):
                    return

                # Get the cache prefix
                key_version = cache.get('django:tos:key_version')

                # Efficiently cache all of the users who are allowed to skip the TOS
                # agreement check
                cache.set_many({
                    'django:tos:skip_tos_check:{}'.format(staff_user.id): True
                    for staff_user in get_user_model().objects.filter(
                        Q(is_staff=True) | Q(is_superuser=True))
                }, version=key_version)

            # Immediately add staff users to the cache
            add_staff_users_to_tos_cache()
