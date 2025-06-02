from django.apps import AppConfig


class GatewayConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gateway'

    def ready(self):
        from django.conf import settings
        from . import router
        router.load_router(settings.LITELLM_ROUTER_CONFIG_FILE_PATH)
