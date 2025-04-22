"""
URL configuration for management project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

import management.views as views

urlpatterns = [
    # SSO view from management app
    path(r'login/', views.SSOTemplateView.as_view(), name='sso'),
    path(r'aqueduct/admin/login/', views.SSOTemplateView.as_view(), name='admin_sso'),

    path('oidc/', include('mozilla_django_oidc.urls')),
    path("aqueduct/management/", include("management.urls")),
    path("aqueduct/admin/", admin.site.urls),
    # contains catch all path so has to come last
    path('', include('gateway.urls')),
]
