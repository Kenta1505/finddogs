"""MyApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
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
from django.urls import path
from django.urls import include
import file_upload.views as file_upload
from django.conf import settings
from django.conf.urls.static import static
# import eraser.views as eraser

urlpatterns = [
    # path('success/url/',file_upload.success),
    path('',include('file_upload.urls')),
    # path('', eraser.eraser,name="eraser"),
    path('admin/', admin.site.urls),
    path('finddogs/', include('FindDogs.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# if settings.DEBUG:
#     urlpatterns += static(settings.IMAGE_URL, document_root=settings.IMAGE_ROOT)
