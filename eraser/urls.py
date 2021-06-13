from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from . import views

app_name='eraser'

urlpatterns = [
    path('', views.eraser, name='eraser'),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)