from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from . import views

app_name='file_upload'

urlpatterns = [
    # path('', views.test_ajax_app),
    # path('', views.test_ajax_response),
    path('', views.file_upload, name='file_upload'),
    path('response/', views.file_upload_response, name="upload_response"),
    # path("", views.ImageUpload, name="imgupload"),
    # path("ajax_response/", views.test_ajax_response),
    # path("ajax/", views.test_ajax_app),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)