from django.urls import path

from . import views

app_name='file_upload'

urlpatterns = [
    path('', views.test_ajax_app),
    path('ajax/', views.test_ajax_response),
    # path('', views.file_upload, name='file_upload'),
    # path("", views.ImageUpload, name="imgupload"),
    # path("ajax_response/", views.test_ajax_response),
    # path("ajax/", views.test_ajax_app),
]