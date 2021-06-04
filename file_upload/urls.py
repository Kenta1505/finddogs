from django.urls import path

from . import views

urlpatterns = [
    path('', views.file_upload, name='file_upload'),
    path("ajax_response/", views.test_ajax_response),
    path("ajax/", views.test_ajax_app),
]