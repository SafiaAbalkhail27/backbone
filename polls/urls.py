from django.urls import path
from . import views 

urlpatterns = [
    path("", views.analayse_trends),
    path('upload/', views.upload_file, name='upload_file'),
]