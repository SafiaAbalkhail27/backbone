from django.urls import path
from . import views 

urlpatterns = [
    path("", views.analayse_trends),
]