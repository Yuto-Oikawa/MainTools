from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classified', views.search_button, name='search_button'),
    path('classified/redraw', views.redraw, name='redraw'),
]
