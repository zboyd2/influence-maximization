"""
URL configuration for influence_maximization project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path
from influence_game.views import home, distribution_view, tree_view, ladder_view, square_view, hexagon_view, triangle_view, cycle_view, random_proximity_view

urlpatterns = [
    path('', home, name='home'),
    path('api/distribution/', distribution_view, name='distribution'),
    path('api/tree/', tree_view, name='tree'),
    path('api/ladder/', ladder_view, name='ladder'),
    path('api/square/', square_view, name='square'),
    path('api/hexagon/', hexagon_view, name='hexagon'),
    path('api/triangle/', triangle_view, name='triangle'),
    path('api/cycle/', cycle_view, name='cycle'),
    path('api/random_proximity/', random_proximity_view, name='random_proximity'),
]
