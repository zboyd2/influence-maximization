from django.shortcuts import render


def home(request):
    return render(request, 'influence_game/index.html')
