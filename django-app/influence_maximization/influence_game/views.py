from django.shortcuts import render
from django.http import JsonResponse
import influence_game.algorithms.graph_types as gt


def home(request):
    return render(request, 'influence_game/index.html')


def random_proximity_view(request):
    if request.method == 'GET':
        nodes, edges = gt.random_proximity()
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
