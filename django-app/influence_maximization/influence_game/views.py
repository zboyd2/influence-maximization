from django.shortcuts import render
from django.http import JsonResponse
import influence_game.algorithms.graph_types as gt


def home(request):
    return render(request, 'influence_game/index.html')


def distribution_view(request):
    if request.method == 'GET':
        nodes, edges = gt.random_proximity_probability()
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def tree_view(request):
    if request.method == 'GET':
        nodes, edges = gt.tree()
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    

def ladder_view(request):
    if request.method == 'GET':
        num_nodes, nodes, edges = gt.ladder()
        response_data = {
            'numNodes': num_nodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def square_view(request):
    if request.method == 'GET':
        num_nodes, nodes, edges = gt.square_lattice()
        response_data = {
            'numNodes': num_nodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def hexagon_view(request):
    if request.method == 'GET':
        num_nodes, nodes, edges = gt.hexagon_lattice()
        response_data = {
            'numNodes': num_nodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)


def triangle_view(request):
    if request.method == 'GET':
        num_nodes, nodes, edges = gt.triangle_lattice()
        response_data = {
            'numNodes': num_nodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def cycle_view(request):
    if request.method == 'GET':
        nodes, edges = gt.cycle()
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)

    
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
