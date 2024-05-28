from django.shortcuts import render
from django.http import JsonResponse
import influence_game.algorithms.graph_types as gt


def get_num_nodes(request):
    try:
        num_nodes = int(request.GET.get('nodes', 10))
    except ValueError:
        num_nodes = 10
    return max(10, min(num_nodes, 100))


def home(request):
    return render(request, 'influence_game/index.html')


def distribution_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        nodes, edges = gt.random_proximity_probability(num_nodes)
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def tree_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        nodes, edges = gt.tree(num_nodes)
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    

def ladder_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        numNodes, nodes, edges = gt.ladder(num_nodes)
        response_data = {
            'numNodes': numNodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def square_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        numNodes, nodes, edges = gt.square_lattice(num_nodes)
        response_data = {
            'numNodes': numNodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def hexagon_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        numNodes, nodes, edges = gt.hexagon_lattice(num_nodes)
        response_data = {
            'numNodes': numNodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)


def triangle_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        numNodes, nodes, edges = gt.triangle_lattice(num_nodes)
        response_data = {
            'numNodes': numNodes,
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
    
    
def cycle_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        nodes, edges = gt.cycle(num_nodes)
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)

    
def random_proximity_view(request):
    if request.method == 'GET':
        num_nodes = get_num_nodes(request)
        nodes, edges = gt.random_proximity(num_nodes)
        response_data = {
            'nodes': nodes,
            'edges': edges,
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid HTTP Method'}, status=405)
