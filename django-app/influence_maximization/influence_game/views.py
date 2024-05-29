from django.shortcuts import render
from django.http import JsonResponse
import json
import numpy as np
import influence_game.algorithms.graph_types as gt
import influence_game.algorithms.influence_maximization_algorithms as im


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


def bot_move(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        laplacian = np.array(data['laplacian'])
        config = np.array(data['config'])
        difficulty = data['difficulty']
        num_turns = int(data['numTurns'])

        if difficulty == 'easy':
            move = int(im.gui_easy_opponent(laplacian, config))
        elif difficulty == 'medium':
            move = int(im.gui_greedy_algorithm(laplacian, config))
        elif difficulty == 'hard':
            move = int(im.gui_minimax_algorithm_opt(laplacian, config, num_turns))
        else:
            return JsonResponse({'error': 'Invalid difficulty'}, status=400)
        return JsonResponse({'move': move})

    return JsonResponse({'error': 'Invalid Method'}, status=405)
