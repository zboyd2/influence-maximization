import networkx as nx
import numpy as np
import random
import scipy.stats as ss
from influence_game.algorithms.global_constants import SCREEN_SIZE, NODE_RADIUS, NUM_NODES

X_MARGIN = int(SCREEN_SIZE[0] * 0.10)
Y_MARGIN = int(SCREEN_SIZE[1] * 0.12)

'''
Graph Generation Functions

Each function returns:
    -A list of node coordinates, stored as an array of tuples
    -A list of edge pairs, stored as an array of tuples
    -The number of nodes IF IT WAS CHANGED by the graph generation
'''

def random_proximity():
    nodes = random_node_placement(NUM_NODES)
    edge_threshold = np.linalg.norm(np.array(SCREEN_SIZE)) * 0.095  #Set edge threshold as a fraction of the diagonal
    edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) < edge_threshold]
    return nodes, edges

def random_proximity_probability():
    nodes = random_node_placement(NUM_NODES)
    edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if 2 * (1 - ss.norm.cdf((np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) / (NODE_RADIUS * 2)), loc=0, scale=3)) > random.random()]
    return nodes, edges

def tree():
    G = nx.full_rary_tree(4, NUM_NODES)
    return list(scale_nodes_to_screen(nx.kamada_kawai_layout(G))), list(G.edges)

def ladder():
    G = nx.ladder_graph(int(NUM_NODES / 2))
    return 2 * int(NUM_NODES/2), list(scale_nodes_to_screen(nx.spring_layout(G))), list(G.edges)

def cycle():
    G = nx.cycle_graph(NUM_NODES)
    return list(scale_nodes_to_screen(nx.circular_layout(G))), list(G.edges)

def square_lattice():
    length = int(np.floor(np.sqrt(NUM_NODES)))
    new_node_count = int(length ** 2)
    H = nx.grid_graph(dim=(length, length))
    node_cords = list(H.nodes)
    G = nx.relabel_nodes(H, {node_cords[i] : i  for i in range(new_node_count)})
    return new_node_count, list(scale_nodes_to_screen({i: node_cords[i] for i in range(new_node_count)})), list(G.edges)

def hexagon_lattice():
    n = int(np.rint(np.sqrt(NUM_NODES)) / 2)
    m = int(np.rint((NUM_NODES - 2 * n) / (2 * n + 2)))
    new_node_count = 2 * (n + n * m + m)
    H = nx.hexagonal_lattice_graph(n, m)
    node_names = [*nx.get_node_attributes(H, 'pos')]
    node_cords = [*nx.get_node_attributes(H, 'pos').values()]
    G = nx.relabel_nodes(H, {node_names[i] : i  for i in range(new_node_count)})
    return new_node_count, list(scale_nodes_to_screen({i: node_cords[i] for i in range(new_node_count)})), list(G.edges)

def triangle_lattice():
    n = int(np.sqrt(NUM_NODES))
    if n % 2 == 0:
        n += 1
    m = int(np.rint((2 * NUM_NODES - 2 * n - 2) / (n + 1)))
    new_node_count = int(0.5 * n * m + 0.5 * m + n + 1)
    H = nx.triangular_lattice_graph(n, m)
    node_names = [*nx.get_node_attributes(H, 'pos')]
    node_cords = [*nx.get_node_attributes(H, 'pos').values()]
    G = nx.relabel_nodes(H, {node_names[i] : i  for i in range(new_node_count)})
    return new_node_count, list(scale_nodes_to_screen({i: node_cords[i] for i in range(new_node_count)})), list(G.edges)

#Helper Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def scale_nodes_to_screen(positions):
    """
    Args:
        positions (dictionary): A dictionary that stores the positions of each of
        the nodes with keys being the node names and values being coordinate tuples

    Returns:
        a list of node coordinate tuples, scaled to fit well in the graphgame screen
    """

    x_range = SCREEN_SIZE[0] - 2 * X_MARGIN
    y_range = SCREEN_SIZE[1] - 2 * Y_MARGIN
    pos_list = positions.values()
    
    x_cords = [cords[0] for cords in pos_list]
    y_cords = [cords[1] for cords in pos_list]
    min_inputed_x = min(x_cords)
    min_inputed_y = min(y_cords)
    input_x_range = max(x_cords) - min_inputed_x
    input_y_range = max(y_cords) - min_inputed_y
    
    return [((x - min_inputed_x) * (x_range) / (input_x_range) + X_MARGIN, (y - min_inputed_y) * (y_range) / (input_y_range) + Y_MARGIN) for x, y in pos_list]

def random_node_placement(N):
    nodes = []
    while len(nodes) < N:
        new_node = (random.randint(X_MARGIN, SCREEN_SIZE[0] - X_MARGIN), random.randint(Y_MARGIN, SCREEN_SIZE[1] - Y_MARGIN))
        if all(np.linalg.norm(np.array(new_node) - np.array(existing_node)) > 2 * NODE_RADIUS for existing_node in nodes): 
            nodes.append(new_node)
    return nodes