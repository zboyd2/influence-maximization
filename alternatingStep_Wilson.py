

"""
This function assigns the optimal next boundary node. See the bottom of the file for how to do multiple
nodes at once using the for loop. Right now you may specify which opinion to do next so if you want to
alternate teams or see which team has the overall optimal move that can be easily done with multiple
function calls.

I used a random test set. Without any starting boundary nodes it seems to do what I would expect by
adding the nodes with large edges first. It seems to behave strangely when there are already existing
boundary nodes. I believe it is error free and uses the math and a similar approach to what was
provided in the alternatingStep.m file provided in the DropBox.

Boundary Nodes are input as a matrix as described in the docstring. Each column represents a different
Sj and each row represents a different node. Thus, a 1 found at (3, 0) should represent node 3 (with zero
indexing) belonging to opinion 0.

I tested this using ksym = 2, but it should generalize without issue.

Let me know if there is anything you would change, or if there is anything missing.
"""

# alternatingStep
import numpy as np
import networkx as nx
import itertools as it


def alternating_step(graph_laplacian, bd_matrix, bd_set):
    """This function takes in the graph Laplacian, the previously selected
    boundary nodes as columns of a matrix with each column representing a
    different boundary node set, and which boundary node set that we are
    adding a node to. It returns the boundary node set matrix with the best
    node added to the appropriate column.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph

        bd_matrix ((n,ksym) ndarray): The previously selected boundary nodes where
        each row represents a node and each column represents a different boundary
        node set. Input as an array of zeros with the boundary nodes set to 1.
        Note, all entries must be 1s and 0s with all rows and columns summing
        to either 1 or 0.

        bd_set (int): The boundary node set to add a node to

    Returns:
        bd_matrix (ndarray): The original boundary node matrix with the new
        node updated in the specified column

        max_node (int): The node that was added to the boundary node set
    """
    # Get dimensions of bd_matrix
    n, ksym = bd_matrix.shape

    # Get all boundary nodes
    nodes = np.arange(n)
    bdnodes = nodes[np.sum(bd_matrix, axis=1) == 1]

    # Get the boundary nodes for the specified boundary set
    bd_set_nodes = bd_matrix[:, bd_set].astype(int)

    # Initialize the influence matrix
    influence = np.array([])

    # Initialize a list of all nodes that are not yet boundary nodes
    remaining_nodes = np.setdiff1d(nodes, bdnodes)

    # Compute the influence of each node
    for i in remaining_nodes:

        # Initialize the indicator vector
        bd_set_nodes_temp = bd_set_nodes.copy()

        # Create the Laplacian of the subgraph
        bd_update = np.sort(np.append(bdnodes, i))
        compliment = np.setdiff1d(nodes, bd_update)
        lsc = graph_laplacian[compliment, :][:, compliment]

        # Compute the boundary block matrix
        b = graph_laplacian[compliment, :][:, bd_update]

        # Update the indicator vector
        bd_set_nodes_temp[i] = 1

        # Calculate the influence using least squares to solve the linear system
        influence = np.append(influence, np.linalg.lstsq(lsc,
                                                         -b[:, bd_set_nodes_temp[bd_update].astype(bool)],
                                                         rcond=None)[0].sum())

    # Find the node with the maximum influence
    max_node = remaining_nodes[np.argmax(influence)]

    # Append the node to the boundary set
    bd_matrix[max_node, bd_set] = 1

    return bd_matrix, max_node, influence


def self_deprecating_step(graph_laplacian, bd_matrix, bd_set):
    """This function takes in the graph Laplacian, the previously selected
    boundary nodes as columns of a matrix with each column representing a
    different boundary node set, and which boundary node set that we are
    adding a node to. It returns the boundary node set matrix with the worst
    node added to the appropriate column.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph

        bd_matrix ((n,ksym) ndarray): The previously selected boundary nodes where
        each row represents a node and each column represents a different boundary
        node set. Input as an array of zeros with the boundary nodes set to 1.
        Note, all entries must be 1s and 0s with all rows and columns summing
        to either 1 or 0.

        bd_set (int): The boundary node set to add a node to

    Returns:
        bd_matrix (ndarray): The original boundary node matrix with the new
        node updated in the specified column

        max_node (int): The node that was added to the boundary node set

        influence (ndarray): The influence of each team
    """
    # Get dimensions of bd_matrix
    n, ksym = bd_matrix.shape

    # Get all boundary nodes
    nodes = np.arange(n)
    bdnodes = nodes[np.sum(bd_matrix, axis=1) == 1]

    # Get the boundary nodes for the specified boundary set
    bd_set_nodes = bd_matrix[:, bd_set].astype(int)

    # Initialize the influence matrix
    influence = np.array([])

    # Initialize a list of all nodes that are not yet boundary nodes
    remaining_nodes = np.setdiff1d(nodes, bdnodes)

    # Compute the influence of each node
    for i in remaining_nodes:

        # Initialize the indicator vector
        bd_set_nodes_temp = bd_set_nodes.copy()

        # Create the Laplacian of the subgraph
        bd_update = np.sort(np.append(bdnodes, i))
        compliment = np.setdiff1d(nodes, bd_update)
        lsc = graph_laplacian[compliment, :][:, compliment]

        # Compute the boundary block matrix
        b = graph_laplacian[compliment, :][:, bd_update]

        # Update the indicator vector
        bd_set_nodes_temp[i] = 1

        # Calculate the influence using least squares to solve the linear system
        influence = np.append(influence, np.linalg.lstsq(lsc,
                                                         -b[:, bd_set_nodes_temp[bd_update].astype(bool)],
                                                         rcond=None)[0].sum())

    # Find the node with the min influence
    min_node = remaining_nodes[np.argmin(influence)]

    # Append the node to the boundary set
    bd_matrix[min_node, bd_set] = 1

    return bd_matrix, min_node, influence


def component_number(m, tol=1e-08):
    """Find the number of connected components in a matrix."""
    eigvals = np.linalg.eig(m)[0]
    return m.shape[0] - np.sum(np.abs(eigvals) > tol)


def unique_close(arr, tol=1e-05, return_counts=False):
    """Find the unique rows in a numpy array within a tolerance.

    Parameters
    ----------
    arr : numpy array
        The array to find the unique rows of.
    tol : float, optional
        The tolerance within which two rows are considered equal. Default is 1e-05.
    return_counts : bool, optional
        If True, return the number of times each unique row appears in the array.
        Default is False.

    Returns
    -------
    unique_array : numpy array
        The array of unique rows.
    counts : numpy array
        The number of times each unique row appears in the array. Only returned if
        return_counts is True.
    """

    # if it is a 1D array, make it a 2D array with one column
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)

    # sort the array
    arr = np.sort(arr, axis=1)

    length = arr.shape[0]
    arr_copy = arr.copy()
    mask = np.ones(length) == 1

    for i in range(length):
        if mask[i]:  # if this row hasn't matched a previous row yet
            for j in range(i+1, length):
                if np.alltrue(np.abs(arr[i] - arr[j]) <= tol):  # if this row is the same as the next row
                    mask[j] = False
                    arr_copy[j] = arr_copy[i]  # make the close rows identical

    # now that the close ones are equal, we can use np.unique
    unique_array = np.unique(arr_copy, axis=0, return_counts=return_counts)

    return unique_array



def random_graph(n):
    """Create a random graph Laplacian of size n x n

    Args:
        n (int): The size of the graph Laplacian

    Returns:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph
    """
    # Throw an error if n not valid
    if n < 2:
        raise ValueError("n must be greater than or equal to 2")

    # Create a random adjacency matrix
    adj = np.random.randint(0, 2, size=(n, n))

    # Make the adjacency matrix symmetric
    adj = (adj + adj.T) // 2

    # Make the diagonal of the adjacency matrix all 0s
    np.fill_diagonal(adj, 0)

    # Create the degree matrix
    deg = np.diag(np.sum(adj, axis=1))

    # Create the graph Laplacian
    graph_laplacian = deg - adj

    # Check if the graph is connected
    if np.linalg.matrix_rank(deg) < n:
        return random_graph(n)
    elif component_number(graph_laplacian) > 1:
        return random_graph(n)

    return graph_laplacian


# Attempt 2


def alternating_step_2(graph_laplacian, bd_matrix, bd_set, node_to_add=None):
    """This differs from the first function in that it 1) does matrix
    multiplication instead of indexing in the lstsq function and 2) it
    uses betweenness centrality in the case that it is the first boundary
    node being chosen. 3) it also returns the influence had on each node in
    the graph instead of just the total influence.

    This function takes in the graph Laplacian, the previously selected
    boundary nodes as columns of a matrix with each column representing a
    different boundary node set, which boundary node set that we are adding
    a node to, and the number of boundary node sets. It returns the boundary
    node set matrix with the best node added to the appropriate column.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph

        bd_matrix ((n,ksym) ndarray): The previously selected boundary nodes where
        each row represents a node and each column represents a different boundary
        node set. Input as an array of zeros with the boundary nodes set to 1.
        Note, all entries must be 1s and 0s with all rows and columns summing
        to either 1 or 0.

        bd_set (int): The boundary node set to add a node to

    Returns:
        bd_matrix (ndarray): The original boundary node matrix with the new
        node updated in the specified column

        max_node (int): The node that was added to the boundary node set
    """
    # Get dimensions of bd_matrix
    n, ksym = bd_matrix.shape

    # Get all boundary nodes
    nodes = np.arange(n)
    bdnodes = nodes[np.sum(bd_matrix, axis=1) == 1]

    # Get the boundary nodes for the specified boundary set
    bd_set_nodes = bd_matrix[:, bd_set].astype(int)

    # Initialize the influence matrix
    influence = np.array([])
    influence_mat = []

    # Initialize a list of all nodes that are not yet boundary nodes
    remaining_nodes = np.setdiff1d(nodes, bdnodes)

    # Compute the influence of each node
    for i in remaining_nodes:
        # Initialize the indicator vector
        bd_set_nodes_temp = bd_set_nodes.copy()

        # Create the Laplacian of the subgraph
        bd_update = np.sort(np.append(bdnodes, i))
        compliment = np.setdiff1d(nodes, bd_update)
        lsc = graph_laplacian[compliment, :][:, compliment]

        # Compute the boundary block matrix
        b = graph_laplacian[compliment, :][:, bd_update]

        # Update the indicator vector
        bd_set_nodes_temp[i] = 1

        # Calculate the influence using least squares to solve the linear system
        influence = np.append(influence, np.linalg.lstsq(lsc,
                                                         -b @ bd_set_nodes_temp[bd_update],
                                                         rcond=None)[0].sum())

        influence_mat.append(np.linalg.lstsq(lsc,
                                             -b @ bd_set_nodes_temp[bd_update],
                                             rcond=None)[0])

    if node_to_add is not None:  # If a node is specified, add that node to the boundary set
        max_node = node_to_add
        influence_val = influence[np.argwhere(remaining_nodes == node_to_add)[0][0]]

    # if bd_matrix is empty, instead of a random node, choose the node with the highest betweenness centrality
    elif np.array_equal(bd_matrix, np.zeros_like(bd_matrix)):
        max_node = np.argmax(
            np.array(list(nx.betweenness_centrality(
                nx.DiGraph(graph_laplacian), normalized=False).values())))

        influence_val = influence[max_node]

    else:  # Otherwise, add the node with the highest influence
        max_node = remaining_nodes[np.argmax(influence)]
        influence_val = np.max(influence)

    # Append the node to the boundary set
    bd_matrix[max_node, bd_set] = 1

    total_influence = influence

    return bd_matrix, max_node, total_influence, influence_val


def get_influence(graph_laplacian, bd_matrix):
    """This function takes in the graph Laplacian and the previously
    selected boundary nodes as columns of a matrix with each column
    representing a different boundary node set. It returns the influence
    had by each team in the graph.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph

        bd_matrix ((n,ksym) ndarray): The previously selected boundary nodes where
        each row represents a node and each column represents a different boundary
        node set. Input as an array of zeros with the boundary nodes set to 1.
        Note, all entries must be 1s and 0s with all rows and columns summing
        to either 1 or 0.

    Returns:
        influence ((n,) ndarray): The influence of each team in the graph
    """

    # Get dimensions of bd_matrix
    n, ksym = bd_matrix.shape
    influence = []

    # Get all boundary nodes
    bdnodes = np.arange(n)[np.sum(bd_matrix, axis=1) == 1]

    for i in range(ksym - 1):

        # Get the boundary nodes for the specified boundary set
        bd_set_nodes = bd_matrix[:, i].astype(int)

        # Create the Laplacian of the subgraph
        compliment = np.setdiff1d(np.arange(n), bdnodes)
        lsc = graph_laplacian[compliment, :][:, compliment]

        # Compute the boundary block matrix
        b = graph_laplacian[compliment, :][:, bdnodes]

        # Calculate the influence using least squares to solve the linear system
        influence.append(np.linalg.lstsq(lsc, -b @ bd_set_nodes[bdnodes], rcond=None)[0].sum())

        # Add influence of each node in the specific boundary set
        influence[i] += bd_set_nodes.sum()

    influence.append(sum(graph_laplacian.shape[0] - np.array(influence)))

    return np.array(influence)


def get_all_configs(graph_laplacian, nodes_per_team, ksym):
    """Given a graph, number of turns, and number of teams gets all configurations of boundary nodes and their influence

    :param graph_laplacian: Laplacian of the graph
    :param nodes_per_team: Number of nodes per team
    :param ksym: Number of teams
    :return: Influence of each team by configuration of boundary nodes
    :return: All possible configurations of boundary nodes
    """

    influences_by_perm = []

    # Get all permutations of boundary nodes
    permutations = list(it.permutations(np.arange(graph_laplacian.shape[0]), nodes_per_team * ksym))

    for perm in permutations:  # For each permutation, get the influence of each team

        bd_matrix = np.zeros((graph_laplacian.shape[0], ksym))

        for i in range(nodes_per_team):  # Set the boundary nodes for each team
            for j in range(ksym):
                bd_matrix[perm[j + i * ksym], j] = 1

        # Get the influence of each team
        influences_by_perm.append(get_influence(graph_laplacian, bd_matrix))

    return np.array(influences_by_perm), np.array(permutations)


# Not working for ksym > 2
def opponent_dumb(graph_laplacian, nodes_per_team, ksym):
    """This function takes in the graph Laplacian, the boundary matrix, and the number of nodes per team. It returns
    the node that minimizes the influence of the opponent. This is done by calculating the average influence of each
    node as a potential move and then selecting the node that minimizes the average influence of the opponent.

    Args:
        graph_laplacian ((n, n) ndarray): The graph Laplacian for the given graph

        nodes_per_team (int): The number of nodes per team

        ksym (int): The number of teams

    Returns:
        max_node (int): The node that minimizes the influence of the opponent

        influence_val ((1, ksym) ndarray): The influence of the player and the opponent in that order
    """

    # Get all possible configurations of boundary nodes and their influence
    all_influences, permutations = get_all_configs(graph_laplacian=graph_laplacian,
                                               nodes_per_team=nodes_per_team,
                                               ksym=ksym)

    c = 0
    config = np.array([])
    nodes = np.arange(graph_laplacian.shape[0])
    mask = np.ones(permutations.shape[0], dtype=bool)

    while c < nodes_per_team:  # while there are turns left

        # Get user input
        selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"))

        # Add the node to the boundary set
        config = np.append(config, selected_node)

        # Update the mask which is used to narrow down the possible configurations to the ones that are still possible
        mask &= permutations[:, config.shape[0] - 1] == selected_node

        for i in range(ksym - 1):  # for each remaining team
            influ_temp = np.array([])

            # if it is the last node, do the self-deprecating step (opposite of greedy)
            if c == nodes_per_team - 1 and i == ksym - 2:  # on the last node do greedy

                # create the boundary matrix for the greedy step
                bd_matrix = np.zeros((graph_laplacian.shape[0], ksym))
                for j in range(nodes_per_team * ksym - 1):
                    bd_matrix[int(config[j]), j % ksym] = 1

                bd_set = ksym - 1

                greedy_node = self_deprecating_step(graph_laplacian, bd_matrix, bd_set)[1]

                config = np.append(config, greedy_node)
                mask &= permutations[:, config.shape[0] - 1] == int(config[-1])

            else:
                for j in np.setdiff1d(nodes, config):  # for each node not in the boundary set

                    # create a temporary mask to narrow down the possible configurations for the current node
                    mask_temp = mask & (permutations[:, config.shape[0]] == j)

                    # calculate the average influence of the current node
                    # (to find the least average of all possible configurations for all remaining nodes)
                    influ_temp = np.append(
                        influ_temp,
                        all_influences[:, 0][mask_temp].sum() / mask_temp.sum()) if mask_temp.sum() > 0 else np.append(
                        influ_temp,
                        np.inf)

                # update the configuration with the node that minimizes the influence of the opponent
                config = np.append(config, np.setdiff1d(nodes, config)[np.argmin(influ_temp)])

                # update the mask to narrow down the possible configurations to the ones that are still possible
                mask &= permutations[:, config.shape[0] - 1] == int(config[-1])
        c += 1

    # convert the configuration to a tuple, so it can be used as a key in a dictionary
    permutations = list(map(tuple, permutations))

    # create a dictionary of all possible configurations and their influence to quickly look up the influence of config
    influ_dict = dict(zip(permutations, all_influences))

    return config, influ_dict[tuple(config)]  # return the configuration and its influence


# Not working for ksym > 2
def opponent_dumb_2(graph_laplacian, nodes_per_team, ksym, player=False):
    """This function takes in the graph Laplacian, the boundary matrix, and the number of nodes per team. It returns
    the node that minimizes the influence of the opponent. This is done by minimizing the amount of winning moves the
    opponent can make.

    Args:
        graph_laplacian ((n, n) ndarray): The graph Laplacian for the given graph

        nodes_per_team (int): The number of nodes per team

        ksym (int): The number of teams

        player (bool): If the player is the one making the move. If False, the player moves default to the
        nodes 0 and 1. If 1 is chosen by the opponent it will be changed to 2.

    Returns:
        max_node (int): The node that minimizes the influence of the opponent

        influence_val ((1, ksym) ndarray): The influence of the player and the opponent in that order
    """

    # Get all possible configurations of boundary nodes and their influence
    all_influences, permutations = get_all_configs(graph_laplacian=graph_laplacian,
                                               nodes_per_team=nodes_per_team,
                                               ksym=ksym)

    c = 0
    config = np.array([])
    nodes = np.arange(graph_laplacian.shape[0])
    mask = np.ones(permutations.shape[0], dtype=bool)

    while c < nodes_per_team:  # while there are turns left

        if player:
            # Get user input
            selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"))
        else:
            if config.shape[0] == 0:
                selected_node = 0
            elif config[-1] == 1:
                selected_node = 2
            else:
                selected_node = 1

        # Add the node to the boundary set
        config = np.append(config, selected_node)

        # Update the mask which is used to narrow down the possible configurations to the ones that are still possible
        mask &= permutations[:, config.shape[0] - 1] == selected_node

        for i in range(ksym - 1):  # for each remaining team
            influ_temp = np.array([])

            # if it is the last node, do the self-deprecating step (opposite of greedy)
            if c == nodes_per_team - 1 and i == ksym - 2:  # on the last node do greedy

                # create the boundary matrix for the greedy step
                bd_matrix = np.zeros((graph_laplacian.shape[0], ksym))
                for j in range(nodes_per_team * ksym - 1):
                    bd_matrix[int(config[j]), j % ksym] = 1

                bd_set = ksym - 1

                greedy_node = alternating_step(graph_laplacian, bd_matrix, bd_set)[1]

                config = np.append(config, greedy_node)
                mask &= permutations[:, config.shape[0] - 1] == int(config[-1])

            else:
                for j in np.setdiff1d(nodes, config):  # for each node not in the boundary set

                    # create a temporary mask to narrow down the possible configurations for the current node
                    mask_temp = mask & (permutations[:, config.shape[0]] == j)

                    # calculate the average influence of the current node
                    # (to find the least amount of winning moves for the opponent)
                    influ_temp = np.append(
                        influ_temp,
                        (all_influences[:, 0][mask_temp] < (graph_laplacian.shape[0] / 2)).sum())

                # update the configuration with the node that minimizes the influence of the opponent
                config = np.append(config, np.setdiff1d(nodes, config)[np.argmax(influ_temp)])

                # update the mask to narrow down the possible configurations to the ones that are still possible
                mask &= permutations[:, config.shape[0] - 1] == int(config[-1])
        c += 1

    # convert the configuration to a tuple, so it can be used as a key in a dictionary
    permutations = list(map(tuple, permutations))

    # create a dictionary of all possible configurations and their influence to quickly look up the influence of config
    influ_dict = dict(zip(permutations, all_influences))

    return config, influ_dict[tuple(config)]  #influ_dict  # return the configuration and its influence





### TESTING AREA ###

ex4 = np.array([[1, -1, 0, 0],
                [-1, 2, -1, 0],
                [0, -1, 2, -1],
                [0, 0, -1, 1]])

ex5 = np.array([[1, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -1, 1]])

ex6 = np.array([[1, -1, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0],
                [0, -1, 2, -1, 0, 0],
                [0, 0, -1, 2, -1, 0],
                [0, 0, 0, -1, 2, -1],
                [0, 0, 0, 0, -1, 1]])

ex7 = np.array([[1, -1, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, 0],
                [0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, -1, 2, -1],
                [0, 0, 0, 0, 0, -1, 1]])

# all_influences, L = get_all_configs(ex7, 3, 2)
# # print(L)
# L = np.array(L)
# print(len(L))
# G = opponent_greedy(ex7, 3, 2).astype(int)
# print(len(G))
# print(L)
# print(G)
# print(L.shape)
#
# # indices = np.where(G.reshape(G.size, 1) == L)[1]
# indices = np.where(G[:, None] == L[None, :])[1]
# print(len(indices))
# print(L[indices].shape)
#
# Me trying to get the indices where the greedy occurs in the
# all_configs array.


# print(opponent_dumb(ex7, 3, 2))

# x = np.array([1, 2, 3, 4, 5, 6, 7])
# y = np.array([2, 4, 6, 8, 10, 12, 14])
# z = dict(zip(x, y))

# print(z)
# print(z[1])


# ex_graph = random_graph(9)
# # print(ex_graph)
# # visualize ex_graph using networkx
# vis_ex_graph = ex_graph.copy()
# # set diagonal to 0
# np.fill_diagonal(vis_ex_graph, 0)
# # make it a circle graph
# pos = nx.circular_layout(nx.Graph(vis_ex_graph))
# # draw the graph
# nx.draw(nx.Graph(vis_ex_graph), pos, with_labels=True)
# plt.show()

# nodes_per_team = 2
# ksym = 2

# print(opponent_dumb(ex_graph, 2, 2))
# conf, influ, all_influ = opponent_dumb_2(ex_graph, nodes_per_team, ksym)
# print(conf, influ)
# print(opponent_greedy(ex_graph, nodes_per_team, ksym))
# print(opponent_dumb(ex_graph, nodes_per_team, ksym))

# bd_matrix = np.zeros((ex_graph.shape[0], ksym))
# nodes = np.arange(ex_graph.shape[0])
# config = np.array([])
# c = 0
# do_it = False
# if do_it is True:
#     while c < nodes_per_team:
#         selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"))
#         config = np.append(config, selected_node)
#
#         bd_matrix[selected_node, 0] = 1
#
#         bd_matrix, max_node = alternating_step(ex_graph, bd_matrix, 1)[0:2]
#
#         config = np.append(config, max_node)
#
#         c += 1
#
#     print(config, all_influ[tuple(config)])
#
# bd_matrix = np.zeros((ex_graph.shape[0], ksym))
# bd_matrix[3, 0] = 1
# bd_matrix[5, 1] = 1
# bd_matrix[4, 0] = 1

# print(opponent_greedy(ex_graph, bd_matrix, nodes_per_team, ksym))


# # how to store graphs using dstack
# for i in range(4):
#     ex_graph = random_graph(9)
#     if i == 0:
#         ex_graphs = ex_graph
#         print(ex_graph)
#         continue
#
#     # efficient way to store the graphs using np.dstack
#     ex_graphs = np.dstack((ex_graphs, ex_graph))
#
# print(ex_graphs[:, :, 0].astype(int))


