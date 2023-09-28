import alternatingStep_Wilson as alt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it

# # It's like greedy_2, but it is not as good
# def opponent_least_best(graph_laplacian, current_config, nodes_per_team, ksym=2):
#     """Oppenent model that chooses the move that minimizes the best payoff for the other player.
#     Implemented for 2 players.
#     """
#
#     configs_list = []
#     max_nodes = nodes_per_team * ksym
#
#     # initialize for the first recursive step
#     nodes_remaining_start = np.arange(graph_laplacian.shape[0])
#
#     bd_matrix_start = np.zeros((graph_laplacian.shape[0], ksym))
#
#     # Each config will be a list of boundary nodes in the order they were chosen by each team sequentially
#     config_start = np.array([])
#
#
#     def recursive_step(graph_laplacian, bd_matrix, nodes_remaining, config):
#         if int(bd_matrix.sum()) == int(max_nodes):  # Base case
#             configs_list.append(tuple(config.copy()))
#             return
#
#         for node in nodes_remaining:  # loops through each node not already a boundary node
#             turn = bd_matrix.sum() % ksym
#             if turn == 0:
#                 bd_matrix[node, 0] = 1
#                 config = np.append(config, node)
#                 turn = 1
#
#             for k in range(int(turn), ksym):  # does the greedy for the other teams
#                 bd_matrix, node_added = alt.alternating_step(graph_laplacian, bd_matrix, k)[0:2]
#                 config = np.append(config, node_added)
#
#             nodes_update = np.setdiff1d(nodes_remaining, config)
#             recursive_step(graph_laplacian, bd_matrix, nodes_update, config)
#             bd_matrix[node, 0] = 0
#
#
#             bd_matrix[config[-(ksym - 1):].astype(int), np.arange(1, ksym)] = 0
#             config = config[:-ksym]
#
#     recursive_step(graph_laplacian=graph_laplacian, bd_matrix=bd_matrix_start.copy(),
#                    nodes_remaining=nodes_remaining_start.copy(), config=config_start.copy())
#
#     return np.array(configs_list), len(configs_list)


def opponent_greedy_2(graph_laplacian, bd_matrix_start, nodes_per_team, ksym):

    configs_list = []
    max_nodes = nodes_per_team * ksym
    nodes = np.arange(graph_laplacian.shape[0])

    # initialize for the first recursive step
    turn_original = bd_matrix_start.sum()
    nodes_remaining_start = np.arange(graph_laplacian.shape[0])
    nodes_remaining_start = np.delete(nodes_remaining_start, np.sum(bd_matrix_start, axis=1).nonzero()[0])

    # Each config will be a list of boundary nodes in the order they were chosen by each team sequentially
    config_start = np.array([])

    if bd_matrix_start.sum() == 0:
        pass
    else:  # if there are already boundary nodes, then we need to add them to the config_start
        config_temp = [nodes[bd_matrix_start[:, i].astype(int).nonzero()] for i in range(ksym)]

        for i in range(len(config_temp[0])):
            for j in range(ksym):
                if len(config_temp[j]) > i:
                    config_start = np.append(config_start, config_temp[j][i])


    def recursive_step(graph_laplacian, bd_matrix, nodes_remaining, config):
        if int(bd_matrix.sum()) == int(max_nodes):  # Base case
            configs_list.append(tuple(config.copy()))
            return

        for node in nodes_remaining:  # loops through each node not already a boundary node
            turn = bd_matrix.sum() % ksym
            if turn == 0:
                bd_matrix[node, 0] = 1
                config = np.append(config, node)
                turn = 1

            for k in range(int(turn), ksym):  # does the greedy for the other teams
                bd_matrix, node_added = alt.alternating_step(graph_laplacian, bd_matrix, k)[0:2]
                config = np.append(config, node_added)

            nodes_update = np.setdiff1d(nodes_remaining, config)
            recursive_step(graph_laplacian, bd_matrix, nodes_update, config)
            bd_matrix[node, 0] = 0

            # This is to reset the boundary matrix and config to the original state
            if config.shape[0] < turn_original + ksym:
                return
            else:
                bd_matrix[config[-(ksym-1):].astype(int), np.arange(1, ksym)] = 0
                config = config[:-ksym]

    recursive_step(graph_laplacian=graph_laplacian, bd_matrix=bd_matrix_start.copy(), nodes_remaining=nodes_remaining_start.copy(), config=config_start.copy())

    return np.array(configs_list), len(configs_list)




def opponent_smart(graph_laplacian, nodes_per_team, ksym, player=False):
    """This function takes in the graph Laplacian, the boundary matrix, and the number of nodes per team. It calculates
    the best move for the opponent by considering what would minimize the best next move for the player. It returns
    the configuration with the nodes chosen and the influence of that configuration.

    Parameters
    ----------
    graph_laplacian : numpy array
        The graph Laplacian of the graph
    nodes_per_team : int
        The number of nodes per team
    ksym : int
        The number of teams
    player : bool
        If True, then the user will be prompted to select the nodes to add to the boundary set. If False, then the
        player nodes default to the first nodes in the graph.

    Returns
    -------
    config : numpy array
        The configuration of boundary nodes
    influence : float
        The influence of the configuration
    """

    # Get all possible configurations of boundary nodes and their influence
    all_influences, permutations = alt.get_all_configs(graph_laplacian=graph_laplacian,
                                               nodes_per_team=nodes_per_team,
                                               ksym=ksym)

    c = 0
    config = np.array([])
    nodes = np.arange(graph_laplacian.shape[0])
    mask = np.ones(permutations.shape[0], dtype=bool)

    while c < nodes_per_team:  # while there are turns left

        if player:
            # Get user input
            while True:
                try:
                    selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"
                                              f"\nNodes selected so far: {config}"))
                    if selected_node in config:
                        print("That node has already been selected. Try again.")
                        continue
                    break
                except ValueError:
                    print("That's not a valid node. Try again.")
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

                greedy_node = alt.alternating_step(graph_laplacian, bd_matrix, bd_set)[1]

                config = np.append(config, greedy_node)
                mask &= permutations[:, config.shape[0] - 1] == int(config[-1])

            else:
                for j in np.setdiff1d(nodes, config):  # for each node not in the boundary set

                    # create a temporary mask to narrow down the possible configurations for the current node
                    mask_temp = mask & (permutations[:, config.shape[0]] == j)

                    # calculate the best possible move for the opponent
                    # (by minimizing the best possible move for the player)
                    influ_temp = np.append(
                        influ_temp,
                        np.max(all_influences[:, 0][mask_temp]))

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


def opponent_greedy_complete(graph_laplacian, nodes_per_team, ksym, player=False):
    """This function takes in the graph Laplacian, the number of nodes per team, the number of teams, and if the player
    is manually selecting the nodes. It calculates the greedy move for the opponent.

    Parameters
    ----------
    graph_laplacian : numpy array
        The graph Laplacian of the graph
    nodes_per_team : int
        The number of nodes per team
    ksym : int
        The number of teams
    player : bool
        If True, then the user will be prompted to select the nodes to add to the boundary set. If False, then the
        player nodes default to the first nodes in the graph.

    Returns
    -------
    config : numpy array
        The configuration of boundary nodes
    influence : float
        The influence of the configuration
    """

    all_influ, all_configs = alt.get_all_configs(graph_laplacian, nodes_per_team, ksym)

    # convert the configurations to tuples, so they can be used as a key in a dictionary
    permutations = list(map(tuple, all_configs))
    dictionary = dict(zip(permutations, all_influ))

    c = 0
    nodes = np.arange(graph_laplacian.shape[0])
    config = np.array([])
    bd_matrix = np.zeros((graph_laplacian.shape[0], ksym))


    while c < nodes_per_team:  # while there are turns left

        if player:  # if the player is selecting the nodes
            selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"))
        else:  # if the player is not selecting the nodes default to the first nodes in the graph
            if config.shape[0] == 0:
                selected_node = 0
            elif config[-1] == 1:
                selected_node = 2
            else:
                selected_node = 1

        # Add the node to the boundary set
        config = np.append(config, selected_node)

        # Update the boundary matrix
        bd_matrix[selected_node, 0] = 1

        # Do the greedy step
        bd_matrix, max_node = alt.alternating_step(graph_laplacian, bd_matrix, 1)[0:2]

        # Add the node to the boundary set
        config = np.append(config, max_node)

        c += 1

    return config, dictionary[tuple(config)]  # return the configuration and its influence


def opponent_smart_step(graph_laplacian, config, nodes_per_team, ksym, all_influences, permutations):
    """This function takes in the graph Laplacian, current configuration, the number of nodes per team, the number of
    teams, the influence of all configurations, and all possible configurations. It calculates the best possible move
    for the opponent. This is done by minimizing the best possible move for the player.

    Parameters
    ----------
    graph_laplacian : numpy array
        The graph Laplacian of the graph
    config : numpy array
        The current configuration of boundary nodes
    nodes_per_team : int
        The number of nodes per team
    ksym : int
        The number of teams
    all_influences : numpy array
        The influence of all possible configurations
    permutations : numpy array
        All possible configurations

    Returns
    -------
    config : numpy array
        The configuration of boundary nodes
    influence : float
        The influence of the configuration if it is the last node, otherwise it returns an empty array
    """

    team = config.shape[0] % ksym
    nodes = np.arange(graph_laplacian.shape[0])
    mask = np.ones(permutations.shape[0], dtype=bool)
    for m in range(config.shape[0]):
        mask &= permutations[:, m] == config[m]

    influ_temp = np.array([])

    # if it is the last node, do the self-deprecating step (opposite of greedy)
    if config.shape[0] == nodes_per_team * ksym - 1:  # on the last node do greedy

        # create the boundary matrix for the greedy step
        bd_matrix = np.zeros((graph_laplacian.shape[0], ksym))
        for j in range(nodes_per_team * ksym - 1):
            bd_matrix[int(config[j]), j % ksym] = 1

        bd_set = ksym - 1

        greedy_node = alt.alternating_step(graph_laplacian, bd_matrix, bd_set)[1]

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
                np.max(all_influences[:, (team + 1) % ksym][mask_temp]))

        # update the configuration with the node that minimizes the influence of the opponent
        config = np.append(config, np.setdiff1d(nodes, config)[np.argmin(influ_temp)])

        # update the mask to narrow down the possible configurations to the ones that are still possible
        mask &= permutations[:, config.shape[0] - 1] == int(config[-1])



    # convert the configuration to a tuple, so it can be used as a key in a dictionary
    permutations = list(map(tuple, permutations))

    # create a dictionary of all possible configurations and their influence to quickly look up the influence of config
    influ_dict = dict(zip(permutations, all_influences))

    # return the configuration and its influence if it is the last node,
    # otherwise return the configuration and an empty array
    return config, influ_dict[tuple(config)] if config.shape[0] == nodes_per_team * ksym else np.array([])


def opponent_smart_recursive_check(graph_laplacian, nodes_per_team, ksym):
    """Checks all configurations when the opponent is using the opponent_smart_step function.

    Parameters
    ----------
    graph_laplacian : numpy array
        The graph Laplacian of the graph
    nodes_per_team : int
        The number of nodes per team
    ksym : int
        The number of teams

    Returns
    -------
    configs_array : numpy array
        A list of all possible configurations
    length : int
        The number of possible configurations"""

    all_influences, permutations = alt.get_all_configs(graph_laplacian=graph_laplacian,
                                                       nodes_per_team=nodes_per_team,
                                                       ksym=ksym)

    configs_list = []
    max_nodes = nodes_per_team * ksym
    nodes = np.arange(graph_laplacian.shape[0])

    # initialize for the first recursive step
    nodes_remaining_start = np.arange(graph_laplacian.shape[0])

    # Each config will be a list of boundary nodes in the order they were chosen by each team sequentially
    config_start = np.array([])

    def recursive_step(graph_laplacian, nodes_remaining, config):
        if config.shape[0] == int(max_nodes):  # Base case
            configs_list.append(tuple(config.copy()))
            return

        for node in nodes_remaining:  # loops through each node not already a boundary node
            config = np.append(config, node)  # add the node to the boundary set

            # do the smart step for the other team
            config = opponent_smart_step(graph_laplacian, config, nodes_per_team, ksym, all_influences, permutations)

            nodes_update = np.setdiff1d(nodes_remaining, config)
            recursive_step(graph_laplacian, nodes_update, config)

            # This is to reset the boundary matrix and config to the original state
            config = config[:-ksym]

    recursive_step(graph_laplacian=graph_laplacian,
                   nodes_remaining=nodes_remaining_start.copy(),
                   config=config_start.copy())

    return np.array(configs_list), len(configs_list)

def opponent_minimax_recursive_check(graph_laplacian, nodes_per_team, ksym):
    """Checks all configurations when the opponent is using the opponent_smart_step function.

    Parameters
    ----------
    graph_laplacian : numpy array
        The graph Laplacian of the graph
    nodes_per_team : int
        The number of nodes per team
    ksym : int
        The number of teams

    Returns
    -------
    configs_array : numpy array
        A list of all possible configurations
    length : int
        The number of possible configurations"""

    all_influences, permutations = alt.get_all_configs(graph_laplacian=graph_laplacian,
                                                       nodes_per_team=nodes_per_team,
                                                       ksym=ksym)

    configs_list = []
    max_nodes = nodes_per_team * ksym
    nodes = np.arange(graph_laplacian.shape[0])

    # initialize for the first recursive step
    nodes_remaining_start = np.arange(graph_laplacian.shape[0])

    # Each config will be a list of boundary nodes in the order they were chosen by each team sequentially
    config_start = np.array([])

    def recursive_step(graph_laplacian, nodes_remaining, config):
        if config.shape[0] == int(max_nodes):  # Base case
            configs_list.append(tuple(config.copy()))
            return

        for node in nodes_remaining:  # loops through each node not already a boundary node
            config = np.append(config, node)  # add the node to the boundary set

            # do the smart step for the other team
            _, next = minimax(graph_laplacian, config, nodes_per_team, ksym, 1)

            config = np.append(config, next)

            nodes_update = np.setdiff1d(nodes_remaining, config)
            recursive_step(graph_laplacian, nodes_update, config)

            # This is to reset the boundary matrix and config to the original state
            config = config[:-ksym]

    recursive_step(graph_laplacian=graph_laplacian,
                   nodes_remaining=nodes_remaining_start.copy(),
                   config=config_start.copy())

    return np.array(configs_list), len(configs_list)


def get_influence_from_config(graph_laplacian, config):
    """This function takes in the graph Laplacian and the previously
    selected boundary nodes as columns of a matrix with each column
    representing a different boundary node set. It returns the influence
    had by each team in the graph.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for the given graph

        config (ndarray): The previously selected boundary nodes in sequential order alternating team

    Returns:
        influence ((2,) ndarray): The influence of each team in the graph
    """

    # Get all boundary nodes for team 1 and create an n length array with 1's in those positions
    team_1_nodes = config[1::2]
    team_1_bd = np.zeros(graph_laplacian.shape[0]).astype(int)
    team_1_bd[team_1_nodes] = 1

    # Create the Laplacian of the subgraph
    compliment = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)
    lsc = graph_laplacian[compliment, :][:, compliment]

    # Compute the boundary block matrix
    b = graph_laplacian[compliment, :][:, config]

    # Calculate the influence using least squares to solve the linear system
    influence_1 = np.linalg.lstsq(lsc, -b @ team_1_bd[config], rcond=None)[0].sum()

    # Add influence of each node in the specific boundary set
    influence_1 += team_1_bd.sum()

    # Calculate the influence of the other team
    influence_0 = graph_laplacian.shape[0] - influence_1

    return np.array((influence_0, influence_1))


def minimax(graph_laplacian, config, nodes_per_team, ksym, turn):

    if len(config) == nodes_per_team * ksym:
        return get_influence_from_config(graph_laplacian, config.astype(int))[0], None

    nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

    if turn == 0:  # Maximizing team
        value = -np.inf
        best_node = None
        for node in nodes_remaining:
            new_value, _ = minimax(graph_laplacian, np.append(config, node), nodes_per_team, ksym, 1)
            # if the value is updated, update the best move
            if value < new_value:
                value = new_value
                best_node = node
        return value, best_node

    if turn == 1:  # Minimizing team
        value = np.inf
        best_node = None
        for node in nodes_remaining:
            new_value, _ = minimax(graph_laplacian, np.append(config, node), nodes_per_team, ksym, 0)
            # if the value is updated, update the best move
            if value > new_value:
                value = new_value
                best_node = node
        return value, best_node


# Picking up floating point errors which alters what the best value is.
#
# config = np.array([])
# for i in range(2):
#     _, next = minimax(alt.ex7, config, 2, 2, 0)
#     config = np.append(config, next)
#     config, _ = opponent_smart_step(alt.ex7, config, 2, 2, *alt.get_all_configs(alt.ex7, 2, 2))
#     print(_, config)
#
# config = np.array([])
# for i in range(2):
#     _, next = minimax(alt.ex7, config, 2, 2, 0)
#     config = np.append(config, next)
#     _, next_2 = minimax(alt.ex7, config, 2, 2, 1)
#     config = np.append(config, next_2)
#     print(_, config)
#
# config = np.array([])
# for i in range(2):
#     config, _ = opponent_smart_step(alt.ex7, config, 2, 2, *alt.get_all_configs(alt.ex7, 2, 2))
#     config, _ = opponent_smart_step(alt.ex7, config, 2, 2, *alt.get_all_configs(alt.ex7, 2, 2))
#     print(_, config)











#################### TESTS ####################

# ex_graph = alt.ex7
# all_influences, permutations = alt.get_all_configs(ex_graph, 2, 2)
#
# config = opponent_smart_step(ex_graph, np.array([]), 2, 2, all_influences, permutations)[0]
# config = opponent_smart_step(ex_graph, config, 2, 2, all_influences, permutations)[0]
# config = opponent_smart_step(ex_graph, config, 2, 2, all_influences, permutations)[0]
# print(opponent_smart_step(ex_graph, config, 2, 2, all_influences, permutations))




# ex_graph = alt.random_graph(20)

# ex7 = np.array([[1, -1, 0, 0, 0, 0, 0],
#                 [-1, 2, -1, 0, 0, 0, 0],
#                 [0, -1, 2, -1, 0, 0, 0],
#                 [0, 0, -1, 2, -1, 0, 0],
#                 [0, 0, 0, -1, 2, -1, 0],
#                 [0, 0, 0, 0, -1, 2, -1],
#                 [0, 0, 0, 0, 0, -1, 1]])
#
# ex_graph = ex7

### Mickey
# ear1 = alt.random_graph(5)
# head = alt.random_graph(10)
# ear2 = alt.random_graph(5)
#
# filler_top = np.zeros((ear1.shape[0], head.shape[0]))
# filler_side = np.zeros((head.shape[0], ear2.shape[0]))
# filler_corner = np.zeros((ear1.shape[0], ear2.shape[0]))
#
# top = np.hstack((ear1, filler_top, filler_corner))
# mid = np.hstack((filler_side, head, filler_side))
# bot = np.hstack((filler_corner, filler_top, ear2))
#
# ex_graph = np.vstack((top, mid, bot)).astype(int)
#
# ex_graph[4, 5] = -1
# ex_graph[5, 4] = -1
# ex_graph[4, 4] += 1
# ex_graph[5, 5] += 1
# ex_graph[14, 15] = -1
# ex_graph[15, 14] = -1
# ex_graph[14, 14] += 1
# ex_graph[15, 15] += 1




# print(ex_graph)
# # visualize ex_graph using networkx
# vis_ex_graph = ex_graph.copy()
# # set diagonal to 0
# np.fill_diagonal(vis_ex_graph, 0)
# # make it a circle graph
# pos = nx.circular_layout(nx.Graph(vis_ex_graph))
# # draw the graph
# nx.draw(nx.Graph(vis_ex_graph), pos, with_labels=True)
# plt.show()

# nodes_per_team = 3

# result, length = opponent_least_best(ex_graph, nodes_per_team)
# print(result, length)

# bd_matrix_start = np.zeros((ex_graph.shape[0], 2))
# bd_matrix_start[0, 0] = 1
# bd_matrix_start[4, 1] = 1
# bd_matrix_start[2, 0] = 1

# print(opponent_greedy_2(ex_graph, bd_matrix_start, nodes_per_team, ksym=2))
#
# print('smart')
# print(opponent_smart(ex_graph, 2, 2))
# # Greedy boi
# ksym = 2
# nodes_per_team = 2
# print('dumb')
# conf, influ, all_influ = alt.opponent_dumb_2(ex_graph, nodes_per_team, ksym)
# print(conf, influ)
# bd_matrix = np.zeros((ex_graph.shape[0], ksym))
# nodes = np.arange(ex_graph.shape[0])
# config = np.array([])
# c = 0
# do_it = True
# print('greedy')
# if do_it is True:
#     while c < nodes_per_team:
#         selected_node = int(input(f"Select a node to add to the boundary set: {np.setdiff1d(nodes, config)}"))
#         config = np.append(config, selected_node)
#
#         bd_matrix[selected_node, 0] = 1
#
#         bd_matrix, max_node = alt.alternating_step(ex_graph, bd_matrix, 1)[0:2]
#
#         config = np.append(config, max_node)
#
#         c += 1
#
#     print(config, all_influ[tuple(config)])


# print(ex_graph)


# ex_graph = alt.random_graph(12)
# print(opponent_smart(ex_graph, 2, 2, player=False))



