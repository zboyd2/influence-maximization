import numpy as np


def get_influence(graph_laplacian, config):
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
    config = config.astype(int)
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

    # Calculate influence of team 0 using same approach as above
    team_0_nodes = config[0::2]
    team_0_bd = np.zeros(graph_laplacian.shape[0]).astype(int)
    team_0_bd[team_0_nodes] = 1
    influence_0 = np.linalg.lstsq(lsc, -b @ team_0_bd[config], rcond=None)[0].sum()
    influence_0 += team_0_bd.sum()

    '''
    More efficient way to calculate influece_0, but it only works if network is connected
    
    # Calculate the influence of the other team
    influence_0 = graph_laplacian.shape[0] - influence_1
    '''

    return np.array((influence_0, influence_1))


def greedy_algorithm(graph_laplacian, config, nodes_per_team):
    """This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the greedy algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

        nodes_per_team (int): The number of nodes per team

    Returns:
        best_node (int): The best node to add to the boundary set
    """

    if type(config) is not np.ndarray:
        config = np.array(config)

    turn = config.shape[0] % 2
    nodes = np.arange(graph_laplacian.shape[0])
    nodes_remaining = np.setdiff1d(nodes, config)

    best_node = None
    best_value = -np.inf

    if config.shape[0] == 0:  # First move
        return np.argmax(np.diag(graph_laplacian))
    elif config.shape[0] >= nodes_per_team * 2:  # Game is over
        return None
    else:  # Not first move
        for n in nodes_remaining:  # Check each node to see which one is best
            new_val = get_influence(graph_laplacian, np.append(config, n).astype(int))[turn]
            if new_val > best_value:
                best_value = new_val
                best_node = n
    return best_node


def minimax_algorithm(graph_laplacian, config, nodes_per_team):
    """This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the minimax algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

        nodes_per_team (int): The number of nodes per team

    Returns:
        best_node (int): The best node to add to the boundary set
    """


    def minimax_recursive_step(graph_laplacian, config, nodes_per_team):

        if type(config) is not np.ndarray:
            config = np.array(config)

        turn = config.shape[0] % 2  # 0 for maximizing team, 1 for minimizing team

        if len(config) == nodes_per_team * 2:
            return get_influence(graph_laplacian, config.astype(int))[0], None

        nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

        if turn == 0:  # Maximizing team
            value = -np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(graph_laplacian, np.append(config, node), nodes_per_team)
                # if the value is updated, update the best move
                if value < new_value:
                    value = new_value
                    best_node = node
            return value, best_node

        if turn == 1:  # Minimizing team
            value = np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(graph_laplacian, np.append(config, node), nodes_per_team)
                # if the value is updated, update the best move
                if value > new_value:
                    value = new_value
                    best_node = node
            return value, best_node

    return minimax_recursive_step(graph_laplacian, config, nodes_per_team)[1]


# Optimized Using Alpha-Beta Pruning
def minimax_algorithm_opt(graph_laplacian, config, nodes_per_team, depth=np.inf):
    """ Optimized using alpha-beta pruning.

    This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the minimax algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

        nodes_per_team (int): The number of nodes per team

    Returns:
        best_node (int): The best node to add to the boundary set
    """


    def minimax_recursive_step(config, depth, alpha, beta):

        if type(config) is not np.ndarray:
            config = np.array(config)

        turn = config.shape[0] % 2  # 0 for maximizing team, 1 for minimizing team

        if depth == 0 or config.shape[0] == nodes_per_team * 2:
            return get_influence(graph_laplacian, config.astype(int))[0], None

        nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

        if turn == 0:  # Maximizing team
            value = -np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value < new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value > beta:
                    break
                alpha = max(alpha, value)
            return value, best_node

        if turn == 1:  # Minimizing team
            value = np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value > new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value < alpha:
                    break
                beta = min(beta, value)
            return value, best_node

    return minimax_recursive_step(config, depth, -np.inf, np.inf)[1]


# Reverse minimax to make easy opponent
def rev_minimax_algorithm_opt(graph_laplacian, config, nodes_per_team, depth=np.inf):
    """ Optimized using alpha-beta pruning.

    This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the minimax algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

        nodes_per_team (int): The number of nodes per team

    Returns:
        best_node (int): The best node to add to the boundary set
    """


    def minimax_recursive_step(config, depth, alpha, beta):

        if type(config) is not np.ndarray:
            config = np.array(config)

        turn = config.shape[0] % 2  # 0 for maximizing team, 1 for minimizing team

        if depth == 0 or config.shape[0] == nodes_per_team * 2:
            return get_influence(graph_laplacian, config.astype(int))[0], None

        nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

        if turn == 0:  # Maximizing team
            value = np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value > new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value > beta:
                    break
                alpha = max(alpha, value)
            return value, best_node

        if turn == 1:  # Minimizing team
            value = -np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value < new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value < alpha:
                    break
                beta = min(beta, value)
            return value, best_node

    return minimax_recursive_step(config, depth, -np.inf, np.inf)[1]


# Optimized against greedy
def minimax_algorithm_vs_greedy(graph_laplacian, config, nodes_per_team, mover="first"):
    """This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the minimax algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

        nodes_per_team (int): The number of nodes per team

    Returns:
        best_node (int): The best node to add to the boundary set
    """


    def minimax_recursive_step(graph_laplacian, config, nodes_per_team):
        if type(config) is not np.ndarray:
            config = np.array(config)

        turn = config.shape[0] % 2  # 0 for maximizing team, 1 for minimizing team

        if len(config) == nodes_per_team * 2:
            return get_influence(graph_laplacian, config.astype(int))[0], None

        nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

        if turn == 0:  # Maximizing team
            value = -np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(graph_laplacian, np.append(config, node), nodes_per_team)
                # if the value is updated, update the best move
                if value < new_value:
                    value = new_value
                    best_node = node
            return value, best_node

        if turn == 1:  # Minimizing team
            value = np.inf
            best_node = None

            best_node = greedy_algorithm(graph_laplacian, config, nodes_per_team)
            value = get_influence(graph_laplacian, np.append(config, best_node).astype(int))[0]

            return value, best_node

    return minimax_recursive_step(graph_laplacian, config, nodes_per_team)[1]


def component_number(m, tol=1e-08):
    """Find the number of connected components in a matrix.
    Used in random_graph to ensure that the graph is connected.
    """
    eigvals = np.linalg.eig(m)[0]
    return m.shape[0] - np.sum(np.abs(eigvals) > tol)


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
    if (np.diag(graph_laplacian) == 0).sum() > 0:
        return random_graph(n)
    elif component_number(graph_laplacian) > 1:
        return random_graph(n)

    return graph_laplacian


def random_graph_generator(num_nodes, num_graphs=1000):
    """Generates a set of random Graph Laplacians. When more than one graph is generated, the graphs are stacked along
    the third dimension.

    Args:
        num_nodes (int): The number of nodes in the graph Laplacian

        num_graphs (int): The number of graph Laplacians to generate

    Returns:
        all_graphs ((num_nodes, num_nodes, num_graphs) ndarray): The set of graph Laplacians
    """

    all_graphs = random_graph(num_nodes)

    for i in range(num_graphs - 1):
        r_graph = random_graph(num_nodes)
        all_graphs = np.dstack((all_graphs, r_graph))

    return all_graphs


# Matchups #############################################################################################################


def matchup_greedy_greedy(graph_laplacian):
    """This function takes in a graph Laplacian and returns the
    influence of the greedy algorithm vs the greedy algorithm.
    """

    config = np.array([])

    for i in range(2):
        config = np.append(config, greedy_algorithm(graph_laplacian, config, 2))

        config = np.append(config, greedy_algorithm(graph_laplacian, config, 2))

    return get_influence(graph_laplacian, config.astype(int))[0]


def matchup_greedy_minimax(graph_laplacian):
    """This function takes in a graph Laplacian and returns the
    influence of the greedy algorithm vs the minimax algorithm.
    """

    config = np.array([])

    for i in range(2):
        config = np.append(config, greedy_algorithm(graph_laplacian, config, 2))

        config = np.append(config, minimax_algorithm_opt(graph_laplacian, config, 2))

    return get_influence(graph_laplacian, config.astype(int))[0]


def matchup_minimax_greedy(graph_laplacian):
    """This function takes in a graph Laplacian and returns the
    influence of the minimax algorithm vs the greedy algorithm.
    """

    config = np.array([])

    for i in range(2):
        config = np.append(config, minimax_algorithm_opt(graph_laplacian, config, 2))

        config = np.append(config, greedy_algorithm(graph_laplacian, config, 2))

    return get_influence(graph_laplacian, config.astype(int))[0]


def matchup_minimax_minimax(graph_laplacian):
    """This function takes in a graph Laplacian and returns the
    influence of the minimax algorithm vs the minimax algorithm."""

    config = np.array([])

    for i in range(2):
        config = np.append(config, minimax_algorithm_opt(graph_laplacian, config, 2))

        config = np.append(config, minimax_algorithm_opt(graph_laplacian, config, 2))

    return get_influence(graph_laplacian, config.astype(int))[0]


# Testing ##############################################################################################################


def generate_matchup_influences(num_nodes=10, num_graphs=1000):
    """This function generates a set of random graph Laplacians and then
    runs the four matchup functions on each graph. It returns the resulting
    influence values for each matchup.

    Args:
        num_nodes (int): The number of nodes in the graph Laplacian

        num_graphs (int): The number of graph Laplacians to generate

    Returns:
        greedy_greedy ((num_graphs,) ndarray): The influence of the greedy algorithm vs the greedy algorithm for each
        graph Laplacian in the set

        greedy_minimax ((num_graphs,) ndarray): The influence of the greedy algorithm vs the minimax algorithm for
        each graph Laplacian in the set

        minimax_greedy ((num_graphs,) ndarray): The influence of the minimax algorithm vs the greedy algorithm for
        each graph Laplacian in the set

        minimax_minimax ((num_graphs,) ndarray): The influence of the minimax algorithm vs the minimax algorithm for
        each graph Laplacian in the set
    """

    all_graphs = random_graph_generator(num_nodes, num_graphs)

    greedy_greedy = np.array([])
    greedy_minimax = np.array([])
    minimax_greedy = np.array([])
    minimax_minimax = np.array([])

    # Loop through each matchup and append the resulting influence for team 1
    for i in range(num_graphs):

        greedy_greedy = np.append(greedy_greedy, matchup_greedy_greedy(all_graphs[:, :, i]))
        greedy_minimax = np.append(greedy_minimax, matchup_greedy_minimax(all_graphs[:, :, i]))
        minimax_greedy = np.append(minimax_greedy, matchup_minimax_greedy(all_graphs[:, :, i]))
        minimax_minimax = np.append(minimax_minimax, matchup_minimax_minimax(all_graphs[:, :, i]))

    return greedy_greedy, greedy_minimax, minimax_greedy, minimax_minimax


# Algorithms for arbitrary nodes_per_team ##############################################################################


def gui_easy_opponent(graph_laplacian, config):
    if graph_laplacian.size == 0:
        return 0
    return np.random.choice(np.setdiff1d(np.arange(graph_laplacian.shape[0]), config))


def gui_greedy_algorithm(graph_laplacian, config):
    """This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the greedy algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

    Returns:
        best_node (int): The best node to add to the boundary set
    """

    if type(config) is not np.ndarray:
        config = np.array(config)

    turn = config.shape[0] % 2
    nodes = np.arange(graph_laplacian.shape[0])
    nodes_remaining = np.setdiff1d(nodes, config)

    best_node = None
    best_value = -np.inf

    if config.shape[0] == 0:  # First move
        try:
            return np.argmax(np.diag(graph_laplacian))
        except ValueError:
            pass

    else:  # Not first move
        for n in nodes_remaining:  # Check each node to see which one is best
            new_val = get_influence(graph_laplacian, np.append(config, n).astype(int))[turn]
            if new_val > best_value:
                best_value = new_val
                best_node = n

    return best_node


def gui_minimax_algorithm_opt(graph_laplacian, config, num_turns, depth=3):
    """ Optimized using alpha-beta pruning.

    This function takes in the graph Laplacian, the previously
    selected boundary nodes as an iterable, and the number of nodes
    per team. It returns the best node to add to the boundary set
    using the minimax algorithm.

    Args:
        graph_laplacian ((n,n) ndarray): The graph Laplacian for
        the given graph

        config (ndarray): The previously selected boundary nodes
        in sequential order alternating team

    Returns:
        best_node (int): The best node to add to the boundary set
    """


    def minimax_recursive_step(config, depth, alpha, beta):

        if type(config) is not np.ndarray:
            config = np.array(config)

        turn = config.shape[0] % 2  # 0 for maximizing team, 1 for minimizing team

        if depth == 0 or config.shape[0] >= num_turns * 2:
            return get_influence(graph_laplacian, config.astype(int))[0], None

        nodes_remaining = np.setdiff1d(np.arange(graph_laplacian.shape[0]), config)

        if turn == 0:  # Maximizing team
            value = -np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value < new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value > beta:
                    break
                alpha = max(alpha, value)
            return value, best_node

        if turn == 1:  # Minimizing team
            value = np.inf
            best_node = None
            for node in nodes_remaining:
                new_value, _ = minimax_recursive_step(np.append(config, node), depth - 1, alpha, beta)
                # if the value is updated, update the best move
                if value > new_value:
                    value = new_value
                    best_node = node
                # Alpha-Beta Pruning
                if value < alpha:
                    break
                beta = min(beta, value)
            return value, best_node

    return minimax_recursive_step(config, depth, -np.inf, np.inf)[1]
