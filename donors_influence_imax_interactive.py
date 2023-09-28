

import numpy as np
import influence_maximization_algorithms as im


# create a game system that askes for input
def game_setup(nodes_question=False):
    # Select opponent difficulty
    # Choose number of nodes
    # choose to go first, second, or random

    # Game setup
    while True:
        difficulty = str(input("Select opponent difficulty (easy, medium, hard): "))
        if difficulty == "easy" or difficulty == "medium" or difficulty == "hard" or difficulty == "random":
            break
        else:
            print("Please enter a valid difficulty.")
    while True:
        num_nodes = float(input("Choose number of nodes (8-20): "))
        if num_nodes >= 8 and num_nodes <= 20 and num_nodes == int(num_nodes) or num_nodes == -1:
            num_nodes = int(num_nodes)
            break
        else:
            print("Please enter a valid number of nodes.")
    while True:
        turn_choice = input("Choose who starts (first, second, random): ")
        if turn_choice == "first" or turn_choice == "second" or turn_choice == "random":
            break
        else:
            print("Please enter a valid choice.")
    if nodes_question:
        while True:
            nodes_per_team = float(input("Choose number of nodes per team (2-4): "))
            if nodes_per_team >= 2 and nodes_per_team <= 4 and nodes_per_team == int(nodes_per_team):
                nodes_per_team = int(nodes_per_team)
                break
            else:
                print("Please enter a valid number of nodes per team.")
        return [difficulty, num_nodes, turn_choice, nodes_per_team]

    return [difficulty, num_nodes, turn_choice]


def gameplay(difficulty, num_nodes, turn_choice, nodes_per_team=2):

    # If num_nodes == -1 make it random
    if num_nodes == -1:
        num_nodes = np.random.randint(8, 21)

    # Create the graph
    graph = im.random_graph(num_nodes)

    # initialize turn counter, config, and array of remaining nodes
    c = 0
    nodes = np.arange(num_nodes)
    config = np.array([])

    # Convert difficulty
    if difficulty == "random":
        difficulty = np.random.choice(["easy", "medium", "hard"])
    if difficulty == "easy":
        opponent = im.rev_minimax_algorithm_opt
    elif difficulty == "medium":
        opponent = im.greedy_algorithm
    else:
        opponent = im.minimax_algorithm_opt

    # Convert turn choice to a number
    if turn_choice == "random":
        player = np.random.choice([1, 2])
    elif turn_choice == "first":
        player = 1
    else:  # turn_choice == "second"
        player = 2

    # Say what player you are
    print(f"You are player {player}.")
    player = player % 2
    player_num = player

    while c < 2 * nodes_per_team:  # while there are turns left

        # HERE IS WHERE YOU WOULD UPDATE A RENDERING

        if player:  # if the player is selecting the nodes
            while True:
                selected_node = float(input(f"Select a node ({np.setdiff1d(nodes, config)}): "))
                if selected_node in np.setdiff1d(nodes, config):
                    selected_node = int(selected_node)
                    break
                else:
                    print("Please enter a valid node.")
        else:  # opponent is selecting the nodes
            selected_node = opponent(graph, config, nodes_per_team)
            print(f"Your opponent selected node {selected_node}")

        # Add the node to the config
        config = np.append(config, selected_node)

        # Update the player
        player = (player + 1) % 2

        # Update turn number
        c += 1

    return im.get_influence(graph, config), player_num


def main():
    # Game setup
    game_setup_list = game_setup(nodes_question=True)

    # Play the game
    influence, player_number = gameplay(*game_setup_list)

    # Print the influence
    if player_number:  # if you are player 1
        print(f"Your Score: {influence[0]}\n"
              f"Opponent's Score: {influence[1]}")
        if abs(influence[0] - influence[1]) < .0001:
            print("You tied")
        elif influence[0] > influence[1]:
            print("Congratulations, you won!")
        else:
            print("Better luck next time :(")
    else:  # if you are player 2
        print(f"Your Score: {influence[1]}\n"
              f"Opponent's Score: {influence[0]}")
        if abs(influence[0] - influence[1]) < .0001:
            print("You tied")
        elif influence[0] < influence[1]:
            print("Congratulations, you won!")
        else:
            print("Better luck next time :(")

main()