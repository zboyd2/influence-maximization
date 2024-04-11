import time

import pygame
import random
import numpy as np
import scipy.stats as ss
import networkx as nx
import influence_maximization_algorithms as im

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (800, 600)
NUM_NODES = 50
NODE_RADIUS = 15
CONTROL_MARK_RADIUS = 5
LINE_THICKNESS = 2
FONT_SIZE = 24

NAVY = (0, 46, 93)
WHITE = (255, 255, 255)
ROYAL = (0, 61, 165)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)

NODE_COLORS = [RED, BLUE]
CONTROL_MARK_COLORS = [WHITE, BLACK]

# Initialize screen and clock
screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("DISE: Dynamic Influence Spread Estimator")
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)
highlighted_node = None

corner_radius = 10
def draw_rounded_rect(screen, color, rect, corner_radius):
    """Draw a rounded rectangle"""
    x, y, width, height = rect
    
    # Draw the main body of the rectangle
    pygame.draw.rect(screen, color, (x, y + corner_radius, width, height - 2*corner_radius))
    pygame.draw.rect(screen, color, (x + corner_radius, y, width - 2*corner_radius, height))
    
    # Draw the four rounded corners
    pygame.draw.circle(screen, color, (x + corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + width - corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + corner_radius, y + height - corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + width - corner_radius, y + height - corner_radius), corner_radius)


def initialize_graph(approach=6):
    global NUM_NODES 
    # Scaled placement margins
    x_margin = int(SCREEN_SIZE[0] * 0.10)
    y_margin = int(SCREEN_SIZE[1] * 0.12)

    if approach <= 1: #Positions nodes randomly and draws edges according to an approach below
        nodes = []
        screen_diagonal = np.linalg.norm(np.array(SCREEN_SIZE))
        edge_threshold = screen_diagonal * 0.095  #Set edge threshold as a fraction of the diagonal

        while len(nodes) < NUM_NODES:
            new_node = (random.randint(x_margin, SCREEN_SIZE[0] - x_margin), random.randint(y_margin, SCREEN_SIZE[1] - y_margin))
            if all(np.linalg.norm(np.array(new_node) - np.array(existing_node)) > 2 * NODE_RADIUS for existing_node in nodes): 
                nodes.append(new_node)

        if(approach == 0): #Draw edge between nodes if they are closer than the threshold distance
            edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) < edge_threshold]
        else: #Draw edges according to a gaussian distribution based on the distance between nodes
            edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if 2 * (1 - ss.norm.cdf((np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) / (NODE_RADIUS * 2)), loc=0, scale=3)) > random.random()]

    else: #Generate graph based on a predefined NetworkX graph
        if approach == 2:
            G = nx.full_rary_tree(4, NUM_NODES)
            positions = nx.kamada_kawai_layout(G)
        elif approach == 3:
            G = nx.ladder_graph(int(NUM_NODES / 2))
            positions = nx.spring_layout(G)
        elif approach == 4: #square lattice 
            #Pick a length for a length x length square lattice with approximately NUM_NODES nodes
            length = int(np.floor(np.sqrt(NUM_NODES)))
            NUM_NODES = int(length ** 2)
            H = nx.grid_graph(dim=(length, length))
            node_cords = list(H.nodes)
            adjusted_cords = np.interp(node_cords, [0, length], [-1, 1])
            positions = {i : adjusted_cords[i] for i in range(NUM_NODES)}
            G = nx.relabel_nodes(H, {node_cords[i] : i  for i in range(NUM_NODES)})
        elif approach == 5: #hexagon lattice
            #Pick an n and m that will create an n x m hexagon lattice with approximately NUM_NODES nodes
            n = int(np.rint(np.sqrt(NUM_NODES)) / 2)
            m = int(np.rint((NUM_NODES - 2 * n) / (2 * n + 2)))
            NUM_NODES = 2 * (n + n * m + m)

            H = nx.hexagonal_lattice_graph(n, m)
            node_names = [*nx.get_node_attributes(H, 'pos')]
            node_cords = [*nx.get_node_attributes(H, 'pos').values()]
            x_cords = [x for x, _ in node_cords]
            y_cords = [y for _, y in node_cords]
            adjusted_x_cords = np.interp(x_cords, [0, np.max(x_cords)], [-1, 1])
            adjusted_y_cords = np.interp(y_cords, [0, np.max(y_cords)], [-1, 1])
            positions = {i : np.array([adjusted_x_cords[i], adjusted_y_cords[i]]) for i in range(NUM_NODES)}
            G = nx.relabel_nodes(H, {node_names[i] : i  for i in range(NUM_NODES)})
        elif approach == 6: #triangle lattice
            #Pick an n and m that will create an n x m triangular lattice with approximately NUM_NODES nodes
            n = int(np.sqrt(NUM_NODES))
            if n % 2 == 0:
                n += 1
            m = int(np.rint((2 * NUM_NODES - 2 * n - 2) / (n + 1)))
            NUM_NODES = int(0.5 * n * m + 0.5 * m + n + 1)

            H = nx.triangular_lattice_graph(n, m)
            node_names = [*nx.get_node_attributes(H, 'pos')]
            node_cords = [*nx.get_node_attributes(H, 'pos').values()]
            x_cords = [x for x, _ in node_cords]
            y_cords = [y for _, y in node_cords]
            adjusted_x_cords = np.interp(x_cords, [0, np.max(x_cords)], [-1, 1])
            adjusted_y_cords = np.interp(y_cords, [0, np.max(y_cords)], [-1, 1])
            positions = {i : np.array([adjusted_x_cords[i], adjusted_y_cords[i]]) for i in range(NUM_NODES)}
            G = nx.relabel_nodes(H, {node_names[i] : i  for i in range(NUM_NODES)})
        else:
            G = nx.cycle_graph(NUM_NODES)
            positions = nx.spring_layout(G)
        
        #Get edges and node coordinates based on the screen sizing
        nodes = []
        edges = G.edges
        for key in positions:
            #Scale node positions to fit in game screen
            x_cord = x_margin + (SCREEN_SIZE[0] - 2 * x_margin) * ((positions[key][0] + 1) / 2)
            y_cord = y_margin + (SCREEN_SIZE[1] - 2 * y_margin) * ((positions[key][1] + 1) / 2)
            nodes.append((x_cord, y_cord))
    
    # Create adjacency list
    adj_list = [[] for _ in range(NUM_NODES)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return nodes, edges, adj_list


def get_graph_laplacian(edges):
    # Create adjacency matrix
    adj_mat = np.zeros((NUM_NODES, NUM_NODES))
    for i, j in edges:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # Get degree matrix
    deg_mat = np.diag(np.sum(adj_mat, axis=1))

    return deg_mat - adj_mat  # Laplacian matrix


nodes, edges, adj_list = initialize_graph()

# Get Laplacian matrix
laplacian = get_graph_laplacian(edges)

opinions = [0 for _ in range(NUM_NODES)]
controls = [None for _ in range(NUM_NODES)]

# Initialize game state
current_player = 0
turn_count = 0
resize_cooldown = 0  # To control the cooldown after a resize event

# For buffering VIDEORESIZE events
latest_resize_event = None

p1_bot_index = 0
p2_bot_index = 0
bot_names = ["Human", "Easy", "Medium", "Hard"]
bot_colors = [(200, 200, 200), (220, 220, 220), (240, 240, 240), (250, 250, 250)]

# Opponent Variable initialization
human_went = False
nodes_per_team = 10
depth = 3
config = np.array([])
start_time_p1 = time.perf_counter()
start_time_p2 = time.perf_counter()
wait_time = 0.75
NUM_TURNS = 6
first_move = True

def draw_bot_choice_button(screen, font, current_player):
    button_width, button_height = 150, 40
    button_x = SCREEN_SIZE[0] - (button_width + 10) * (3-current_player)
    button_y = 40

    # Label for players
    if current_player == 1:
        player_label = font.render("Player 1", True, (0,0,0))
        bot_index = p1_bot_index
    else:
        player_label = font.render("Player 2", True, (0,0,0))
        bot_index = p2_bot_index

    # Display the label
    screen.blit(player_label, (button_x, button_y - 30))  # Adjust the y-value to position the label above the button

    # Draw the button
    draw_rounded_rect(screen, bot_colors[bot_index], (button_x, button_y, button_width, button_height), corner_radius)

    label = font.render(bot_names[bot_index], True, (0, 0, 0))
    screen.blit(label, (button_x + 30, button_y + 10))
    
    # Return the button's rectangle for click detection
    return (button_x, button_y, button_width, button_height)

def draw_ask_coach(screen, font):
    button_width, button_height = 150, 40
    button_x = 10
    button_y = 50

    # Draw the button
    draw_rounded_rect(screen, (200, 200, 200), (button_x, button_y, button_width, button_height), corner_radius)
    label = font.render("Ask coach?", True, (0, 0, 0))
    screen.blit(label, (button_x + 30, button_y + 10))
    
    # Return the button's rectangle for click detection
    return (button_x, button_y, button_width, button_height)


# Game loop
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for button clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # Check for clicks on the Reset button
            if reset_button_x <= x <= reset_button_x + reset_button_w and reset_button_y <= y <= reset_button_y + reset_button_h:
                current_player = 0
                turn_count = 0
                controls = [None for _ in range(NUM_NODES)]
                opinions = [0 for _ in range(NUM_NODES)]  # Reset opinions to neutral
                config = np.array([])
                first_move = True
                p1_bot_index = 0
                p2_bot_index = 0

                nodes, edges, adj_list = initialize_graph()

                # Get Laplacian matrix
                laplacian = get_graph_laplacian(edges)
                continue  # Skip the rest of this loop iteration
            # Check for changes in bot type
            elif bot_x1 <= event.pos[0] <= bot_x1 + bot_width1 and bot_y1 <= event.pos[1] <= bot_y1 + bot_height1:
                p1_bot_index = (p1_bot_index + 1) % len(bot_names)
                click_time = time.perf_counter()
            elif bot_x2 <= event.pos[0] <= bot_x2 + bot_width2 and bot_y2 <= event.pos[1] <= bot_y2 + bot_height2:
                p2_bot_index = (p2_bot_index + 1) % len(bot_names)

            # Check for clicks on the Ask Coach button
            elif coach_x <= event.pos[0] <= coach_x + coach_width and coach_y <= event.pos[
                1] <= coach_y + coach_height:
                # highlight a random node that is not controlled by the player
                highlighted_node = im.gui_minimax_algorithm_opt(laplacian, config, depth)

        elif event.type == pygame.VIDEORESIZE:
            latest_resize_event = event

        if turn_count < NUM_TURNS * 2:  # While turns remain
            # Player 1 or Player 2
            if 1 - current_player:  # Player 1
                if not p1_bot_index:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        # Check for clicks on nodes
                        for i, (nx, ny) in enumerate(nodes):
                            if np.linalg.norm(np.array([x, y]) - np.array([nx, ny])) < NODE_RADIUS:
                                if controls[i] is None:
                                    controls[i] = current_player
                                    current_player = 1 - current_player
                                    turn_count += 1
                                    config = np.append(config, i)
                                highlighted_node = None
                elif time.perf_counter() - start_time_p2 > wait_time:
                    if p1_bot_index == 1:
                        opponent = im.gui_easy_opponent
                    elif p1_bot_index == 2:
                        opponent = im.gui_greedy_algorithm
                    else:
                        opponent = im.gui_minimax_algorithm_opt

                    if first_move:
                        if time.perf_counter() - click_time > wait_time:
                            first_move = False
                        continue

                    bot_move = opponent(laplacian, config, depth)
                    config = np.append(config, bot_move)
                    controls[bot_move] = current_player
                    current_player = 1 - current_player
                    turn_count += 1

                start_time_p1 = time.perf_counter()  # Reset the timer

            else:  # Player 2
                if not p2_bot_index:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos

                        # Check for clicks on nodes
                        for i, (nx, ny) in enumerate(nodes):
                            if np.linalg.norm(np.array([x, y]) - np.array([nx, ny])) < NODE_RADIUS:
                                if controls[i] is None:
                                    controls[i] = current_player
                                    current_player = 1 - current_player
                                    turn_count += 1
                                    config = np.append(config, i)
                                highlighted_node = None
                elif time.perf_counter() - start_time_p1 > wait_time:
                    if p2_bot_index == 1:
                        opponent = im.gui_easy_opponent
                    elif p2_bot_index == 2:
                        opponent = im.gui_greedy_algorithm
                    else:
                        opponent = im.gui_minimax_algorithm_opt

                    bot_move = opponent(laplacian, config, depth)
                    config = np.append(config, bot_move)
                    controls[bot_move] = current_player
                    current_player = 1 - current_player
                    turn_count += 1

                start_time_p2 = time.perf_counter()  # Reset the timer

        else:
            # TODO Display the winner
            pass



        # elif event.type == pygame.MOUSEBUTTONDOWN or human_went:
        #     if event.type == pygame.MOUSEBUTTONDOWN:
        #         x, y = event.pos
        #         # Check for clicks on the Reset button
        #         if reset_button_x <= x <= reset_button_x + reset_button_w and reset_button_y <= y <= reset_button_y + reset_button_h:
        #             current_player = 0
        #             turn_count = 0
        #             controls = [None for _ in range(NUM_NODES)]
        #             opinions = [0 for _ in range(NUM_NODES)]  # Reset opinions to neutral
        #             continue  # Skip the rest of this loop iteration
        #
        #     # if cool_down < 2 and p1 went
        #     if human_went:
        #         if time.perf_counter() - timer > 1:
        #             # TODO Add Opponent generalization
        #             bot_move = im.minimax_algorithm_opt(laplacian, config, nodes_per_team, depth)
        #             config = np.append(config, bot_move)
        #             controls[bot_move] = current_player
        #             current_player = 1 - current_player
        #             turn_count += 1
        #             human_went = False
        #     else:
        #         # Check for clicks on nodes
        #         for i, (nx, ny) in enumerate(nodes):
        #             if np.linalg.norm(np.array([x, y]) - np.array([nx, ny])) < NODE_RADIUS:
        #                 if controls[i] is None:
        #                     controls[i] = current_player
        #                     config = np.append(config, i)
        #                     current_player = 1 - current_player
        #                     turn_count += 1
        #                     if p2_bot_index != 0:
        #                         human_went = True
        #                         timer = time.perf_counter()
        #
        #
        #
        #     # Check for changes in bot type
        #     if bot_x1 <= event.pos[0] <= bot_x1 + bot_width1 and bot_y1 <= event.pos[1] <= bot_y1 + bot_height1:
        #             p1_bot_index = (p1_bot_index + 1) % len(bot_names)
        #     if bot_x2 <= event.pos[0] <= bot_x2 + bot_width2 and bot_y2 <= event.pos[1] <= bot_y2 + bot_height2:
        #             p2_bot_index = (p2_bot_index + 1) % len(bot_names)

    if latest_resize_event and pygame.time.get_ticks() - resize_cooldown > 500: # 500ms cool-down
        # Get new dimensions
        new_w, new_h = latest_resize_event.w, latest_resize_event.h
        
        # Calculate scaling factors
        x_scale = new_w / SCREEN_SIZE[0]
        y_scale = new_h / SCREEN_SIZE[1]
        
        # Update screen size constants
        SCREEN_SIZE = (new_w, new_h)
        
        # Scale node positions and other elements
        nodes = [(int(x * x_scale), int(y * y_scale)) for (x, y) in nodes]
        NODE_RADIUS = int(NODE_RADIUS * (x_scale + y_scale) / 2)  # average scaling factor for radius
        
        # Resize screen
        screen = pygame.display.set_mode((latest_resize_event.w, latest_resize_event.h), pygame.RESIZABLE)

        # Reset the cooldown timer
        resize_cooldown = pygame.time.get_ticks()
        latest_resize_event = None

        # Explicitly fill screen with a background color
        screen.fill(WHITE)
        pygame.display.flip()

    # Update Opinions
    new_opinions = []
    for i in range(NUM_NODES):
        neighbor_opinions = [opinions[j] for j in adj_list[i]]
        if controls[i] is not None:
            new_opinions.append(controls[i] * 2 - 1)
        elif neighbor_opinions:
            new_opinions.append(sum(neighbor_opinions) / len(neighbor_opinions))
        else:
            new_opinions.append(opinions[i])
    opinions = new_opinions

    # Draw UI
    screen.fill(WHITE)

    # Draw edges
    for edge in edges:
        pygame.draw.line(screen, (128, 128, 128), nodes[edge[0]], nodes[edge[1]], LINE_THICKNESS)

    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        color = tuple(int((NODE_COLORS[0][j] * (1 - opinions[i]) + NODE_COLORS[1][j] * (1 + opinions[i])) / 2) for j in range(3))
        pygame.draw.circle(screen, color, (x, y), NODE_RADIUS)
        if controls[i] is not None:
            pygame.draw.circle(screen, CONTROL_MARK_COLORS[controls[i]], (x, y), CONTROL_MARK_RADIUS)

    # Drawing the Reset button
    reset_button_color = (200, 200, 200)  # Gray color
    reset_button_x, reset_button_y, reset_button_w, reset_button_h = 10, SCREEN_SIZE[1] - 60, 100, 40
    draw_rounded_rect(screen, reset_button_color, (reset_button_x, reset_button_y, reset_button_w, reset_button_h), 10)
    reset_text = font.render("Reset", True, (0, 0, 0))
    screen.blit(reset_text, (reset_button_x + 25, reset_button_y + 10))

    # Draw bot choice buttons
    bot_x1, bot_y1, bot_width1, bot_height1 = draw_bot_choice_button(screen, font, 1)
    bot_x2, bot_y2, bot_width2, bot_height2 = draw_bot_choice_button(screen, font, 2)

    # Draw colorbar for the score
    score = sum(opinions) / len(opinions)
    bar_x_start, bar_y_start, bar_width, bar_height = 120, SCREEN_SIZE[1] - 60, SCREEN_SIZE[0] - 140, 20
    pygame.draw.rect(screen, NODE_COLORS[0], (bar_x_start, bar_y_start, bar_width // 2, bar_height))
    pygame.draw.rect(screen, NODE_COLORS[1], (bar_x_start + bar_width // 2, bar_y_start, bar_width // 2, bar_height))
        
    # Draw a vertical indicator based on the score
    indicator_x = bar_x_start + int((score + 1) / 2 * bar_width)
    pygame.draw.line(screen, (0, 0, 0), (indicator_x, bar_y_start), (indicator_x, bar_y_start + bar_height), 3)
    
    # Label the bar
    label_surface = font.render("Score", True, (0, 0, 0))
    screen.blit(label_surface, (bar_x_start + bar_width // 2 - 20, bar_y_start - 30))

    # Display turn count
    turn_surface = font.render(f"Turn: {turn_count}", True, (0, 0, 0))
    screen.blit(turn_surface, (10, 10))

    # Display ask coach button
    coach_x, coach_y, coach_width, coach_height = ask_coach_res = draw_ask_coach(screen, font)

    # draw a yellow circle around the highlighted node
    if highlighted_node is not None:
        pygame.draw.circle(screen, (0, 255, 0), nodes[highlighted_node], NODE_RADIUS + 45, 20)

    # Display current player
    player_surface = font.render(f"Player {current_player + 1}'s Turn", True, (0, 0, 0))
    screen.blit(player_surface, (SCREEN_SIZE[0] - 500, 10))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()

