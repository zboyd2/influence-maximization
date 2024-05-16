import time
import pygame
import numpy as np
import influence_maximization_algorithms as im
import graph_types as gt

import pygame_widgets
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

#Import Constants
from global_constants import NODE_RADIUS, CONTROL_MARK_RADIUS, LINE_THICKNESS, FONT_SIZE, TURNS_PER_PLAYER

# DECLARE GAME VARIABLES  - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
# Initialize Pygame
pygame.init()
once = True
NAVY = (0, 46, 93)
WHITE = (255, 255, 255)
ROYAL = (0, 61, 165)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)
NODE_COLORS = [RED, BLUE]
CONTROL_MARK_COLORS = [WHITE, BLACK]

# Initialize screen and clock
SCREEN_SIZE = (800, 600)
screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("DISE: Dynamic Influence Spread Estimator")
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)
highlighted_node = None

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
wait_time = 3
first_move = True

# GAME LOGIC FUNCTIONS - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
node_count = 0
def modify_node_count(n):
    global node_count
    node_count = n

def initialize_graph(n, approach): 
    global node_count
    """Generate the graph to play the game on based on the 
    selected number of nodes and graph type
    """

    #Get node coordinates and edges pairs based on the selected graph type. Adjust node_count if it was changed during graph generation
    if approach == "distribution":
        nodes, edges = gt.random_proximity_probability(n, SCREEN_SIZE)
    elif approach == "tree":
        nodes, edges = gt.tree(n, SCREEN_SIZE)
    elif approach == "ladder":
        node_count, nodes, edges = gt.ladder(n, SCREEN_SIZE)
    elif approach == "square":
        node_count, nodes, edges = gt.square_lattice(n, SCREEN_SIZE)
    elif approach == "hexagon":
        node_count, nodes, edges = gt.hexagon_lattice(n, SCREEN_SIZE)
    elif approach == "triangle":
        node_count, nodes, edges = gt.triangle_lattice(n, SCREEN_SIZE)
    elif approach == "cycle":
        nodes, edges = gt.cycle(n, SCREEN_SIZE)
    else: #Default
        nodes, edges = gt.random_proxmity(n, SCREEN_SIZE)
    
    # Create adjacency list
    adj_list = [[] for _ in range(node_count)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return nodes, edges, adj_list


def get_graph_laplacian(edges):
    # Create adjacency matrix
    adj_mat = np.zeros((node_count, node_count))
    for i, j in edges:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # Get degree matrix
    deg_mat = np.diag(np.sum(adj_mat, axis=1))

    return deg_mat - adj_mat  # Laplacian matrix


# FUNCTIONS FOR DRAWING GUI - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
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

#Drawing End Game Button
def draw_end_game(screen, font, winner):
    button_width, button_height = 150, 100
    #button Coordinates
    button_x = (SCREEN_SIZE[0] // 2) - (button_width // 2)
    button_y = (SCREEN_SIZE[1] // 2) - (button_height // 2)
    corner_radius = 20
    #winner box colors
    if winner == "Player 1":
        button_color = (220, 0, 0)
    elif winner == "Player 2":
        button_color = (0, 0, 220)
    else:
        button_color = (150, 100, 200)
    draw_rounded_rect(screen, button_color, (button_x, button_y, button_width, button_height), corner_radius)
    label = font.render(f"{f'{winner} wins!' if winner != 'Draw' else 'Draw!'}", True, (0, 0, 0))
    # Centering label within Button
    label_x = button_x + (button_width - label.get_width()) // 2
    label_y = button_y + (button_height - label.get_height()) // 2
    screen.blit(label, (label_x, label_y))

def draw_start_up_screen(screen, font):
    global node_count_slider, graph_type_dropdown, player1_dropdown, player2_dropdown, node_count_display
    
    screen.fill((255, 255, 255))

    title = font.render('Influence Maximization Game', True, (0, 0, 0))
    screen.blit(title, (SCREEN_SIZE[0] // 2 - title.get_width() // 2, 10))

    node_count_setting = font.render('Number of Nodes: ', True, (0, 0, 0))
    screen.blit(node_count_setting, (SCREEN_SIZE[0] // 4, SCREEN_SIZE[1] // 4))

    graph_type_setting = font.render('Graph Type: ', True, (0, 0, 0))
    screen.blit(graph_type_setting, (SCREEN_SIZE[0] // 4, 1.5 * SCREEN_SIZE[1] // 4))

    player_one_setting = font.render('Player 1 Setting: ', True, (0, 0, 0))
    screen.blit(player_one_setting, (SCREEN_SIZE[0] // 4, 2 * SCREEN_SIZE[1] // 4))
    
    player_two_setting = font.render('Player 2 Setting: ', True, (0, 0, 0))
    screen.blit(player_two_setting, (SCREEN_SIZE[0] // 4, 2.5 * SCREEN_SIZE[1] // 4))

    enter_to_continue = font.render('Press Enter to Start the Game!', True, (0, 0, 0))
    screen.blit(enter_to_continue, (SCREEN_SIZE[0] // 2 - title.get_width() // 2, SCREEN_SIZE[1] - 30))

    
node_count_slider = Slider(screen, 8 * SCREEN_SIZE[0] // 17, SCREEN_SIZE[1] // 4, 200, 20, min=10, max=100, step=1, initial=50)
node_count_display = TextBox(screen, 34 * SCREEN_SIZE[0] // 45, SCREEN_SIZE[1] // 4 - 5, 30, 30, fontSize=20)
node_count_display.disable() 

#Dropdown menu for graph type
graph_type_dropdown = Dropdown(screen, 8 * SCREEN_SIZE[0] // 17, 1.5 * SCREEN_SIZE[1] // 4 - 5, 200, 30, 
        name='Select Graph Type', 
        choices=['Geometric Random', 'Edge Probability Function', 'Tree', 'Ladder', 'Square Lattice', 'Hexagon Lattice', 'Triangle Lattice', 'Cycle'], 
        borderRadius=3, 
        colour=pygame.Color('gray'), 
        values=['default', 'distribution', 'tree', 'ladder', 'square', 'hexagon', 'triangle', 'cycle'], 
        direction='down', 
        textHAlign='centre'
)

#Dropdown menu for player 1 setttings
player1_dropdown = Dropdown(screen, 8 * SCREEN_SIZE[0] // 17, 2.0 * SCREEN_SIZE[1] // 4 - 5, 200, 30, 
        name='Select Difficulty', 
        choices=['Human', 'Easy Bot', 'Medium Bot', 'Hard Bot'], 
        borderRadius=3, 
        colour=pygame.Color('gray'), 
        values=[None, im.gui_easy_opponent, im.gui_greedy_algorithm, im.gui_minimax_algorithm_opt], 
        direction='down', 
        textHAlign='centre'
)

#Dropdown menu for player 2 settings
player2_dropdown = Dropdown(screen, 8 * SCREEN_SIZE[0] // 17, 2.5 * SCREEN_SIZE[1] // 4 - 5, 200, 30, 
        name='Select Difficulty', 
        choices=['Human', 'Easy Bot', 'Medium Bot', 'Hard Bot'], 
        borderRadius=3, 
        colour=pygame.Color('gray'), 
        values=[None, im.gui_easy_opponent, im.gui_greedy_algorithm, im.gui_minimax_algorithm_opt], 
        direction='down', 
        textHAlign='centre'
)
widget_list = [node_count_slider, node_count_display, graph_type_dropdown, player1_dropdown, player2_dropdown]

def update_widget_positioning(widget, x_scale, y_scale):
    #widget.set('width', int(widget.getWidth() * x_scale))
    #widget.set('height', int(widget.getHeight() * y_scale))
    widget.set('x', int(widget.getX() * x_scale))
    widget.set('y', int(widget.getY() * y_scale))

# GAME LOOPS  - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
running1 = True # For Start-up screen
running2 = True # For Game screen

# START UP SCREEN
while running1:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running1 = False
            running2 = False
        elif event.type == pygame.VIDEORESIZE:
            latest_resize_event = event
    
    if latest_resize_event and pygame.time.get_ticks() - resize_cooldown > 500: # 500ms cool-down
        # Get new dimensions
        new_w, new_h = latest_resize_event.w, latest_resize_event.h
        
        # Calculate scaling factors
        x_scale = new_w / SCREEN_SIZE[0]
        y_scale = new_h / SCREEN_SIZE[1]
        
        # Update screen size constants
        SCREEN_SIZE = (new_w, new_h)
        
        # Update the location of the Pygame Widgets
        for widget in widget_list:
            update_widget_positioning(widget, x_scale, y_scale)

        # Resize screen
        screen = pygame.display.set_mode((latest_resize_event.w, latest_resize_event.h), pygame.RESIZABLE)

        # Reset the cooldown timer
        resize_cooldown = pygame.time.get_ticks()
        latest_resize_event = None

        # Explicitly fill screen with a background color
        screen.fill(WHITE)
        pygame.display.flip()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_RETURN]: 
        # Generates the graph according to the entered settings. Switches the game state.
        modify_node_count(node_count_slider.getValue())
        nodes, edges, adj_list = initialize_graph(node_count, graph_type_dropdown.getSelected())
        opponent1 = player1_dropdown.getSelected()
        opponent2 = player2_dropdown.getSelected()
        laplacian = get_graph_laplacian(edges)
        opinions = [0 for _ in range(node_count)]
        controls = [None for _ in range(node_count)]
        running1 = False
    node_count_slider.listen(events)
    node_count_display.setText(str(node_count_slider.getValue()))
    node_count_display.listen(events)
    screen.fill(WHITE)
    draw_start_up_screen(screen, font)
    pygame_widgets.update(events)
    pygame.display.flip()

# GAME SCREEN
while running2:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running2 = False
    
        # Check for button clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # Check for clicks on the Reset button
            if reset_button_x <= x <= reset_button_x + reset_button_w and reset_button_y <= y <= reset_button_y + reset_button_h:
                current_player = 0
                turn_count = 0
                controls = [None for _ in range(node_count)]
                opinions = [0 for _ in range(node_count)]  # Reset opinions to neutral
                config = np.array([])
                first_move = True
                p1_bot_index = 0
                p2_bot_index = 0
                highlighted_node = None
                once = True

                nodes, edges, adj_list = initialize_graph(node_count, graph_type_dropdown.getSelected())

                # Get Laplacian matrix
                laplacian = get_graph_laplacian(edges)
                continue  # Skip the rest of this loop iteration
            # Check for clicks on the Ask Coach button
            elif turn_count < TURNS_PER_PLAYER * 2 and coach_x <= event.pos[0] <= coach_x + coach_width and coach_y <= event.pos[
                1] <= coach_y + coach_height:
                # highlight an optimal node to pick that is not already controlled by a player
                highlighted_node = im.gui_minimax_algorithm_opt(laplacian, config, depth)

        elif event.type == pygame.VIDEORESIZE:
            latest_resize_event = event

        # HUMAN MOVES
        if turn_count < TURNS_PER_PLAYER * 2:  # While turns remain
            if 1 - current_player:  # Player 1
                if opponent1 is None:
                    if event.type == pygame.MOUSEBUTTONDOWN: 
                        x, y = event.pos
                        # Check for clicks on nodes
                        for i, (n_x, n_y) in enumerate(nodes):
                            if np.linalg.norm(np.array([x, y]) - np.array([n_x, n_y])) < NODE_RADIUS:
                                if controls[i] is None:
                                    controls[i] = current_player
                                    current_player = 1 - current_player
                                    turn_count += 1
                                    config = np.append(config, i)
                                    start_time_p1 = time.perf_counter()  # Reset the timer
                                highlighted_node = None
            else:  # Player 2
                if opponent2 is None:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        # Check for clicks on nodes
                        for i, (n_x, n_y) in enumerate(nodes):
                            if np.linalg.norm(np.array([x, y]) - np.array([n_x, n_y])) < NODE_RADIUS:
                                if controls[i] is None:
                                    controls[i] = current_player
                                    current_player = 1 - current_player
                                    turn_count += 1
                                    config = np.append(config, i)
                                    start_time_p2 = time.perf_counter()  # Reset the timer
                                highlighted_node = None

        # WINDOW RESIZE
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
    for i in range(node_count):
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

    # Draw a green circle around the highlighted node
    if highlighted_node is not None:
        pygame.draw.circle(screen, (0, 255, 0), nodes[highlighted_node], NODE_RADIUS + 45, 20)

    # Display current player
    player_surface = font.render(f"Player {current_player + 1}'s Turn", True, (0, 0, 0))
    screen.blit(player_surface, (SCREEN_SIZE[0] - 500, 10))

    # If the game is over, display the winner
    if turn_count >= TURNS_PER_PLAYER * 2:
        influence = im.get_influence(laplacian, config)
        if once:
            once = not once
            print(f"Player 1: {influence[0]} | Player 2: {influence[1]}")
        
        winner = "Player 1"
        if np.abs(influence[0] - influence[1]) < 0.0000001:
            winner = "Draw"    
        elif influence[0] < influence[1]:
            winner = "Player 2"
    
        draw_end_game(screen, font, winner)
    
    pygame.display.flip()
    clock.tick(10)

    # BOT MOVES
    if turn_count < TURNS_PER_PLAYER * 2:  # While turns remain
        # Player 1 or Player 2
        if 1 - current_player:  # Player 1
            if opponent1 is not None:
                if time.perf_counter() - start_time_p2 > wait_time:
                    bot_move = opponent1(laplacian, config, depth)
                    config = np.append(config, bot_move)
                    controls[bot_move] = current_player
                    current_player = 1 - current_player
                    turn_count += 1
                    pygame.display.flip()
                    clock.tick(10)
                    start_time_p1 = time.perf_counter()  # Reset the timer
        else: # Player 2
            if opponent2 is not None:
                if time.perf_counter() - start_time_p1 > wait_time:
                    bot_move = opponent2(laplacian, config, depth)
                    config = np.append(config, bot_move)
                    controls[bot_move] = current_player
                    current_player = 1 - current_player
                    turn_count += 1
                    pygame.display.flip()
                    clock.tick(10)
                    start_time_p2 = time.perf_counter()  # Reset the timer

pygame.quit()