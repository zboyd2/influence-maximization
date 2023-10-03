import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (800, 600)
NUM_NODES = 50
NODE_RADIUS = 20
CONTROL_MARK_RADIUS = 5
LINE_THICKNESS = 2
NODE_COLORS = [(255, 0, 0), (0, 0, 255)]
CONTROL_MARK_COLORS = [(255, 255, 255), (0, 0, 0)]
FONT_SIZE = 24

# Initialize screen and clock
screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("DISE: Dynamic Influence Spread Estimator")
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)


def initialize_graph():
    # Initialize Nodes and Edges
    nodes = []
    while len(nodes) < NUM_NODES:
        new_node = (random.randint(50, SCREEN_SIZE[0] - 50), random.randint(50, SCREEN_SIZE[1] - 50))
        if all(np.linalg.norm(np.array(new_node) - np.array(existing_node)) > 2 * NODE_RADIUS for existing_node in nodes): nodes.append(new_node)
    edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) < 100]

    # Create adjacency list
    adj_list = [[] for _ in range(NUM_NODES)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return nodes, edges, adj_list

nodes, edges, adj_list = initialize_graph()
opinions = [0 for _ in range(NUM_NODES)]
controls = [None for _ in range(NUM_NODES)]

# Initialize game state
current_player = 0
turn_count = 0
resize_cooldown = 0  # To control the cooldown after a resize event

# For buffering VIDEORESIZE events
latest_resize_event = None

#p1_bot_index = 0
#p2_bot_index = 0
#bot_names = ["Human", "Easy", "Medium", "Hard"]
#bot_colors = [(200, 200, 200), (220, 220, 220), (240, 240, 240), (250, 250, 250)]
#def draw_bot_choice_button(screen, font, current_player):
#    button_width, button_height = 150, 40
#    button_x = SCREEN_SIZE[0] - (button_width + 10) * current_player
#    button_y = 60
#    
#    # Draw the button
#    if current_player == 1:
#        bot_index = p1_bot_index
#    else:
#        bot_index = p2_bot_index
#    pygame.draw.rect(screen, bot_colors[bot_index], (button_x, button_y, button_width, button_height))
#    label = font.render(bot_names[bot_index], True, (0, 0, 0))
#    screen.blit(label, (button_x + 30, button_y + 10))
#    
#    # Return the button's rectangle for click detection
#    return (button_x, button_y, button_width, button_height)
p1_bot_index = 0
p2_bot_index = 0
bot_names = ["Human", "Easy", "Medium", "Hard"]
bot_colors = [(200, 200, 200), (220, 220, 220), (240, 240, 240), (250, 250, 250)]

def draw_bot_choice_button(screen, font, current_player):
    button_width, button_height = 150, 40
    button_x = SCREEN_SIZE[0] - (button_width + 10) * (3-current_player)
    button_y = 80

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
    pygame.draw.rect(screen, bot_colors[bot_index], (button_x, button_y, button_width, button_height))
    label = font.render(bot_names[bot_index], True, (0, 0, 0))
    screen.blit(label, (button_x + 30, button_y + 10))
    
    # Return the button's rectangle for click detection
    return (button_x, button_y, button_width, button_height)


# Game loop
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            # Check for clicks on the Reset button
            if reset_button_x <= x <= reset_button_x + reset_button_w and reset_button_y <= y <= reset_button_y + reset_button_h:
                current_player = 0
                turn_count = 0
                controls = [None for _ in range(NUM_NODES)]
                opinions = [0 for _ in range(NUM_NODES)]  # Reset opinions to neutral
                continue  # Skip the rest of this loop iteration
        
            # Check for clicks on nodes
            for i, (nx, ny) in enumerate(nodes):
                if np.linalg.norm(np.array([x, y]) - np.array([nx, ny])) < NODE_RADIUS:
                    if controls[i] is None:
                        controls[i] = current_player
                        current_player = 1 - current_player
                        turn_count += 1

            # Check for changes in bot type
            if bot_x1 <= event.pos[0] <= bot_x1 + bot_width1 and bot_y1 <= event.pos[1] <= bot_y1 + bot_height1:
                    p1_bot_index = (p1_bot_index + 1) % len(bot_names)
            if bot_x2 <= event.pos[0] <= bot_x2 + bot_width2 and bot_y2 <= event.pos[1] <= bot_y2 + bot_height2:
                    p2_bot_index = (p2_bot_index + 1) % len(bot_names)

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
        
        # Scale node positions and other elements
        nodes = [(int(x * x_scale), int(y * y_scale)) for (x, y) in nodes]
        NODE_RADIUS = int(NODE_RADIUS * (x_scale + y_scale) / 2)  # average scaling factor for radius
        
        # Resize screen
        screen = pygame.display.set_mode((latest_resize_event.w, latest_resize_event.h), pygame.RESIZABLE)

        # Reset the cooldown timer
        resize_cooldown = pygame.time.get_ticks()
        latest_resize_event = None

        # Explicitly fill screen with a background color
        screen.fill((245, 245, 245))
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
    screen.fill((245, 245, 245))

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
    pygame.draw.rect(screen, reset_button_color, (reset_button_x, reset_button_y, reset_button_w, reset_button_h))
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

    # Display current player
    player_surface = font.render(f"Player {current_player + 1}'s Turn", True, (0, 0, 0))
    screen.blit(player_surface, (SCREEN_SIZE[0] - 150, 10))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()

