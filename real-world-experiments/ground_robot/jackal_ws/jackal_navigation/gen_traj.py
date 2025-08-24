import pygame
import json

# Initialize pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw Trajectory with Mouse")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_COLOR = (0, 0, 255)

# Set up the clock
clock = pygame.time.Clock()

# List to store the trajectory points
trajectory = []

# Flag to check if the mouse is being pressed
drawing = False

# Main loop
running = True
while running:
    screen.fill(WHITE)  # Fill the screen with white color

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button pressed
                drawing = True
                trajectory.clear()  # Clear previous trajectory when starting a new one
                # Add the first point
                x, y = pygame.mouse.get_pos()
                trajectory.append((x, y))
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button released
                drawing = False
                # Save the trajectory to a file when done
                with open("trajectory.json", "w") as file:
                    json.dump(trajectory, file)
                print(f"Trajectory saved to 'trajectory.json'")

    # Draw the trajectory
    if drawing:
        x, y = pygame.mouse.get_pos()
        trajectory.append((x, y))  # Add new point to the trajectory
        pygame.draw.circle(screen, LINE_COLOR, (x, y), 5)  # Draw the current mouse position

    # Draw the trajectory line
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            pygame.draw.line(screen, LINE_COLOR, trajectory[i-1], trajectory[i], 2)

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
