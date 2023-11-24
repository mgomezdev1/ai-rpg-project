import pygame

from twiland import TwiLand

rendering_enabled = False
screen : pygame.Surface = None

# Set the width and height of the screen [width, height]
SCREEN_SIZE = (1200, 800)

# Colors
PLAYER_COLOR = (0,255,0)
ENEMY_COLOR = (255,0,0)

# Set some sizing constants
MAP_SIZE = (700, 700)
MAP_OFFSETS = tuple((sx - x) / 2 for sx,x in zip(SCREEN_SIZE,MAP_SIZE))

# Will get recalculated on certain functions (side effects!)
pixel_size = (1,1)

def env_to_screen(env : TwiLand, coord: tuple[int,int]) -> tuple[int,int]:
    global pixel_size
    pixel_size = tuple(mx / sx for sx,mx in zip(env.land.shape, MAP_SIZE))
    return tuple(offset + (x + 0.5) * px for offset,x,px in zip(MAP_OFFSETS, coord, pixel_size))

def enable_rendering():
    global rendering_enabled
    global screen
    if rendering_enabled:
        return
    rendering_enabled = True
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("TwiLand")
    
# Used to manage how fast the screen updates
clock = pygame.time.Clock()

def update_display(env : TwiLand):
 
    # --- Screen-clearing code goes here
    screen.fill((0,0,0))
 
    # --- Drawing code should go here
    map_img = pygame.transform.scale(pygame.image.load(env.map_img_path), MAP_SIZE)
    map_img.blit(screen, MAP_OFFSETS)

    pygame.draw.circle(screen, PLAYER_COLOR, env_to_screen(env, env.player_position), pixel_size[0])
    for enemy in env.enemies:
        pygame.draw.circle(screen, ENEMY_COLOR, env_to_screen(env, enemy.position), pixel_size[0])

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Simple pygame program





'''
# Import and initialize the pygame library
import pygame
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode(SCREEN_SIZE)

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    map_img = pygame.transform.scale(pygame.image.load("./temp/img/twiland_map.png"), MAP_SIZE)
    map_img.blit(screen, MAP_OFFSETS)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()'''