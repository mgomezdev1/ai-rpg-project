import pygame

from twiland import TwiLand
from utils import *

rendering_enabled = False
screen : pygame.Surface = None
font : pygame.font.Font = None

preloaded_skill_images : dict[int, pygame.Surface] = {}
preloaded_resource_images : dict[int, pygame.Surface] = {}

# Set the width and height of the screen [width, height]
SCREEN_SIZE = (1200, 800)

# Colors
PLAYER_COLOR = (125,0,255)
ENEMY_COLOR = (255,0,0)

# Set some sizing constants
MAP_SIZE = (700, 700)
MAP_OFFSETS = tuple((sx - x) / 2 for sx,x in zip(SCREEN_SIZE,MAP_SIZE))

FONT_SIZE = 20

ICON_SIZE = 64
ICON_SPACING = 4
LEFT_ICONS = MAP_OFFSETS[0] / 4
RIGHT_ICONS = SCREEN_SIZE[0] - (MAP_OFFSETS[0] * 3 / 4)
TEXT_TOP_OFFSET = ICON_SIZE / 2 - FONT_SIZE / 2

# Will get recalculated on certain functions (side effects!)
pixel_size = (1,1)

def env_to_screen(env : TwiLand, coord: tuple[int,int], transpose: bool = True) -> tuple[int,int]:
    global pixel_size
    if transpose:
        coord = (coord[1], coord[0])
    pixel_size = tuple(mx / sx for sx,mx in zip(env.land.shape, MAP_SIZE))
    return tuple(offset + (x + 0.5) * px for offset,x,px in zip(MAP_OFFSETS, coord, pixel_size))

def enable_rendering():
    global rendering_enabled
    global screen
    global font
    global preloaded_skill_images
    global preloaded_resource_images
    if rendering_enabled:
        return
    print("Enabling rendering...")
    rendering_enabled = True
    pygame.init()
    font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("TwiLand")

    for identifier, img_name in [(RESOURCE_ENERGY, "Energy"), (RESOURCE_FRUIT, "Fruit"), (RESOURCE_WOOD, "Wood"), (RESOURCE_RAW_FISH, "Fish_Raw"), (RESOURCE_COOKED_FISH, "Fish_Cooked"), (RESOURCE_ORE, "Ore"), (RESOURCE_AXE, "Axe"), (RESOURCE_PICKAXE, "Pickaxe"), (RESOURCE_FISHING_ROD, "Fishing_Rod"), (RESOURCE_SWORD, "Sword")]:
        preloaded_resource_images[identifier] = pygame.transform.scale(pygame.image.load(f"./icons/{img_name}.png"), (ICON_SIZE, ICON_SIZE))
    for identifier, img_name in [(SKILL_CHOPPING, "Chopping_Skill"), (SKILL_FISHING, "Fishing_Skill"), (SKILL_MINING, "Mining_Skill"), (SKILL_CRAFTING, "Crafting_Skill"), (SKILL_COMBAT, "Combat_Skill")]:
        preloaded_skill_images[identifier] = pygame.transform.scale(pygame.image.load(f"./icons/{img_name}.png"), (ICON_SIZE, ICON_SIZE))
    
# Used to manage how fast the screen updates
clock = pygame.time.Clock()

def update_display(env : TwiLand):
    # --- Screen-clearing code goes here
    screen.fill((0,0,0))
 
    # --- Drawing code should go here
    map_img = pygame.transform.scale(pygame.image.load(env.map_img_path), MAP_SIZE)
    screen.blit(map_img, MAP_OFFSETS, map_img.get_rect())

    pygame.draw.circle(screen, PLAYER_COLOR, env_to_screen(env, env.player_position), pixel_size[0])
    for enemy in env.enemies:
        pygame.draw.circle(screen, ENEMY_COLOR, env_to_screen(env, enemy.position), pixel_size[0])

    y = ICON_SIZE + ICON_SPACING
    for skill in ORDERED_SKILLSET:
        img = preloaded_skill_images[skill]
        screen.blit(img, (LEFT_ICONS, y), img.get_rect())
        txt = font.render(f"Lv {env.player_skills[skill]:.0f}", True, (255,255,255))
        screen.blit(txt, (LEFT_ICONS + ICON_SIZE + ICON_SPACING, y + TEXT_TOP_OFFSET), txt.get_rect())
        y += ICON_SIZE + ICON_SPACING
        
    y = ICON_SIZE + ICON_SPACING
    for resource in ORDERED_RESOURCESET:
        img = preloaded_resource_images[resource]
        screen.blit(img, (RIGHT_ICONS, y), img.get_rect())
        txt = font.render(f"x {env.resources[resource]:.2f}" if resource == RESOURCE_ENERGY else f"x {env.resources[resource]:.0f}", True, (255,255,255))
        screen.blit(txt, (RIGHT_ICONS + ICON_SIZE + ICON_SPACING, y + TEXT_TOP_OFFSET), txt.get_rect())
        y += ICON_SIZE + ICON_SPACING

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Simple pygame program
