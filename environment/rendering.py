from typing import Literal, TypeAlias
import pygame
import sys

from twiland import VIEW_DISTANCE, VIEW_NORM, NormType, Observation, TwiLand, get_mask, land_info
from utils import *

rendering_enabled = False
screen : pygame.Surface = None
font : pygame.font.Font = None
large_font : pygame.font.Font = None
RENDER_FULL = "full"
RENDER_OBS = "obs"
RenderMode: TypeAlias = Literal["full","obs"]
rendering_mode : RenderMode = RENDER_FULL

preloaded_skill_images : dict[int, pygame.Surface] = {}
preloaded_resource_images : dict[int, pygame.Surface] = {}

# Set the width and height of the screen [width, height]
SCREEN_SIZE = (1200, 800)
BACKGROUND_COLOR = (15,15,15)

# Colors
PLAYER_COLOR = (125,0,255)
ENEMY_COLOR = (255,0,0)

# Set some sizing constants
MAP_SIZE = (700, 700)
MAP_OFFSETS = tuple((sx - x) / 2 for sx,x in zip(SCREEN_SIZE,MAP_SIZE))

FONT_SIZE = 20
LARGE_FONT_SIZE = 32

ICON_SIZE = 64
ICON_SPACING = 4
LEFT_ICONS = MAP_OFFSETS[0] / 4
RIGHT_ICONS = SCREEN_SIZE[0] - (MAP_OFFSETS[0] * 3 / 4)
TEXT_TOP_OFFSET = ICON_SIZE / 2 - FONT_SIZE / 2

# Title text
TITLE_MARGIN = 10
title = ""

# Will get recalculated on certain functions (side effects!)
pixel_size = (1,1)

def env_to_screen(env : TwiLand, coord: tuple[int,int], transpose: bool = True) -> tuple[int,int]:
    global pixel_size
    if transpose:
        coord = (coord[1], coord[0])
    pixel_size = tuple(mx / sx for sx,mx in zip(env.land.shape, MAP_SIZE))
    return tuple(offset + (x + 0.5) * px for offset,x,px in zip(MAP_OFFSETS, coord, pixel_size))

land_colors = {}
def render_observation_map(observation: Observation, view_distance: int = VIEW_DISTANCE, view_norm : NormType = VIEW_NORM):
    from matplotlib import pyplot as plt
    mask = get_mask(view_distance * 2 + 1, view_norm)
    shape = mask.shape
    map_matrix = np.zeros(shape, dtype=np.int8)
    mask_size = mask.sum()
    enemy_id = len(land_info) + 1
    np.place(map_matrix, mask, observation.flat_enemies.astype(np.int8) * enemy_id)
    for i,t in enumerate(land_info):
        new_elements = np.zeros(shape)
        np.place(new_elements, mask, observation.flat_neighbors_encoded[i*mask_size:(i+1)*mask_size] * (t.id + 1))
        map_matrix = np.maximum(map_matrix, new_elements)
    mx, my = map_matrix.shape
    width, height = MAP_SIZE
    left, top = MAP_OFFSETS
    dx, dy = width / mx, height / my
    for i in range(my):
        for j in range(mx):
            col = land_colors[map_matrix[i,j] - 1]
            pygame.draw.rect(screen, col, (left + dx * j, top + dy * i, dx, dy))    

def enable_rendering():
    global rendering_enabled
    global screen
    global font
    global large_font
    global preloaded_skill_images
    global preloaded_resource_images
    if rendering_enabled:
        return
    print("Enabling rendering...")
    rendering_enabled = True
    pygame.init()
    font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)
    large_font = pygame.font.Font(pygame.font.get_default_font(), LARGE_FONT_SIZE)
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("TwiLand")

    app_icon = pygame.image.load("./icons/App_Icon.png")
    pygame.display.set_icon(app_icon)

    for identifier, img_name in [(RESOURCE_ENERGY, "Energy"), (RESOURCE_FRUIT, "Fruit"), (RESOURCE_WOOD, "Wood"), (RESOURCE_RAW_FISH, "Fish_Raw"), (RESOURCE_COOKED_FISH, "Fish_Cooked"), (RESOURCE_ORE, "Ore"), (RESOURCE_AXE, "Axe"), (RESOURCE_PICKAXE, "Pickaxe"), (RESOURCE_FISHING_ROD, "Fishing_Rod"), (RESOURCE_SWORD, "Sword")]:
        preloaded_resource_images[identifier] = pygame.transform.scale(pygame.image.load(f"./icons/{img_name}.png"), (ICON_SIZE, ICON_SIZE))
    for identifier, img_name in [(SKILL_CHOPPING, "Chopping_Skill"), (SKILL_FISHING, "Fishing_Skill"), (SKILL_MINING, "Mining_Skill"), (SKILL_CRAFTING, "Crafting_Skill"), (SKILL_COMBAT, "Combat_Skill")]:
        preloaded_skill_images[identifier] = pygame.transform.scale(pygame.image.load(f"./icons/{img_name}.png"), (ICON_SIZE, ICON_SIZE))
    global land_colors
    for l in land_info:
        land_colors[l.id] = hex_to_rgb(l.color)
    land_colors[-1] = (0,0,0)
    land_colors[len(land_info)] = ENEMY_COLOR

def set_title_text(new_text: str):
    global title
    title = new_text

def set_rendering_mode(new_rendering_mode: RenderMode):
    global rendering_mode
    rendering_mode = new_rendering_mode
    
def toggle_rendering_mode():
    set_rendering_mode(RENDER_FULL if rendering_mode == RENDER_OBS else RENDER_OBS)

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

def update_display(env : TwiLand):
    # --- Screen-clearing code goes here
    screen.fill(BACKGROUND_COLOR)
 
    # --- Drawing code should go here
    if rendering_mode == RENDER_FULL:
        map_img = pygame.transform.scale(pygame.image.load(env.map_img_path), MAP_SIZE)
        screen.blit(map_img, MAP_OFFSETS, map_img.get_rect())

        pygame.draw.circle(screen, PLAYER_COLOR, env_to_screen(env, env.player_position), pixel_size[0])
        for enemy in env.enemies:
            pygame.draw.circle(screen, ENEMY_COLOR, env_to_screen(env, enemy.position), pixel_size[0])
    else:
        render_observation_map(env.get_observation())

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

    title_txt = large_font.render(title, True, (255,255,255))
    screen.blit(title_txt, (SCREEN_SIZE[0] / 2 - title_txt.get_rect().width / 2, TITLE_MARGIN), title_txt.get_rect())

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Simple pygame program

def basic_event_loop():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_TAB:
                toggle_rendering_mode()