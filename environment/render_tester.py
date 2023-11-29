import sys
import rendering
import twiland
import pygame
from actions import *
from utils import *
from recipes import *
from matplotlib import pyplot as plt 

env = twiland.TwiLand(twiland.generate_map((50,50)))

keys = {
    pygame.K_a: ACTION_MOVE_LEFT,
    pygame.K_w: ACTION_MOVE_UP, # Matrix coords are vertically flipped
    pygame.K_d: ACTION_MOVE_RIGHT,
    pygame.K_s: ACTION_MOVE_DOWN,
    pygame.K_SPACE: ACTION_INTERACT,
    pygame.K_LEFT: ACTION_INTERACT_LEFT,
    pygame.K_UP: ACTION_INTERACT_UP,
    pygame.K_RIGHT: ACTION_INTERACT_RIGHT,
    pygame.K_DOWN: ACTION_INTERACT_DOWN,
    pygame.K_z: ACTION_CRAFT + len(TRAINING_RECIPES), # Eat apple
    pygame.K_x: ACTION_CRAFT + len(TRAINING_RECIPES) + 1, # Eat raw fish
    pygame.K_c: ACTION_CRAFT + len(TRAINING_RECIPES) + 2, # Eat cooked fish
}
keys.update({pygame.K_0 + i: ACTION_CRAFT + len(TRAINING_RECIPES) + len(EATING_RECIPES) + i for i in range(len(CRAFTING_RECIPES))})

dead = False
score = 0
while True:
    if dead:
        env.set_map(twiland.generate_map((50,50)))
        env.reset()
        score = 0
        dead = False
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            for k,act in keys.items():
                if e.key == k:
                    obs, r, ter, trun, info = env.step(act)
                    print(obs)
                    score += r
                    if ter or trun:
                        dead = True
            if e.key == pygame.K_p:
                twiland.draw_world(env.land, env.player_position, [e.position for e in env.enemies], show=True)
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    rendering.set_title_text(f"Current Day: {env.time:.0f}. Score: {score}")
    rendering.update_display(env)
 
    # --- Limit to 60 frames per second
    rendering.clock.tick(60)