import sys
import rendering
import twiland
import pygame
from actions import *
from utils import *
from matplotlib import pyplot as plt 

env = twiland.TwiLand(twiland.generate_map((50,50)))

keys = {
    pygame.K_a: ACTION_MOVE_LEFT,
    pygame.K_w: ACTION_MOVE_UP, # Matrix coords are vertically flipped
    pygame.K_d: ACTION_MOVE_RIGHT,
    pygame.K_s: ACTION_MOVE_DOWN
}

dead = False
while True:
    if dead:
        env.set_map(twiland.generate_map((50,50)))
        env.reset()
        dead = False
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            for k,act in keys.items():
                if e.key == k:
                    _, r, ter, trun, info = env.step(act)
                    if ter or trun:
                        dead = True
            if e.key == pygame.K_p:
                twiland.draw_world(env.land, env.player_position, [e.position for e in env.enemies], show=True)
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    rendering.update_display(env)
 
    # --- Limit to 60 frames per second
    rendering.clock.tick(60)