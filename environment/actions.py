import numpy as np
from recipes import RECIPESET

from utils import RESOURCE_COOKED_FISH, RESOURCE_ENERGY, RESOURCE_RAW_FISH, RESOURCE_WOOD, SKILLSET, Recipe

# Moving to adjacent edges
ACTION_WAIT = 0
ACTION_MOVE_RIGHT = 1
ACTION_MOVE_UP = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_DOWN = 4
ACTIONSET_MOVE = {ACTION_WAIT, ACTION_MOVE_RIGHT, ACTION_MOVE_UP, ACTION_MOVE_LEFT, ACTION_MOVE_DOWN}

# Extracting resources or attacking enemies in tiles
ACTION_INTERACT = 5
ACTION_INTERACT_RIGHT = 6
ACTION_INTERACT_UP = 7
ACTION_INTERACT_LEFT = 8
ACTION_INTERACT_DOWN = 9
ACTIONSET_INTERACT = {ACTION_INTERACT, ACTION_INTERACT_RIGHT, ACTION_INTERACT_UP, ACTION_INTERACT_LEFT, ACTION_INTERACT_DOWN}

ACTION_CRAFT = 10
ACTIONSET_CRAFT = {ACTION_CRAFT + i for i in range(len(RECIPESET))}

position_offsets: list[tuple[int,int]] = [(0,0), (0,1), (-1,0), (0,-1), (1,0)]

ACTIONTYPE_MOVE = 0
ACTIONTYPE_INTERACT = 1
ACTIONTYPE_CRAFT = 2

ACTIONSET_ALL = ACTIONSET_MOVE.union(ACTIONSET_INTERACT).union(ACTIONSET_CRAFT)

def parse_action(action) -> tuple[int,int]:
    if action in ACTIONSET_MOVE:
        return (ACTIONTYPE_MOVE, action - ACTION_WAIT)
    elif action in ACTIONSET_INTERACT:
        return (ACTIONTYPE_INTERACT, action - ACTION_INTERACT)
    elif action in ACTIONSET_CRAFT:
        return (ACTIONTYPE_CRAFT, action - ACTION_CRAFT)

