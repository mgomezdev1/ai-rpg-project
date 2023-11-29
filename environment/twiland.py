import os
from matplotlib.image import AxesImage
import numpy as np
import scipy
from termcolor import colored
import gymnasium
import random
import torch.nn as nn
from typing import Literal, TypeAlias, TypeVar
from matplotlib import pyplot as plt
from matplotlib import colors

from actions import ACTIONSET_ALL, ACTIONTYPE_MOVE, ACTIONTYPE_INTERACT, ACTIONTYPE_CRAFT, parse_action, position_offsets
from recipes import ORDERED_RECIPES
from utils import *

TAU = np.pi * 2

LAND_PLAINS = 0
LAND_FOREST = 1
LAND_MOUNTAIN = 2
LAND_WATER = 3

skill_training_efficiency = {
    SKILL_CHOPPING: 1,
    SKILL_MINING: 1,
    SKILL_FISHING: 1,
    SKILL_CRAFTING: 0.5,
    SKILL_COMBAT: 0
}

ENEMY_SPAWN_RANGE = (3, 6)
VIEW_DISTANCE = 4
NormType : TypeAlias = Literal["1", "2", "inf"] 
VIEW_NORM : NormType = "1"
LAND_COMPRESSION = False
DEFAULT_MAP_SIZE = 50

class LandType:
    def __init__(self, name: str, id: int, representation: str, color: str, move_cost: float, harvest: Recipe):
        self.name = name
        self.id = id
        self.representation = representation
        self.color = color
        self.move_cost = move_cost,
        self.harvest = harvest

land_info = [
    LandType("Plains", LAND_PLAINS, colored("P", "black", "on_light_green"), "#44ff55", 0.1,
        Recipe(
            {RESOURCE_ENERGY: 1} # Wastes one energy
        )
    ),
    LandType("Forest", LAND_FOREST, colored("F", "white", "on_green"), '#00bb00', 0.15,
        RecipeStack(
            Recipe(
                {RESOURCE_ENERGY: 1.5, RESOURCE_AXE: 1}, 
                {RESOURCE_WOOD: 3, RESOURCE_FRUIT: 2, RESOURCE_AXE: 0.75}, # Base 25% chance of breaking the tool
                {SKILL_CHOPPING: 1}, # Action affected by the chopping skill
                {RESOURCE_WOOD: 0.5, RESOURCE_FRUIT: 0.2, RESOURCE_AXE: 0.02}, # Every skill point grants a cumulative 2% chance of not consuming the axe
                {SKILL_CHOPPING: 0.35}, 
                {RESOURCE_AXE: 0.95} # At least, 5% chance of breaking the axe
            ),
            Recipe(
                {RESOURCE_ENERGY: 2}, 
                {RESOURCE_WOOD: 1, RESOURCE_FRUIT: 1.5},
                {SKILL_CHOPPING: 1}, # Action affected by the chopping skill
                {RESOURCE_WOOD: 0.25, RESOURCE_FRUIT: 0.1}, # Every skill point grants an extra 0.25 wood and 0.1 fruit
                {SKILL_CHOPPING: 0.25} # Level up 25% of chopping per chopping
            )
        )
    ),
    LandType("Mountain", LAND_MOUNTAIN, colored("M", "white", "on_grey"), "#333355", 0.7,
        RecipeStack (
            Recipe(
                {RESOURCE_ENERGY: 1.5, RESOURCE_PICKAXE: 1}, 
                {RESOURCE_ORE: 2, RESOURCE_PICKAXE: 0.75}, # Base 25% chance of breaking the tool
                {SKILL_MINING: 1},
                {RESOURCE_ORE: 0.65, RESOURCE_PICKAXE: 0.02}, # Every skill point grants a cumulative 2% chance of not consuming the tool
                {SKILL_MINING: 0.5}, 
                {RESOURCE_AXE: 0.95} # At least, 5% chance of breaking the tool
            ),
            Recipe(
                {RESOURCE_ENERGY: 6}, 
                {RESOURCE_ORE: 1},
                {SKILL_MINING: 1}, # Action affected by the mining skill
                {RESOURCE_ORE: 0.25},
                {SKILL_MINING: 0.25}
            )
        )
    ),
    LandType("Water", LAND_WATER, colored("W", "white", "on_blue"), '#0000ff', 0.45,
        RecipeStack (
            Recipe(
                {RESOURCE_ENERGY: 2.5, RESOURCE_FISHING_ROD: 1}, 
                {RESOURCE_RAW_FISH: 2, RESOURCE_FISHING_ROD: 0.75}, # Base 25% chance of breaking the tool
                {SKILL_FISHING: 1},
                {RESOURCE_RAW_FISH: 0.65, RESOURCE_FISHING_ROD: 0.02}, # Every skill point grants a cumulative 2% chance of not consuming the tool
                {SKILL_FISHING: 0.6}, 
                {RESOURCE_FISHING_ROD: 0.95} # At least, 5% chance of breaking the tool
            ),
            Recipe(
                {RESOURCE_ENERGY: 5}, 
                {RESOURCE_RAW_FISH: 1},
                {SKILL_FISHING: 1}, # Action affected by the mining skill
                {RESOURCE_RAW_FISH: 0.2},
                {SKILL_FISHING: 0.25}
            )
        )
    )
]

rng = random.Random()

def random_walk(direction: float, length: int, volatility: float) -> list[tuple[int,int]]:
    """Returns a list of offsets for a random walk in a random direction"""
    tendency = 0
    pos = (0.5, 0.5)
    step_offsets = []
    for _ in range(length):
        tendency += (rng.random() * 2 - 1) * volatility
        direction += tendency
        delta = (np.cos(direction), np.sin(direction))
        pos = tuple(a+b for a,b in zip(pos,delta))
        floored_pos = tuple(int(np.floor(n)) for n in pos)
        step_offsets.append(floored_pos)
    return step_offsets

def random_blob(size: tuple[float,float], volatility: float) -> list[tuple[int,int]]:
    """returns a list of offsets to form a roughly ellipsoidal blob of a given size around a center"""
    sx,sy = size
    reach_x = int(np.ceil(sx + volatility))
    reach_y = int(np.ceil(sy + volatility))
    volatility_weight = volatility * volatility / (sx * sx + sy * sy)
    result = []
    for i in range(-reach_x, reach_x + 1):
        for j in range(-reach_y, reach_y + 1):
            weighted_dist = i * i / (sx * sx) + j * j / (sy * sy)
            weighted_dist += volatility_weight * (rng.random() * 2 - 1)
            if weighted_dist < 1:
                result.append((i,j))
    return result

def step(size: tuple[int,int], position: tuple[int,int], step: tuple[int,int]):
    """Steps in a specific direction around a map of a given size, wrapping around the edges."""
    x ,y  = position
    dx,dy = step
    sx,sy = size
    return ((x + dx) % sx, (y + dy) % sy)

T = TypeVar("T")
def set_offsets(land: np.ndarray[T], source: tuple[int,int], offsets: list[tuple[int,int]], value: T):
    pos = source
    for offset in offsets:
        loc = step(land.shape, pos, offset)
        land[loc] = value

def random_pos(size: tuple[int,int]) -> tuple[int,int]:
    sx,sy = size
    return (rng.randrange(0,sx), rng.randrange(0,sy))

def random_size(size_range: tuple[int,int]):
    """Generate a random integral size with a given minimum and maximum (both included) for all coordinates"""
    return (rng.randint(*size_range),rng.randint(*size_range))

def generate_map(size: tuple[int,int], **kwargs) -> np.ndarray[int]:
    """
    Generates a random map using a simple algorithm. Use the following kwargs to alter the generation of features
    `lake_count: int`, `lake_volatility: float`, `lake_size_range: tuple[int,int]`
    `river_count: int`, `river_volatility: float`, `river_length_range: tuple[int,int]`
    `mountain_count: int`, `mountain_volatility: float`, `mountain_size_range: tuple[int,int]`
    `forest_count: int`, `forest_volatility: float`, `forest_size_range: tuple[int,int]`

    The count indicates the number of features of the given type to generate (note that some may overlap). If river count is positive, mountain count must be positive.
    The volatility indicates the erraticity of the generation, basically, the higher it is, the more ragged blobs would be, and the more chaotic rivers will be.
    Size/Length ranges are a tuple (min,max), inclusive, for the random range to use for blob radii and random path lengths.
    """
    # Cover the entire area in plains
    result = np.ones(size, dtype=int) * LAND_PLAINS
    
    lake_count = kwargs.get("lake_count", int(size[0] * size[1] / 200))
    lake_volatility = kwargs.get("lake_volatility", 2.5)
    lake_size_range = kwargs.get("lake_size_range", (1,4))

    river_count = kwargs.get("river_count", int(size[0] * size[1] / 150))
    river_volatility = kwargs.get("river_volatility", 0.3)
    river_length_range = kwargs.get("river_length_range", (4, 24))

    mountain_count = kwargs.get("mountain_count", int(size[0] * size[1] / 100))
    mountain_volatility = kwargs.get("mountain_volatility", 4)
    mountain_size_range = kwargs.get("mountain_size_range", (1,3))
    
    forest_count = kwargs.get("forest_count", int(size[0] * size[1] / 100))
    forest_volatility = kwargs.get("forest_volatility", 1.5)
    forest_size_range = kwargs.get("forest_size_range", (3,5))

    peaks = [random_pos(size) for _ in range(mountain_count)]
    lakes = [
        (random_pos(size), random_blob(random_size(lake_size_range), lake_volatility))
        for _ in range(lake_count)
    ]
    rivers = [
        (rng.sample(peaks,1)[0], random_walk(rng.random() * TAU, rng.randint(*river_length_range), river_volatility))
        for _ in range(river_count)
    ]
    mountains = [
        (peak, random_blob(random_size(mountain_size_range), mountain_volatility))
        for peak in peaks    
    ]
    forests = [
        (random_pos(size), random_blob(random_size(forest_size_range), forest_volatility))
        for _ in range(forest_count)
    ]

    for source,offsets in forests:
        set_offsets(result, source, offsets, LAND_FOREST)
    for source,offsets in lakes:
        set_offsets(result, source, offsets, LAND_WATER)
    for source,offsets in mountains:
        set_offsets(result, source, offsets, LAND_MOUNTAIN)
    for source,offsets in rivers:
        set_offsets(result, source, offsets, LAND_WATER)
        
    return result

def print_map(land: np.ndarray[int]):
    sx, sy = land.shape
    for i in range(sx):
        for j in range(sy):
            print(land_info[land[i,j]].representation, end="")
        print()

def draw_map(land: np.ndarray[int], show = True) -> AxesImage:
    cmap = colors.ListedColormap([land_type.color for land_type in land_info])
    img = plt.imshow(land, cmap=cmap)
    if show: plt.show()
    return img

PLAYER_COLOR = "#cc22ff"
ENEMY_COLOR = "#ff0000"
def build_world_matrix(land: np.ndarray[int], player: tuple[int,int], enemies: list[tuple[int,int]]) -> np.ndarray:
    playerFlag = len(land_info)
    enemyFlag = playerFlag + 1
    world = land.copy()
    world[player] = playerFlag
    for enemy in enemies: world[enemy] = enemyFlag
    return world

def draw_world(land: np.ndarray[int], player: tuple[int,int], enemies: list[tuple[int,int]], show = True) -> AxesImage:
    cmap = colors.ListedColormap([land_type.color for land_type in land_info] + [PLAYER_COLOR, ENEMY_COLOR])
    world = build_world_matrix(land, player, enemies)
    img = plt.imshow(world, cmap=cmap)
    if show: plt.show()
    return img

def crop_map_submatrix(land: np.ndarray[int], center: tuple[int,int], radius: int) -> np.ndarray[int]:
    cx, cy = center
    size = 2 * radius + 1
    return np.roll(land, (-cx + radius, -cy + radius))[:size,:size]

def compute_mask(size: int, norm: NormType) -> np.ndarray[bool]:
    center = size // 2
    mask = np.ones((size, size))
    if norm == "inf":
        return mask
    else:
        dist1d = np.array([abs(i-center) for i in range(size)])
        aux = np.stack(
            [np.tile(dist1d.reshape(1,size), (size,1)),   
             np.tile(dist1d.reshape(size,1), (1,size))],
            axis=2   
        )
        if norm == "1":
            dist2d : np.ndarray = np.sum(aux, axis=2)
        elif norm == "2":
            dist2d : np.ndarray = np.linalg.norm(aux, axis=2)
        else:
            raise ValueError(f"Invalid norm given: {norm}")
        return (dist2d <= center) # Note that the "center" is the same as the radius
    
def get_mask(size: int, norm: NormType) -> np.ndarray[bool]:
    if (size, norm) not in precomputed_masks:
        precomputed_masks[(size, norm)] = compute_mask(size, norm)
    return precomputed_masks[(size, norm)]
            
precomputed_masks: np.ndarray[tuple[int,NormType], np.ndarray[bool]] = dict()
def crop_observation(submatrix: np.ndarray[int], norm: NormType, unknown_value : int = -1) -> np.ndarray[int]:
    sx, sy = submatrix.shape
    assert sx == sy
    if norm == "inf":
        return submatrix
    mask = get_mask(sx, norm)
    result = np.ones((sx,sx)) * unknown_value
    np.copyto(result, submatrix, where=mask)
    return result

class Enemy:
    def __init__(self, position: tuple[int,int], power: float = 1):
        self.position = position
        self.power = power
        self.dead = False

    def seek(self, target: tuple[int,int], land_size: tuple[int,int]) -> bool:
        if target == self.position:
            return True
        x,y = self.position
        tx,ty = target
        sx,sy = land_size
        dx = tx - x
        dy = ty - y
        if dx > sx / 2:
            dx -= sx
        elif dx < -sx / 2: # Too far 
            dx += sx
        if dy > sy / 2: # Too far down, better to wrap on the bottom edge
            dy -= sy
        elif dy < -sy / 2: # Too far up, better to wrap on the top edge
            dy += sy
        final_step = (np.sign(dx),0) if abs(dx) > abs(dy) else (0,np.sign(dy))
        self.position = step(land_size, self.position, final_step)
        return self.position == target
    
    def fight(self) -> Recipe:
        return RecipeStack(
            Recipe( # FIGHT WITH A SWORD
                {RESOURCE_ENERGY: self.power * 0.5, RESOURCE_SWORD: 1},
                {RESOURCE_SWORD: 0.5}, # Base 50% chance of keeping the sword
                {SKILL_COMBAT: 1, SKILL_CRAFTING: 0.2}, # Being 'crafty' can help!
                {RESOURCE_ENERGY: 0.5, RESOURCE_SWORD: 0.025}, # 2.5% extra chance of keeping the sword per combat level (and 5 crafting levels)
                {SKILL_COMBAT: 0.25},
                {RESOURCE_ENERGY: self.power * 0.5, RESOURCE_SWORD: 0.99} # Can't regain energy, at least a 1% chance of losing sword
            ),
            Recipe( # NO SWORD? FIGHT THE MONSTER WITH AN AXE!
                {RESOURCE_ENERGY: self.power * 0.75, RESOURCE_AXE: 1},
                {RESOURCE_AXE: 0.25}, # Pretty high chance of destroying the axe
                {SKILL_COMBAT: 0.5, SKILL_CHOPPING: 0.5}, # Helped by chopping skill
                {RESOURCE_ENERGY: 0.35, RESOURCE_AXE: 0.05}, # Use less energy and keep axe with better skill
                {SKILL_COMBAT: 0.25, SKILL_CHOPPING: 0.1},
                {RESOURCE_ENERGY: self.power * 0.75, RESOURCE_AXE: 0.98}
            ),
            Recipe( # FIGHT BARE-HANDED (not a great idea)
                {RESOURCE_ENERGY: self.power * 1.5},
                {}, # No rewards
                {SKILL_COMBAT: 1},
                {RESOURCE_ENERGY: 0.25}, # Use less energy the better one is at combat
                {SKILL_COMBAT: 1}, # Hard to fight without tools: guaranteed level up in combat!
                {RESOURCE_ENERGY: self.power * 1.5}
            )
        )

class Observation:
    def __init__(self, neighbours: np.ndarray[int], enemies: np.ndarray[int], time: float, skills: np.ndarray[float], resources: np.ndarray[int], view_mask: np.ndarray[bool]):
        self.flat_neighbors = np.extract(view_mask, neighbours)
        self.flat_enemies   = np.extract(view_mask, enemies)
        self.sigmoid_time   = scipy.special.expit(np.array([time / 5 - 1, ((time % 1) - 0.5) * 10]))
        self.sigmoid_skills = scipy.special.expit(skills / 5 - 1)
        self.sigmoid_resources = scipy.special.expit(resources / 5 - 1)
        # Not sure if we want to include sigmoid_resources in flattened data
        self.flattened_data = np.concatenate([self.flat_neighbors, self.flat_enemies, self.sigmoid_time, self.sigmoid_skills], axis=0, dtype=np.float32)

    def configured_size() -> int:
        return TwiLand(generate_map((3 * VIEW_DISTANCE, 3 * VIEW_DISTANCE)), enable_rendering=False).get_observation().flattened_data.shape[0]

class TwiLand(gymnasium.Env):
    def __init__(self, land: np.ndarray[int], player_position: tuple[int,int] | None = None, enable_rendering = True, 
            fail_reward: float = -1, fight_reward: float = 50, craft_reward: float = 20, harvest_reward: float = 5, survival_reward: float = 1,
            max_days: float = 10, starting_energy: int = 10, actions_per_day: int = 10, actions_per_night: int = 10, idle_cost: float = 0.1, enemy_difficulty_scaling: tuple[float] = (0,1.0,),
            **kwargs):
        self.land = land
        if not player_position:
            player_position = random_pos(land.shape)
        self.player_position = player_position
        self.enemies : list[Enemy] = []

        self.player_skills = np.ones(len(SKILLSET))
        self.resources = dict_to_array({RESOURCE_ENERGY: starting_energy}, len(RESOURCESET))
        self.time: float = 0
        self.tstep: int = 0 
        self.actions_per_day = actions_per_day
        self.actions_per_night = actions_per_night
        self.map_img_path = kwargs.get("img_path", "./temp/img/twiland_map.png")
        self.enable_rendering = enable_rendering
        self.fail_reward = fail_reward
        self.fight_reward = fight_reward
        self.craft_reward = craft_reward
        self.harvest_reward = harvest_reward
        self.urvival_reward = survival_reward
        self.max_days = max_days
        self.idle_cost = idle_cost
        self.enemy_difficulty_scaling = enemy_difficulty_scaling

        # variables for reset
        self.starting_energy = starting_energy

        self.info = {}
        if enable_rendering:
            import rendering
            self.save_map()
            rendering.enable_rendering()

    def spawn_enemies(self, count: int = 1, power: float = 1):
        for i in range(count):
            angle = rng.random() * np.pi * 2
            distance = rng.randint(*ENEMY_SPAWN_RANGE)
            offset = (int(distance * np.sin(angle)), int(distance * np.cos(angle)))
            position = step(self.land.shape, self.player_position, offset)
            self.enemies.append(Enemy(position, power))

    def set_map(self, land: np.ndarray[int]):
        self.land = land
        if self.enable_rendering:
            self.save_map()
        
    def save_map(self):
        cmap = colors.ListedColormap([land_type.color for land_type in land_info])
        dir = os.path.dirname(self.map_img_path)
        if (not os.path.exists(dir)): os.makedirs(dir)
        plt.imsave(self.map_img_path, self.land, cmap=cmap)

    def get_observation(self):
        land_submatrix = crop_map_submatrix(self.land, self.player_position, VIEW_DISTANCE)
        mask = get_mask(2 * VIEW_DISTANCE + 1, VIEW_NORM)
        enemy_matrix   = np.zeros(self.land.shape)
        for e in self.enemies:
            enemy_matrix[e.position] = 1
        enemy_submatrix = crop_map_submatrix(enemy_matrix, self.player_position, VIEW_DISTANCE)

        return Observation(land_submatrix, enemy_submatrix, self.time, self.player_skills, self.resources, mask)

    def reset(self) -> tuple[Observation, dict]:
        self.player_position = random_pos(self.land.shape)
        self.time = 0
        self.tstep = 0
        self.player_skills = np.ones(len(SKILLSET))
        self.resources = dict_to_array({RESOURCE_ENERGY: self.starting_energy}, len(RESOURCESET))
        self.enemies = []

        return (self.get_observation(), self.info)

    def get_enemy_power(self, time: float):
        x = 1
        power = 0
        for coeff in self.enemy_difficulty_scaling:
            power += coeff * x
            x *= time
        return power

    def _check_enemy_attacks(self, allow_move: bool = True) -> tuple[int, bool]:
        '''
            May move enemies and forces them to fight if they reach the player or the player reached them

            Returns the number of fights the player fought and a boolean representing whether the player survived the enemy's turn.
        '''
        num_fights = 0
        for e in self.enemies:
            if (e.seek(self.player_position, self.land.shape) if allow_move else e.position == self.player_position):
                success, new_r, new_s = e.fight().craft(self.resources, self.player_skills)
                if not success:
                    return num_fights, False
                e.dead = True
                self.resources = new_r
                self.player_skills = new_s
                num_fights += 1
        if num_fights > 0:
            self.enemies = [e for e in self.enemies if not e.dead]
        return num_fights, True

    def _fail(self) -> tuple[Observation, float, bool, bool, dict]:
        return self._environment_turn(self.fail_reward)

    def _death(self) -> tuple[Observation, float, bool, bool, dict]:
        print(self.tstep)
        return self.get_observation(), -1000, True, False, self.info

    def _environment_turn(self, partial_reward = 0) -> tuple[Observation, float, bool, bool, dict]:
        self.tstep += 1
        self.time += 1 / (self.actions_per_day + self.actions_per_night)
        self.resources[RESOURCE_ENERGY] -= self.idle_cost
        if (self.resources[RESOURCE_ENERGY] <= 0):
            return self._death()

        if self.tstep >= self.actions_per_day + self.actions_per_night:
            self.tstep = 0
        if self.tstep >= self.actions_per_day:
            # Enemies only move during the night...
            fights, survived = self._check_enemy_attacks(allow_move=True)
            if not survived:
                return self._death()
            partial_reward += fights * self.fight_reward
            # After their move, a new enemy may spawn...
            prob = (scipy.special.expit(self.time / 3) * 2 - 1) / self.actions_per_night
            if rng.random() < prob:
                self.spawn_enemies(1, self.get_enemy_power(self.time))

        # energy reward
        partial_reward += np.log(self.resources[RESOURCE_ENERGY])
        # time reward
        partial_reward += self.time

        return self.get_observation(), partial_reward, False, False, self.info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        act_type, data = parse_action(action)
        if act_type == ACTIONTYPE_MOVE:
            offset = position_offsets[data]
            target_square = step(self.land.shape, self.player_position, offset)
            move_cost = 0 if offset == (0,0) else land_info[self.land[target_square]].move_cost
            if self.resources[RESOURCE_ENERGY] < move_cost:
                return self._fail()
            self.player_position = target_square
            self.resources[RESOURCE_ENERGY] -= move_cost
            victories, survived = self._check_enemy_attacks(allow_move=False)
            if not survived:
                return self._death()
            return self._environment_turn(victories * self.fight_reward)
        elif act_type == ACTIONTYPE_INTERACT:
            offset = position_offsets[data]
            target_square = step(self.land.shape, self.player_position, offset)
            success, new_r, new_s = land_info[self.land[target_square]].harvest.craft(self.resources, self.player_skills)
            if not success:
                return self._fail()
            self.resources = new_r
            self.player_skills = new_s
            return self._environment_turn(self.harvest_reward)
        elif act_type == ACTIONTYPE_CRAFT:
            recipe = ORDERED_RECIPES[data]
            success, new_r, new_s = recipe.craft(self.resources, self.player_skills)
            if not success:
                return self._fail()
            self.resources = new_r
            self.player_skills = new_s
            return self._environment_turn(self.craft_reward)
        else:
            raise ValueError(f"Unable to process the given action as any type (action {action})")

    def render(self):
        # May or may not get implemented
        pass

    def close(self):
        pass

# TESTING
if __name__ == "__main__":
    TwiLand(generate_map((50,50)))
    print(Observation.configured_size())