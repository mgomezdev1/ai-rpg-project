import os
from matplotlib.image import AxesImage
import numpy as np
import scipy
from termcolor import colored
import gymnasium
import random
from typing import Literal, TypeAlias, TypeVar
from matplotlib import pyplot as plt
from matplotlib import colors

from actions import ACTIONSET_MOVE, ACTIONTYPE_MOVE, ACTIONTYPE_INTERACT, ACTIONTYPE_TRAIN, SKILL_CHOPPING, SKILL_COMBAT, SKILL_CRAFTING, SKILL_FISHING, SKILL_MINING, SKILLSET, parse_action, position_offsets
import rendering

TAU = np.pi * 2

LAND_PLAINS = 0
LAND_FOREST = 1
LAND_MOUNTAIN = 2
LAND_WATER = 3

RESOURCE_ENERGY = 0
RESOURCE_WOOD = 1
RESOURCE_ORE = 2
RESOURCE_FRUIT = 3
RESOURCE_RAW_FISH = 4
RESOURCE_COOKED_FISH = 5
RESOURCESET = {RESOURCE_ENERGY, RESOURCE_WOOD, RESOURCE_ORE, RESOURCE_FRUIT, RESOURCE_RAW_FISH, RESOURCE_COOKED_FISH}

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

def dict_to_array(dictionary: dict[int, float], size: int):
    result = np.zeros(size)
    for k,v in dictionary.items(): result[k] = v
    return result

class LootTable:
    def __init__(self, base: dict[int, float], skill_resources: dict[int, float], skill_contribution: dict[int, float], skill_improvement: dict[int, float]):
        self.base = dict_to_array(base, len(RESOURCESET))
        self.skill_resources = dict_to_array(skill_resources, len(RESOURCESET))
        self.skill_contribution = dict_to_array(skill_contribution, len(SKILLSET))
        self.skill_improvement = dict_to_array(skill_improvement, len(SKILLSET))
    def roll(self, resources: np.ndarray[float], skills: np.ndarray[float]) -> tuple[bool, np.ndarray[float], np.ndarray[float]]:
        skill_mult = sum(level * contribution for level,contribution in zip(skills, self.skill_contribution))
        final_reward = self.base + self.skill_resources * skill_mult
        # Negative reward = cost, if the player doesn't have enough. The action fails.
        if any(-reward > current for reward,current in zip(final_reward, resources)): return (False, self.base * 0, self.skill_improvement * 0)

class LandType:
    def __init__(self, name: str, id: int, representation: str, color: str, move_cost: float, loot_table: LootTable):
        self.name = name
        self.id = id
        self.representation = representation
        self.color = color
        self.move_cost = move_cost,
        self.loot_table = loot_table

land_info = [
    LandType("Plains", LAND_PLAINS, colored("P", "black", "on_light_green"), "#44ff55", 0.1,
        LootTable(
            {RESOURCE_ENERGY: 0},{},{},{}
        )
    ),
    LandType("Forest", LAND_FOREST, colored("F", "white", "on_green"), '#00bb00', 0.15,
        LootTable(
            {RESOURCE_ENERGY: -2, RESOURCE_WOOD: 1, RESOURCE_FRUIT: 1.5},
            {RESOURCE_WOOD: 0.25, RESOURCE_FRUIT: 0.1}, # Every skill point grants an extra 0.25 wood and 0.1 fruit
            {SKILL_CHOPPING: 1},   # Action affected by the chopping skill
            {SKILL_CHOPPING: 0.25} # Level up 25% of chopping per chopping
        )
    ),
    LandType("Mountain", LAND_MOUNTAIN, colored("M", "white", "on_grey"), "#333355", 0.7,
        LootTable(
            {RESOURCE_ENERGY: 5, RESOURCE_ORE: 1},
            {RESOURCE_ORE: 0.4},
            {SKILL_MINING: 1, SKILL_CRAFTING: 0.1},
            {SKILL_MINING: 0.25}
        )
    ),
    LandType("Water", LAND_WATER, colored("W", "white", "on_blue"), '#0000ff', 0.45,
        LootTable(
            {RESOURCE_ENERGY: -3, RESOURCE_RAW_FISH: 3},
            {RESOURCE_RAW_FISH: 0.5},
            {SKILL_FISHING: 1},
            {SKILL_FISHING: 0.25}
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
    result = np.ones(size) * LAND_PLAINS
    
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
    radius = sx // 2
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

class Observation:
    def __init__(self, neighbours: np.ndarray[int], enemies: np.ndarray[int], time: float, skills: np.ndarray[float], resources: np.ndarray[int], view_mask: np.ndarray[bool]):
        self.flat_neighbors = np.extract(view_mask, neighbours)
        self.flat_enemies   = np.extract(view_mask, enemies)
        self.sigmoid_time   = scipy.special.expit(np.array([time]) / 20) * 2 - 1
        self.sigmoid_skills = scipy.special.expit(skills / 5) * 2 - 1
        self.sigmoid_resources = scipy.special.expit(resources / 5) * 2 - 1
        self.flattened_data = np.concatenate([self.flat_neighbors, self.flat_enemies, self.sigmoid_time, self.sigmoid_skills], axis=0, dtype=np.float32)

    def configured_size() -> int:
        return TwiLand(generate_map((3 * VIEW_DISTANCE, 3 * VIEW_DISTANCE))).get_observation().flattened_data.shape

class TwiLand(gymnasium.Env):
    def __init__(self, land: np.ndarray[int], player_position: tuple[int,int] | None = None, enable_rendering = True):
        self.land = land
        if not player_position:
            player_position = random_pos(land.shape)
        self.player_position = player_position
        self.enemies : list[Enemy] = []

        self.player_skills = np.ones(len(SKILLSET))
        self.resources = np.ones(len(RESOURCESET))
        self.time = 0
        self.map_img_path = "./temp/img/twiland_map.png"
        self.enable_rendering = enable_rendering
        if enable_rendering:
            rendering.enable_rendering()
            self.save_map()

    def spawn_enemies(self, count: int = 1, power: float = 1):
        for i in range(count):
            angle = rng.random() * np.pi * 2
            distance = rng.randint(*ENEMY_SPAWN_RANGE)
            offset = (int(distance * np.sin(angle)) + int(distance * np.cos(angle)))
            position = step(self.land.shape, self.player_position, offset)
            self.enemies.append(Enemy(position, power))

    def set_map(self, land: np.ndarray[int]):
        self.land = land
    def save_map(self):
        cmap = colors.ListedColormap([land_type.color for land_type in land_info])
        print([land_type.color for land_type in land_info])
        print(self.land)

        dir = os.path.dirname(self.map_img_path)
        if (not os.path.exists(dir)): os.makedirs(dir)
        plt.imsave(self.map_img_path, self.land, cmap=cmap)

    def get_observation(self):
        land_submatrix = crop_map_submatrix(self.land, self.player_position, VIEW_DISTANCE)
        mask = get_mask((2 * VIEW_DISTANCE + 1, VIEW_NORM))
        enemy_matrix   = np.zeros(self.land.shape)
        for e in self.enemies:
            enemy_matrix[e.position] = 1
        enemy_submatrix = crop_map_submatrix(enemy_matrix, self.player_position, VIEW_DISTANCE)

        return Observation(land_submatrix, enemy_submatrix, self.time, self.player_skills, mask)

    def reset(self) -> tuple[Observation, dict]:
        self.player_position = random_pos(self.land.shape)
        self.time = 0
        self.player_skills = np.ones(len(SKILLSET))
        self.resources = np.ones(len(RESOURCESET))
        self.enemies = []

        return (self.get_observation(), {})

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        act_type, data = parse_action(action)
        if act_type == ACTIONTYPE_MOVE:
            offset = position_offsets[data]

    def render(self):
        # May or may not get implemented
        pass

    def close(self):
        pass

# TESTING
if __name__ == "__main__":
    TwiLand(generate_map((50,50)))