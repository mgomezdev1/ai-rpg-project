from matplotlib.image import AxesImage
import numpy as np
from termcolor import colored
import random
from typing import TypeVar
from matplotlib import pyplot as plt
from matplotlib import colors

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

SKILL_CHOPPING = 0
SKILL_MINING = 1
SKILL_FISHING = 2
SKILL_CRAFTING = 3
SKILL_COMBAT = 4
SKILLSET = {SKILL_CHOPPING, SKILL_MINING, SKILL_FISHING, SKILL_CRAFTING, SKILL_COMBAT}

skill_training_efficiency = {
    SKILL_CHOPPING: 1,
    SKILL_MINING: 1,
    SKILL_FISHING: 1,
    SKILL_CRAFTING: 0.5,
    SKILL_COMBAT: 0
}

ENEMY_SPAWN_RANGE = (3, 6)

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
def draw_world(land: np.ndarray[int], player: tuple[int,int], enemies: list[tuple[int,int]], show = True) -> AxesImage:
    cmap = colors.ListedColormap([land_type.color for land_type in land_info] + [PLAYER_COLOR, ENEMY_COLOR])
    playerFlag = len(land_info)
    enemyFlag = player + 1
    world = land.copy()
    world[player] = playerFlag
    for enemy in enemies: world[enemy] = enemyFlag
    img = plt.imshow(land, cmap=cmap)
    if show: plt.show()
    return img

class Enemy:
    def __init__(self, position: tuple[int,int], power: float = 1):
        self.position = position
        self.power = power

class TwiLand:
    def __init__(self, land: np.ndarray[int], player_position: tuple[int,int] | None = None):
        self.land = land
        if not player_position:
            player_position = random_pos(land.shape)
        self.player_position = player_position
        self.enemies : list[Enemy] = []

    def spawn_enemies(self, count: int = 1, power: float = 1):
        for i in range(count):
            angle = rng.random() * np.pi * 2
            distance = rng.randint(*ENEMY_SPAWN_RANGE)
            offset = (int(distance * np.sin(angle)) + int(distance * np.cos(angle)))
            position = step(self.land.shape, self.player_position, offset)
            self.enemies.append(Enemy(position, power))

    def take_action(self, action: int):
        pass