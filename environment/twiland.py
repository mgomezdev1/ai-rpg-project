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

land_types = [LAND_PLAINS, LAND_FOREST, LAND_MOUNTAIN, LAND_WATER]
land_repr = {
    LAND_PLAINS: colored("P", "black", "on_light_green"),
    LAND_FOREST: colored("F", "white", "on_green"),
    LAND_MOUNTAIN: colored("M", "white", "on_dark_grey"),
    LAND_WATER: colored("W", "white", "on_blue")
}
land_colors = {
    LAND_PLAINS: '#44ff55',
    LAND_FOREST: '#00bb00',
    LAND_MOUNTAIN: '#333355',
    LAND_WATER: '#0000ff'
}

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
            print(land_repr[land[i,j]], end="")
        print()

def draw_map(land: np.ndarray[int], show = True) -> AxesImage:
    cmap = colors.ListedColormap([land_colors[t] for t in land_types])
    img = plt.imshow(land, cmap=cmap)
    if show: plt.show()
    return img
