from random import Random
import numpy as np

# SKILL SYSTEM

SKILL_CHOPPING = 0
SKILL_MINING = 1
SKILL_FISHING = 2
SKILL_CRAFTING = 3
SKILL_COMBAT = 4
ORDERED_SKILLSET = [SKILL_CHOPPING, SKILL_MINING, SKILL_FISHING, SKILL_CRAFTING, SKILL_COMBAT]
SKILLSET = set(ORDERED_SKILLSET)

# CRAFTING SYSTEM
RESOURCE_ENERGY = 0
RESOURCE_WOOD = 1
RESOURCE_ORE = 2
RESOURCE_FRUIT = 3
RESOURCE_RAW_FISH = 4
RESOURCE_COOKED_FISH = 5
RESOURCE_AXE = 6
RESOURCE_PICKAXE = 7
RESOURCE_FISHING_ROD = 8
RESOURCE_SWORD = 9
ORDERED_RESOURCESET = [RESOURCE_ENERGY, RESOURCE_WOOD, RESOURCE_ORE, RESOURCE_FRUIT, RESOURCE_RAW_FISH, RESOURCE_COOKED_FISH, RESOURCE_AXE, RESOURCE_PICKAXE, RESOURCE_FISHING_ROD, RESOURCE_SWORD]
RESOURCESET = set(ORDERED_RESOURCESET)

def soft_float_to_int(value: np.ndarray[float] | float) -> np.ndarray[int] | int:
    base = np.floor(value)
    base += np.random.binomial(1, value - base)
    return base

class Recipe: 
    def __init__(self, costs: dict[int,float], 
            products: dict[int,float] = {}, 
            skill_weights: dict[int,float] = {}, 
            bonus_products: dict[int,float] = {}, 
            skill_improvements: dict[int, float] = {},
            max_products: dict[int, int] = {}):
        '''
            The costs represent the amount of items required (and consumed) when using the recipe. Decimal values represent a chance of consuming an additional item.
            The products represent the base amount of items produced when the recipe is completed.
            Bonus products are awarded equal to bonus_products multiplied by a "skill bonus", which is calculated as the dot product of skill_weights and player skills.
            Skill improvements indicate the number of skill levels improved by using the recipe. Decimal values represent a chance to obtain an extra level.
        '''
        self.costs = np.zeros(len(RESOURCESET))
        self.products = np.zeros(len(RESOURCESET))
        self.skill_weights = np.zeros(len(SKILLSET))
        self.bonus_products = np.zeros(len(RESOURCESET))
        self.skill_improvements = np.zeros(len(SKILLSET))
        self.max_products = np.ones(len(RESOURCESET)) * 1000000
        for i,n in costs.items():
            self.costs[i] = n
        for i,n in products.items():
            self.products[i] = n
        for i,n in bonus_products.items():
            self.bonus_products[i] = n
        for i,n in skill_weights.items():
            self.skill_weights[i] = n
        for i,n in skill_improvements.items():
            self.skill_improvements[i] = n
        for i,n in max_products.items():
            self.max_products[i] = n

    def craft(self, resources: np.ndarray[int], skills: np.ndarray[int]) -> tuple[bool, np.ndarray[int], np.ndarray[int]]:
        '''
        Attempts to craft the recipe given a current set of resources and skills.
        Returns whether the craft was successful and a new set of resources and potentially leveled skills.
        '''
        if np.any(resources < self.costs): return False, resources, skills
        
        bonus_weight = np.dot(self.skill_weights, skills)
        new_products = np.minimum(self.max_products, self.products + bonus_weight * self.bonus_products)
        new_resources = soft_float_to_int(resources - self.costs + new_products)
        new_skills = soft_float_to_int(skills + self.skill_improvements)
        return True, new_resources, new_skills

class RecipeStack(Recipe):
    def __init__(self, *recipes: Recipe):
        self.recipes = recipes
    def craft(self, resources: np.ndarray[int], skills: np.ndarray[int]) -> tuple[bool, np.ndarray[int], np.ndarray[int]]:
        for r in self.recipes:
            success, final_r, final_s = r.craft(resources, skills)
            if success:
                return True, final_r, final_s
        return False, resources, skills

def dict_to_array(dictionary: dict[int, float], size: int):
    result = np.zeros(size)
    for k,v in dictionary.items(): result[k] = v
    return result
