from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import Agent, random_play
from twiland import TwiLand, generate_map, random_pos

model_names = ["random", "Quicktest", "Fast", "Long"]
models : list[Agent | None] = []
PREFIX = "records_"
SUFFIX = ".json"

MAP_SIZE = 16
ENERGY = 100

NUM_EPISODES = 100
for name in model_names:
    if name == "random": 
        models.append(None)
        continue
    agent = Agent(map_size = MAP_SIZE)
    agent.load_model(f"model_{name}.pt")
    models.append(agent)

results = defaultdict(list)
game = TwiLand(enable_rendering=False, starting_energy=ENERGY, 
    fail_reward=-20, time_reward_factor=0.5, energy_reward_factor=0, successes_reward_factor=1.5, energy_gain_reward=50, death_reward=-10000,
    max_days=40, enemy_spawning=True)

fig, ax = plt.subplots()

for i in tqdm(range(NUM_EPISODES)):
    land = generate_map((MAP_SIZE,MAP_SIZE))
    game.set_map(land)
    pos = random_pos((MAP_SIZE, MAP_SIZE))
    for name, model in zip(model_names, models):
        game.reset()
        game.player_position = pos
        if model == None:
            results[name].append(random_play(game))
        else:
            results[name].append(model.play(game))

# result sorting!
sorted_indices = [i for x,i in sorted((-result,j) for j,result in enumerate(results[model_names[-1]]))]
sorted_results = {}
print(sorted_indices)
for name in model_names:
    sorted_results[name] = [0] * len(sorted_indices)
    for i,j in enumerate(sorted_indices):
        sorted_results[name][i] = results[name][j]

names = []
ax.set_title("Reward of various agents in a set of random episodes")
ax.set_ylabel("Total reward")
ax.set_xlabel("Episode ID")
for name, total_rewards in sorted_results.items():
    names.append(name)
    ax.plot(total_rewards)
ax.legend(names)
fig.show()
input()
