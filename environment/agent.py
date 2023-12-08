import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import json
import os
from numba import cuda
import torch.cuda
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from actions import ACTIONSET_ALL
import rendering
from time import time
from collections import defaultdict

times = defaultdict(float)

from twiland import VIEW_DISTANCE, DEFAULT_MAP_SIZE, Observation, TwiLand, generate_map

class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        # Input size based on number of outputs from Observation
        # This can be obtained from Observation.configured_size()
        self.fc1 = nn.Linear(Observation.configured_size(), 1000)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(1000, 800)
        self.act2 = nn.ReLU()
        # Output size based on number of actions
        self.fc3 = nn.Linear(800, 500)
        self.act3 = nn.Sigmoid()
        self.fco = nn.Linear(500, len(ACTIONSET_ALL))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        # We want to get values that can be highly negative or positive (Q-values)! Not just [0,1]
        x = self.fco(x)
        return x
    
class Agent:
    def __init__(self, generate_new: bool = True, game: TwiLand | None = None, map_size: int | None = None, enable_rendering: bool = False, memory_size: int | None = None, energy_train: int = 500, energy_target: int = 50):
        self.net = AgentNet()
        self.targetnet = AgentNet()
        self.generate_new = generate_new
        self.replay_buffer = []
        self.energy = energy_train
        self.energy_target = energy_target
        self.learning_update_interval = 500

        # Sets the starting game
        if game is not None:
            self.env = game
            self.map_size = game.land.shape[0]
        else:
            # Sets the map size
            if map_size is None:
                self.map_size = DEFAULT_MAP_SIZE
            else:
                self.map_size = map_size
            self.env = TwiLand(generate_map((self.map_size, self.map_size)), enable_rendering=enable_rendering, starting_energy=self.energy, 
                fail_reward=0, time_reward_factor=1.5, energy_reward_factor=0, successes_reward_factor=1.5, energy_gain_reward=50, death_reward=-100000,
                max_days=30, enemy_spawning=False)
        
        self.memory_size = memory_size if memory_size is not None else 50000

    def store_experience(self, experience):
        if (len(self.replay_buffer) >= self.memory_size):
            self.replay_buffer.pop(int(random.random() * self.memory_size))
        self.replay_buffer.append(experience)

    def learn(self, lr: float = 0.01, epochs: int = 100, batch_size: int = 32) -> list[dict]:
        optimizer = optim.Adam(self.net.parameters(), lr=lr) # learning rate schedule?
        criterion = nn.MSELoss()
        #energy_decay = np.log(40 / self.energy) / epochs
        decay_values = np.linspace(self.energy, self.energy_target, num=epochs).astype(int)
        records = []
        # set exploration rate parameters
        epsilon = 1
        epsilon_min = 0.01
        epsilon_decay = 0.99
        GAMMA = 0.995
        learning_cooldown = self.learning_update_interval
        running_rewards = []

        for epoch in range(epochs): 
            total_reward = 0

            t0 = time()
            # generates a new map if necessary
            if self.generate_new:
                self.env.set_map(generate_map((self.map_size, self.map_size)))
            
            # Disable energy decay
            # self.env.starting_energy = int(self.energy * np.exp(energy_decay * epoch))
            self.env.starting_energy = decay_values[epoch]

            print("Starting Energy: ", self.env.starting_energy)

            # implemented epsilon decay on episode reset directly
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            game_finished = False
            observation, _ = self.env.reset()
            times["Map Resetting"] += time() - t0
            total_actions = 0
            training_loss = []
            while not game_finished:
                t1 = time()
                # old exploration rate formula: max(epsilon_min, min(epsilon, 1.0 - np.log10((epoch + 1) * epsilon_decay)))
                action = self.select_action(observation, exploration_rate=epsilon)
                t2 = time()
                times["Action Selection"] += t2 - t1
                # Throws away the game if something blows up
                if action == -1:
                    game_finished = True

                next_observation, reward, terminal, truncated, info = self.env.step(action)
                game_finished = terminal or truncated
                t3 = time()
                times["Environment Step"] += t3 - t2

                # Stops the game if the reward is infinite
                if np.isinf(reward):
                    game_finished = True
                else:
                    total_reward += reward
  
                #store experiences in the replay buffer
                self.store_experience((observation, next_observation, action, reward, terminal, truncated, info))

                #sample a batch from the replay buffer
                t4 = time()
                if len(self.replay_buffer) >= batch_size:
                    batch = random.sample(self.replay_buffer, batch_size)

                    #extract components of the batch
                    observations, next_observations, actions, rewards, terminals, truncated, _ = zip(*batch)

                    #convert observations to tensors
                    flattened_data = np.stack([obs.flattened_data for obs in observations])
                    next_flattened_data = np.stack([obs.flattened_data for obs in next_observations])
                    obs_tensor = torch.tensor(flattened_data, dtype=torch.float32)
                    next_obs_tensor = torch.tensor(next_flattened_data, dtype=torch.float32)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    actions = torch.tensor(actions)
                    finalized = torch.tensor([ter or tru for ter,tru in zip(terminals, truncated)])

                    #compute Q-values for current and next states
                    optimizer.zero_grad()   
                    q_values_current = self.net(obs_tensor)

                    #compute target Q-values
                    t7 = time()
                    with torch.no_grad():
                        q_values_next = self.targetnet(next_obs_tensor)
                        target_q_values = q_values_current.clone().detach()
                        max_rewards, _ = torch.max(q_values_next, dim=1)
                        continues_mask = (~finalized).float()
                        target_q_values[range(batch_size), actions] = rewards + max_rewards * continues_mask * GAMMA
                    t8 = time()
                    loss = criterion(q_values_current, target_q_values)
                    # print(loss.item())
                    training_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    t9 = time()
                    times["MaxQ"] += t8 - t7
                    times["Optimization"] += t9 - t8
                    learning_cooldown -= 1
                    # Every C iterations, we set our action-performing network to take the best actions we have predicted
                    # This improves stability
                    t6 = time()
                    if learning_cooldown <= 0:
                        learning_cooldown = self.learning_update_interval
                        self.targetnet.load_state_dict(self.net.state_dict())
                    times["Learning Net Update"] += time() - t6
                t5 = time()
                times["Learning"] += t5 - t4
                total_actions += 1
                if self.env.rendering_enabled: 
                    rendering.basic_event_loop()
                    rendering.set_title_text(f"Episode {epoch}; Score = {total_reward:.1f}; Day {self.env.time:.0f}")
                    rendering.update_display(self.env)
                    rendering.set_info_text(f"Successes: {self.env.successful_actions:.0f}/{total_actions} ({self.env.successful_actions/total_actions:.1%})")
                times["Rendering"] += time() - t5

                observation = next_observation
            
            # When the game ends:
            running_rewards.append(total_reward)
            records.append({
                "reward": total_reward,
                "accuracy": self.env.successful_actions/total_actions, 
                "time_survived": self.env.time,
                "epsilon": epsilon,
                "mean_loss": np.mean(training_loss)
            })
            #print(len(self.replay_buffer))
            #print(f"\n {times}")
            print(f"REWARD: {total_reward}")
        return records



    def select_action(self, observation: Observation, exploration_rate: float = 0.1):
        flattened_data = torch.tensor(observation.flattened_data, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.net(flattened_data)
            
            # Check for invalid values in action_logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print("Invalid values detected in action_logits.")
                return -1  

            # Apply softmax to get probabilities
            action_prob = torch.exp(torch.log_softmax(action_logits, dim=-1) + 1e-8)
            
            # Check for invalid values in action_prob
            if torch.isnan(action_prob).any() or torch.isinf(action_prob).any():
                print("Invalid values detected in action_prob.")
                return -1 

            if random.random() < exploration_rate:
                action = random.randrange(0, len(ACTIONSET_ALL))
            else:
                action = torch.multinomial(action_prob, 1).item()

            return action

    def play(self, game: TwiLand):
        # Plays the game with the given gameboard to termination
        total_reward = 0

        game_finished = False
        observation = game.get_observation()
        while not game_finished:
            # forward pass for action probabilities
            # sample an action from the probability distribution
            action = self.select_action(observation)

            # take the decided action
            observation, reward, ter, tru, _  = game.step(action)
            game_finished = ter or tru
            total_reward += reward
        
        return total_reward

    def save_model(self, filename: str):
        os.makedirs("models", exist_ok=True)
        file_path = os.path.join("models", filename)

        #state = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in self.net.state_dict().items()}
        state = self.net.state_dict()

        with open(file_path, "wb") as f:
            torch.save(state, f)

    def load_model(self, filename: str):
        file_path = os.path.join("models", filename)
        print(f"Loading model from {file_path}")
        state = torch.load(file_path)
        self.net.load_state_dict(state)

def random_play(game: TwiLand) -> float:
    game_finished = False
    total_reward = 0
    while not game_finished:
        action = random.randrange(len(ACTIONSET_ALL))
        obs, reward, ter, tru, _ = game.step(action)
        game_finished = ter or tru
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.backends.cudnn.enabled)
    #train the agent
    models = [("Quicktest", 10, 0.05), ("Fast", 100, 0.01), ("Long", 500, 0.0025)]
    # testing loss
    # models = [("Losstest", 25, 0.01)]
    max_steps = 1000
    scores = {} # list containing scores from each episode

    ENABLE_RENDERING = False
    MAP_SIZE = 16
    ENERGY = 100

    game = TwiLand(generate_map((MAP_SIZE, MAP_SIZE)), enable_rendering=ENABLE_RENDERING, starting_energy=ENERGY, 
        fail_reward=-20, time_reward_factor=0.5, energy_reward_factor=0, successes_reward_factor=1.5, energy_gain_reward=50, death_reward=-10000,
        max_days=40, enemy_spawning=True)

    for name, episodes, lr in models:
        model_scores = []
        random_scores = []
        agent = Agent(game=game, generate_new=True, memory_size=5000, enable_rendering=ENABLE_RENDERING, energy_train=ENERGY, energy_target=ENERGY)

        # Train the agent and save training data
        records = agent.learn(epochs=episodes, lr=lr)
        with open(f"results/data/records_{name}.json", "w") as f:
            json.dump(records, f, indent = 4)

        # Set the starting energy for testing
        agent.env.starting_energy = ENERGY
        agent.net.eval()

        for episode in range(100):
            # Generate a new map to test on
            agent.env.set_map(generate_map((agent.map_size, agent.map_size)))
            agent.env.reset()
            starting_pos = agent.env.player_position
            episode_reward = agent.play(agent.env) - agent.env.death_reward
            if not np.isinf(episode_reward):
                print(f"Episode {episode + 1}/{100}, Total Reward: {episode_reward}")
                model_scores.append(episode_reward)
            agent.env.reset()
            agent.env.player_position = starting_pos
            random_reward = random_play(agent.env) - agent.env.death_reward
            random_scores.append(random_reward)
        print(f"Mean Score: {np.mean(model_scores)}")
        scores[name] = model_scores
        
        # Save all relevant testing data
        fig, ax = plt.subplots()
        ax.set_title(f"Results for {name} Model")
        ax.plot(np.arange(len(model_scores)), model_scores)
        ax.plot(np.arange(len(random_scores)), random_scores)
        ax.plot(np.arange(len(model_scores)), [m - r for m,r in zip(model_scores,random_scores)])
        ax.set_ylabel('Score')
        ax.set_xlabel('Episode #')
        ax.legend(["model", "random", "difference"])
        fig.savefig(f"figs/model_{name}_results")

        agent.save_model(f"model_{name}.pt")