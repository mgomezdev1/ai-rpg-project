import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from actions import ACTIONSET_ALL
import rendering
from time import time
from collections import defaultdict

times = defaultdict(float)

#from actions import ACTIONSET_MOVE, ACTIONTYPE_MOVE, ACTIONTYPE_INTERACT, ACTIONTYPE_TRAIN, SKILL_CHOPPING, SKILL_COMBAT, SKILL_CRAFTING, SKILL_FISHING, SKILL_MINING, SKILLSET, parse_action, position_offsets
from twiland import VIEW_DISTANCE, DEFAULT_MAP_SIZE, Observation, TwiLand, generate_map

class AgentNet(nn.Module):
    # I just kinda chose most of the numbers, we can play around with node amounts and add or remove layers
    def __init__(self):
        super(AgentNet, self).__init__()
        # Input size based on number of outputs from Observation
        # This can be obtained from Observation.configured_size()
        self.fc1 = nn.Linear(Observation.configured_size(), 500)
        self.batch_norm1 = nn.BatchNorm1d(500, affine=False, track_running_stats=False)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(500, 300)
        self.batch_norm2 = nn.BatchNorm1d(300, affine=False, track_running_stats=False)
        self.act2 = nn.ReLU()
        # Output size based on number of actions
        self.fc3 = nn.Linear(300, len(ACTIONSET_ALL))
        self.act3 = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        #x = self.batch_norm1(x.view(x.size(0), -1))
        #x = x.view(x.size(0), -1)
        x = self.act1(x)
        #print(len(x), "x shape")
        x = self.fc2(x)
        #x = self.batch_norm2(x.view(x.size(0), -1))
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        #x = self.fc4(x)
        #x = F.softmax(x, dim=-1)
        return x
    
class Agent:
    def __init__(self, generate_new: bool = True, game: TwiLand | None = None, map_size: int | None = None, enable_rendering: bool = False, memory_size: int | None = None):
        self.net = AgentNet()
        self.targetnet = AgentNet()
        self.generate_new = generate_new
        self.replay_buffer = []
        self.energy = 100
        self.learning_update_interval = 100
        # Sets the map size
        if map_size is None:
            self.map_size = DEFAULT_MAP_SIZE
        else:
            self.map_size = map_size

        # Sets the starting game
        if game is not None:
            self.env = game
        else:
            self.env = TwiLand(generate_map((self.map_size, self.map_size)), enable_rendering=enable_rendering, starting_energy=self.energy, 
                fail_reward=-15, time_reward_factor=0, energy_reward_factor=0.25, successes_reward_factor=5, energy_gain_reward=10)

        self.memory_size = memory_size if memory_size is not None else 50000

    def store_experience(self, experience):
        if (len(self.replay_buffer) >= self.memory_size):
            self.replay_buffer.pop(int(random.random() * self.memory_size))
        self.replay_buffer.append(experience)

    def learn(self, lr: float = 0.1, epochs: int = 100, batch_size: int = 32):
        optimizer = optim.Adam(self.net.parameters(), lr=lr) # learning rate schedule?
        criterion = nn.MSELoss()
        energy_decay = np.log(40 / self.energy) / epochs
        epsilon = 0.5
        epsilon_min = 0.01
        epsilon_decay = 0.995
        learning_cooldown = self.learning_update_interval
        running_rewards = []

        for epoch in tqdm(range(epochs)): 
            total_reward = 0

            t0 = time()
            # generates a new map if necessary
            if self.generate_new:
                # Do not generate a new environment per epoch, this is very costly and breaks things like rendering, simply reset the variables you need to reset
                #self.env = TwiLand(generate_map((self.map_size, self.map_size)), enable_rendering=False, starting_energy=int(self.energy * np.exp(energy_decay * epoch)))
                                   #, idle_cost=np.round(0.1*(1/epochs)*epoch, 4))
                self.env.set_map(generate_map((self.map_size, self.map_size)))
            
            # Disable energy decay
            # self.env.starting_energy = int(self.energy * np.exp(energy_decay * epoch))

            print("Starting Energy: ", self.env.starting_energy)

            game_finished = False
            observation, _ = self.env.reset()
            times["Map Resetting"] += time() - t0
            while not game_finished:
                t1 = time()
                action = self.select_action(observation, exploration_rate=max(epsilon_min, min(epsilon, 1.0 - np.log10((epoch + 1) * epsilon_decay))))
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
                    rewards = torch.tensor(actions)
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
                        target_q_values[range(batch_size), actions] = rewards + max_rewards * continues_mask
                    t8 = time()
                    loss = criterion(q_values_current, target_q_values)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
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
                if self.env.rendering_enabled: 
                    rendering.basic_event_loop()
                    rendering.set_title_text(f"Episode {epoch}; Score = {total_reward:.1f}; Day {self.env.time:.0f}")
                    rendering.update_display(self.env)
                times["Rendering"] += time() - t5

                observation = next_observation
                
            running_rewards.append(total_reward)
            print(len(self.replay_buffer))
            print(f"\n {times}")
            print(f"REWARD: {total_reward}")



    def select_action(self, observation: Observation, exploration_rate: float = 0.1):
        flattened_data = torch.tensor(observation.flattened_data, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.net(flattened_data)
            
            # Check for invalid values in action_logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print("Invalid values detected in action_logits.")
                return -1  # Or handle this case appropriately

            # Apply softmax to get probabilities
            action_prob = torch.exp(torch.log_softmax(action_logits, dim=-1) + 1e-8)
            
            # Check for invalid values in action_prob
            if torch.isnan(action_prob).any() or torch.isinf(action_prob).any():
                print("Invalid values detected in action_prob.")
                return -1  # Or handle this case appropriately

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
            # or pick the action with the highest probability
            # action = torch.argmax(action_prob).item()

            # take the decided action
            observation, reward, ter, tru, _  = game.step(action)
            game_finished = ter or tru
            total_reward += reward
        
        return total_reward
    
    def save_model(self, filename: str):
        os.makedirs("models", exist_ok=True)
        file_path = os.path.join("models", filename)

        state = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in self.net.state_dict().items()}

        with open(file_path, "wb") as f:
            torch.save(state, f)

    def load_model(self, filename: str):
        file_path = os.path.join("models", filename)
        with open(file_path, "r") as f:
            state = torch.load(f)
            self.net.load_state_dict(state)

#train the agent
num_episodes = [100,500,1000,5000]
max_steps = 1000
scores = {} # list containing scores from each episode

for model in num_episodes:
    model_scores = []
    agent = Agent(map_size=50, generate_new=True, memory_size=1000, enable_rendering=True)
    agent.learn(epochs=model)

    # Generate a new game to test on
    # This is done on agent creation!!!
    # agent.env = TwiLand(generate_map((agent.map_size, agent.map_size)), enable_rendering=False, starting_energy=50)
    #agent.env.starting_energy = 50
    for episode in range(100):
        agent.env.reset()
        episode_reward = agent.play(agent.env) + 1000
        # Add 1000 to reward to account for death
        if not np.isinf(episode_reward):
            print(f"Episode {episode + 1}/{100}, Total Reward: {episode_reward}")
            model_scores.append(episode_reward)
    print(f"Mean Score: {np.mean(model_scores)}")
    scores[model] = model_scores
    agent.save_model(f"{model}it.pt")

# Plot scores obtained per episode
for model, scores in scores.items():
    plt.figure()
    plt.title(f"Model Trained for {model} Episodes")
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()