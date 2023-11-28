import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import os
from tqdm import tqdm

#from actions import ACTIONSET_MOVE, ACTIONTYPE_MOVE, ACTIONTYPE_INTERACT, ACTIONTYPE_TRAIN, SKILL_CHOPPING, SKILL_COMBAT, SKILL_CRAFTING, SKILL_FISHING, SKILL_MINING, SKILLSET, parse_action, position_offsets
from twiland import VIEW_DISTANCE, DEFAULT_MAP_SIZE, Observation, TwiLand, generate_map

class AgentNet(nn.Module):
    # I just kinda chose most of the numbers, we can play around with node amounts and add or remove layers
    def __init__(self):
        super(AgentNet, self).__init__()
        # Input size based on number of outputs from Observation
        # This can be obtained from Observation.configured_size()
        self.fc1 = nn.Linear(Observation.configured_size(), 32)
        self.batch_norm1 = nn.BatchNorm1d(32, affine=False, track_running_stats=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 20)
        self.batch_norm2 = nn.BatchNorm1d(20, affine=False, track_running_stats=False)
        self.relu2 = nn.ReLU()
        # Output size based on number of actions, couldn't find constant of how many actions exist, probably don't need one, just update if more actions
        self.fc3 = nn.Linear(20, 11)


    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x.view(x.size(0), -1))
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        print(x, "x shape")
        x = self.fc2(x)
        x = self.batch_norm2(x.view(x.size(0), -1))
        x = self.relu2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x
    
class Agent:
    def __init__(self, generate_new: bool = True, game: TwiLand | None = None, map_size: int | None = None):
        self.net = AgentNet()
        self.generate_new = generate_new
        self.replay_buffer = []
        # Sets the map size
        if map_size is None:
            self.map_size = DEFAULT_MAP_SIZE
        else:
            self.map_size = map_size

        # Sets the starting game
        if game is not None:
            self.env = game
        else:
            self.env = TwiLand(generate_map((self.map_size, self.map_size)))


    def store_experience(self, experience):
        self.replay_buffer.append(experience)

    def learn(self, lr: float = 0.0001, epochs: int = 100, replay_buffer_size: int = 1000, batch_size: int = 32):
        optimizer = optim.Adam(self.net.parameters(), lr=lr) # learning rate schedule?
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs)): 
            total_reward = 0

            # generates a new map if necessary
            if self.generate_new:
                self.env.set_map(generate_map((self.map_size, self.map_size)))

            game_finished = False
            observation, _ = self.env.reset()
            while not game_finished:
                action = self.select_action(observation, exploration_rate=0.1)

                next_observation, reward, terminal, truncated, info = self.env.step(action)
                game_finished = terminal or truncated

                total_reward += reward

                #store experiences in the replay buffer
                self.store_experience((observation, action, reward, terminal, truncated, info))

                #sample a batch from the replay buffer
                if len(self.replay_buffer) >= replay_buffer_size:
                    batch = random.sample(self.replay_buffer, batch_size)

                    #extract components of the batch
                    observations, actions, rewards, terminals, _, _ = zip(*batch)

                    #convert observations to tensors
                    flattened_data = [obs.flattened_data for obs in observations]
                    obs_tensor = torch.tensor(flattened_data, dtype=torch.float32)
                    next_obs_tensor = torch.tensor(next_observation.flattened_data, dtype=torch.float32)

                    #compute Q-values for current and next states
                    q_values_current = self.net(obs_tensor)
                    q_values_next = self.net(next_obs_tensor)

                    #compute target Q-values
                    target_q_values = q_values_current.clone().detach()
                    for i in range(batch_size):
                        target = rewards[i]
                        if not terminals[i]:
                            target += 0.9 * torch.max(q_values_next[i])
                        target_q_values[i][actions[i]] = target

                    loss = criterion(q_values_current, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=1.0)
                    optimizer.step()

                observation = next_observation


    def select_action(self, observation: Observation, exploration_rate: float = 0.1):
        flattened_data = torch.tensor(observation.flattened_data, dtype=torch.float32)
        action_logits = self.net(flattened_data)
        
        # Check for invalid values in action_logits
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print("Invalid values detected in action_logits.")
            return 0  # Or handle this case appropriately

        # Apply softmax to get probabilities
        action_prob = torch.exp(torch.log_softmax(action_logits, dim=-1) + 1e-8)
        
        # Check for invalid values in action_prob
        if torch.isnan(action_prob).any() or torch.isinf(action_prob).any():
            print("Invalid values detected in action_prob.")
            return 0  # Or handle this case appropriately

        if random.random() < exploration_rate:
            action = random.randint(0, 10)
        else:
            print(action_prob)
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
            observation, reward, game_finished, _, _  = game.step(action)
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
num_episodes = [100, 1000, 5000]
max_steps = 1000
scores = {} # list containing scores from each episode

for model in num_episodes:
    model_scores = []
    agent = Agent(map_size=50)
    agent.learn(epochs=model)
    for episode in range(100):
        episode_reward = agent.play(agent.env)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")
        model_scores.append(episode_reward) 
    scores[model] = model_scores
    agent.save_model(f"{model}it.pt")

# Plot scores obtained per episode
for model, scores in scores.items():
    plt.figure()
    plt.title(f"Model Trained for {model} Episodes")
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()