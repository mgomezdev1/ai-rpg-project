import torch.nn as nn
import torch.optim as optim
import torch

#from actions import ACTIONSET_MOVE, ACTIONTYPE_MOVE, ACTIONTYPE_INTERACT, ACTIONTYPE_TRAIN, SKILL_CHOPPING, SKILL_COMBAT, SKILL_CRAFTING, SKILL_FISHING, SKILL_MINING, SKILLSET, parse_action, position_offsets
from twiland import VIEW_DISTANCE, DEFAULT_MAP_SIZE, TwiLand, generate_map

class AgentNet(nn.Module):
    # I just kinda chose most of the numbers, we can play around with node amounts and add or remove layers
    def __init__(self):
        super(AgentNet, self).__init__()
        # Input size based on number of outputs from Observation, not sure if thats the right amount
        self.fc1 = nn.Linear(2 * VIEW_DISTANCE + 12, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 20)
        self.relu2 = nn.ReLU()
        # Output size based on number of actions, couldn't find constant of how many actions exist, probably don't need one, just update if more actions
        self.fc3 = nn.Linear(20, 11)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class Agent:
    def __init__(self, generate_new: bool = True, game: TwiLand | None = None, map_size: int | None = None):
        self.net = AgentNet()
        self.generate_new = generate_new

        # Sets the map size
        if map_size is None:
            self.map_size = DEFAULT_MAP_SIZE
        else:
            self.map_size = map_size

        # Sets the starting game
        if game is not None:
            self.env = game
        else:
            self.env = TwiLand(generate_map(self.map_size, self.map_size))


    def learn(self, lr: float = 0.01, epochs: int = 100):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs): 
            total_reward = 0

            game_finished = False
            observation = self.env.get_observation()
            while not game_finished:

                # Definitely doesn't work, I need to figure out what the flattened data looks like
                action_prob = torch.softmax(self.net(torch.tensor(observation.flattened_data, dtype=torch.float32)), dim=-1)

                # Sample an action from the probability distribution
                action = torch.multinomial(action_prob, 1).item()

                # Take the decided action
                observation, reward, game_finished, truncated, info  = self.env.step(action)
                total_reward += reward

                # compute the loss
                target = torch.tensor([action], dtype=torch.long)
                loss = criterion(action_prob.view(1, -1), target)
            
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # No back pass, not sure how we want to implement it

            # Resets the map or generates a new map
            if self.generate_new:
                self.env = TwiLand(generate_map(self.map_size, self.map_size))
            else:
                self.env.reset()

    def play(self, game: TwiLand):
        # Plays the game with the given gameboard to termination
        total_reward = 0

        game_finished = False
        observation = game.get_observation()
        while not game_finished:
            # forward pass for action probabilities
            flattened_data = torch.tensor(observation.flattened_data, dtype=torch.float32)
            action_prob = torch.softmax(self.net(flattened_data), dim=-1)

            # sample an action from the probability distribution
            action = torch.multinomial(action_prob, 1).item()
            # or pick the action with the highest probability
            # action = torch.argmax(action_prob).item()

            # take the decided action
            observation, reward, game_finished, truncated, info  = game.step(action)
            total_reward += reward
        
        return total_reward