import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

# Define the structure for a single transition (experience)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """A cyclic buffer of fixed size to store transitions observed by the agent."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    The Deep Q-Network model.
    It's a simple feed-forward neural network that takes a state tensor
    and outputs Q-values for each possible action.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """
    The agent that interacts with and learns from the environment.
    """
    def __init__(self, state_dim, action_dim, capacity=10000, batch_size=128, gamma=0.99, lr=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the policy network and the target network
        # The policy_net is the one we actively train.
        # The target_net is used to stabilize training. Its weights are copied from the policy_net periodically.
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(capacity)

        self.steps_done = 0
        # Epsilon-greedy strategy parameters
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        # Calculate current epsilon
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > epsilon_threshold:
            # Exploitation: choose the best action from the policy network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            # Exploration: choose a random action
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of the optimization (on the policy network)."""
        if len(self.memory) < self.batch_size:
            return  # Not enough memories to train yet

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                                           for s in batch.next_state if s is not None])

        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat([torch.tensor([r], device=self.device) for r in batch.reward])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values (the target)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """Update the target network's weights with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="dqn_snake_model.pth"):
        """Saves the policy network's state dictionary."""
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="dqn_snake_model.pth"):
        """Loads a pre-trained policy network's state dictionary."""
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.update_target_net()
            print(f"Successfully loaded model from {path}")
        except FileNotFoundError:
            print(f"No model found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")


if __name__ == '__main__':
    # Example usage:
    # These dimensions would be determined by your state representation
    STATE_DIMENSION = 11  # e.g., the size of the feature vector from the original AI
    ACTION_DIMENSION = 3  # e.g., turn left, go straight, turn right

    agent = DQNAgent(STATE_DIMENSION, ACTION_DIMENSION)
    print("DQN Agent created successfully.")
    print("Device:", agent.device)
    print("Policy Network:", agent.policy_net)
