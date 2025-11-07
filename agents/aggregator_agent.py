import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Q-Network for Aggregator Agent
# -------------------------------
class AggregatorQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 128)):
        super(AggregatorQNetwork, self).__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Aggregator Agent (DQN-based)
# -------------------------------
class AggregatorAgentDQN:
    def __init__(self, region_id, consumers=None, aggregator_type="regional",
                 device=None, replay_capacity=50000, lr=1e-4, gamma=0.95,
                 batch_size=64, target_update_freq=500, epsilon_decay=0.999):

        self.id = region_id
        self.type = aggregator_type
        self.consumers = consumers if consumers is not None else []

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # State (7 features: total_demand, solar, soc, emergencies, trust, hour, price)
        self.observation_dim = 7
        # Actions: (price_signal, incentive_signal)
        self.action_space = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.5]], dtype=np.float32)
        self.action_dim = len(self.action_space)

        # DQN setup
        self.q_network = AggregatorQNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network = AggregatorQNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_capacity)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.train_steps = 0
        self.target_update_freq = target_update_freq

        # Bookkeeping
        self.total_demand = 0.0
        self.trust_scores = np.ones(len(self.consumers), dtype=np.float32)
        self.last_signal = np.array([0.1, 0.05], dtype=np.float32)
        self.recent_rewards = deque(maxlen=50)
        self.history = deque(maxlen=100)
        self.last_state = None
        self.last_action_index = 0

    def __repr__(self):
        return f"AggregatorAgentDQN(id={self.id}, consumers={len(self.consumers)}, last_signal={self.last_signal})"

    # -------------------------------
    # Observation Construction
    # -------------------------------
    def get_observation(self, regional_observations, regional_obs_names):
        num_consumers = len(self.consumers)
        if num_consumers == 0:
            return np.zeros(self.observation_dim, dtype=np.float32)

        regional_info = np.zeros(self.observation_dim, dtype=np.float32)
        emergency_count = 0

        for i, consumer in enumerate(self.consumers):
            obs_dict = dict(zip(regional_obs_names[i], regional_observations[i]))
            regional_info[0] += obs_dict.get("non_shiftable_load", 0.0)
            regional_info[1] += obs_dict.get("solar_generation", 0.0)
            regional_info[2] += obs_dict.get("electrical_storage_soc", 0.0)
            if getattr(consumer, "emergency_flag", False):
                emergency_count += 1

        shared_dict = dict(zip(regional_obs_names[0], regional_observations[0]))
        regional_info[3] = emergency_count
        regional_info[4] = np.mean(self.trust_scores) if len(self.trust_scores) else 1.0
        regional_info[5] = shared_dict.get("hour", 0.0)
        regional_info[6] = shared_dict.get("electricity_pricing", 0.0)
        regional_info[2] /= num_consumers

        self.total_demand = regional_info[0]
        self.history.append(self.total_demand)
        return regional_info

    # -------------------------------
    # Action Selection (DQN policy)
    # -------------------------------
    def select_action(self, obs):
        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_index = int(torch.argmax(q_values, dim=1).item())

        action = self.action_space[action_index]
        self.last_state = obs
        self.last_action_index = action_index
        self.last_signal = action.astype(np.float32)
        return self.last_signal, action_index

    # -------------------------------
    # Store Experience
    # -------------------------------
    def store_experience(self, state, action_index, reward, next_state, done):
        self.replay_buffer.append((
            np.array(state, dtype=np.float32),
            int(action_index),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    # -------------------------------
    # Learning Step
    # -------------------------------
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()
        return loss.item()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # -------------------------------
    # Trust and Reward Dynamics
    # -------------------------------
    def update_trust(self, consumer_id, participated):
        if 0 <= consumer_id < len(self.trust_scores):
            delta = 0.05 if participated else -0.1
            self.trust_scores[consumer_id] = np.clip(self.trust_scores[consumer_id] + delta, 0.0, 1.0)

    def get_reward(self, regional_demands):
        if not regional_demands:
            return 0.0

        arr = np.asarray(regional_demands, dtype=np.float32)
        avg_demand = np.mean(arr)
        if avg_demand == 0:
            return 0.0

        # Reward inversely proportional to demand imbalance
        reward = - (np.max(arr) / avg_demand)
        reward += 0.1 * (np.mean(self.trust_scores) - 0.8)
        self.recent_rewards.append(reward)
        return reward

    def reset_episode(self):
        self.last_signal = np.array([0.1, 0.05], dtype=np.float32)
        self.recent_rewards.clear()
        self.history.clear()
        self.epsilon = 1.0
