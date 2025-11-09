import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# --- Define the Actor Network (UNCHANGED) ---
# It still only sees its own state (90-dim)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output actions between 0.0 and 1.0
        )
        
    def forward(self, state):
        return self.net(state)

# --- MADDPG CHANGE: Critic Network (UPGRADED) ---
# It now sees its state, its action, AND all the consumers' actions
class Critic(nn.Module):
    def __init__(self, state_dim, agg_action_dim, consumer_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # --- MADDPG CHANGE: Input dimension is now larger ---
        # state_dim (90) + agg_action_dim (10) + consumer_action_dim (5)
        input_dim = state_dim + agg_action_dim + consumer_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single Q-value
        )

    def forward(self, state, agg_action, consumer_actions):
        # --- MADDPG CHANGE: Concatenate all inputs ---
        x = torch.cat([state, agg_action, consumer_actions], dim=1)
        return self.net(x)

# --- The MADDPG Aggregator Agent ---
class AggregatorAgentDDPG:
    def __init__(self, region_id, consumers=None, 
                 device=None, replay_capacity=100000, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, batch_size=64, tau=0.005):
        
        self.id = region_id
        self.consumers = consumers if consumers is not None else []
        self.num_consumers = len(self.consumers)
        
        # --- State and Action Dimensions ---
        self.consumer_obs_dim = 18
        self.consumer_action_dim = 1 # Each consumer takes 1 float action
        self.observation_dim = self.num_consumers * self.consumer_obs_dim  # 90
        self.action_dim = self.num_consumers * 2                          # 10
        
        # --- MADDPG CHANGE: Add total consumer action dim ---
        self.total_consumer_action_dim = self.num_consumers * self.consumer_action_dim # 5
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # --- Create Actor (UNCHANGED) ---
        self.actor = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # --- MADDPG CHANGE: Create Critic (UPGRADED) ---
        self.critic = Critic(
            self.observation_dim, 
            self.action_dim, 
            self.total_consumer_action_dim  # Pass in new dim
        ).to(self.device)
        self.critic_target = Critic(
            self.observation_dim, 
            self.action_dim, 
            self.total_consumer_action_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- Hyperparameters (UNCHANGED) ---
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.replay_buffer = deque(maxlen=replay_capacity)

        # --- Bookkeeping (UNCHANGED) ---
        self.total_demand = 0.0
        self.trust_scores = np.ones(self.num_consumers, dtype=np.float32)
        self.last_state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards = deque(maxlen=50)

    def __repr__(self):
        return f"AggregatorAgentDDPG(id={self.id}, consumers={self.num_consumers})"

    # --- get_observation (UNCHANGED) ---
    def get_observation(self, consumer_observations):
        if not consumer_observations:
            return np.zeros(self.observation_dim, dtype=np.float32)
        try:
            state = np.concatenate(consumer_observations).astype(np.float32)
            if state.shape[0] != self.observation_dim:
                correct_state = np.zeros(self.observation_dim, dtype=np.float32)
                l = min(state.shape[0], self.observation_dim)
                correct_state[:l] = state[:l]
                state = correct_state
        except ValueError:
            state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_state = state
        return state

    # --- select_action (UNCHANGED) ---
    # Actor is decentralized, so it only sees its own state
    def select_action(self, obs, noise_std=0.1):
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        noise = np.random.normal(0, noise_std, size=self.action_dim)
        action = np.clip(action + noise, 0.0, 1.0)
        self.last_actions = action.astype(np.float32)
        return self.last_actions

    # --- MADDPG CHANGE: Store Experience (UPGRADED) ---
    def store_experience(self, state, action, consumer_actions, reward, next_state, done):
        self.replay_buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(consumer_actions, dtype=np.float32), # <-- NEW: Store consumer actions
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    # --- _soft_update_target_networks (UNCHANGED) ---
    def _soft_update_target_networks(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    # --- MADDPG CHANGE: Learning Step (UPGRADED) ---
    def learn(self, all_next_consumer_actions_batch=None):
        if len(self.replay_buffer) < self.batch_size:
            return None, None # Return no loss

        # --- MADDPG CHANGE: Sample from buffer (includes consumer actions) ---
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, consumer_actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        consumer_actions = torch.tensor(np.array(consumer_actions), dtype=torch.float32, device=self.device) # <-- NEW
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        # If main.py didn't provide next consumer actions, use a simple approximation
        if all_next_consumer_actions_batch is None:
            all_next_consumer_actions_batch = consumer_actions # Use current actions as fallback

        # --- Update Critic ---
        with torch.no_grad():
            # Get next aggregator actions from target actor
            next_actions = self.actor_target(next_states)
            # --- MADDPG CHANGE: Critic target now sees all actions ---
            next_q = self.critic_target(next_states, next_actions, all_next_consumer_actions_batch)
            # Compute the target Q-value
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        
        # --- MADDPG CHANGE: Critic now sees all actions ---
        current_q = self.critic(states, actions, consumer_actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Get actions from the main actor
        actor_actions = self.actor(states)
        
        # --- MADDPG CHANGE: Actor loss uses the centralized critic ---
        # We want to maximize the Q-value, so we take the negative mean.
        # The critic is judging the actor's actions *given what the consumers did*
        actor_loss = -self.critic(states, actor_actions, consumer_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item()

    # -------------------------------
    # Trust and Reward Dynamics (UNCHANGED)
    # -------------------------------
    def update_trust(self, consumer_id, participated):
        if 0 <= consumer_id < len(self.trust_scores):
            delta = 0.05 if participated else -0.1
            self.trust_scores[consumer_id] = np.clip(self.trust_scores[consumer_id] + delta, 0.0, 1.0)

    # --- get_reward (UNCHANGED) ---
    def get_reward(self, regional_demands):
        if not regional_demands: return 0.0
        arr = np.asarray(regional_demands, dtype=np.float32)
        avg_demand = np.mean(arr)
        if avg_demand == 0: return 0.0
        reward = - (np.max(arr) / avg_demand)
        reward += 0.1 * (np.mean(self.trust_scores) - 0.8)
        self.recent_rewards.append(reward)
        return reward

    # --- reset_episode (UNCHANGED) ---
    def reset_episode(self):
        self.last_state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards.clear()