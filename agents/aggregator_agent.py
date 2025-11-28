import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
from security.secure_channel import SecureChannel    # <-- ADDED

# --- Define the Actor Network (UNCHANGED) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
    def forward(self, state):
        return self.net(state)

# --- MADDPG Critic ---
class Critic(nn.Module):
    def __init__(self, state_dim, agg_action_dim, consumer_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = state_dim + agg_action_dim + consumer_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, agg_action, consumer_actions):
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
        
        self.consumer_obs_dim = 18
        self.consumer_action_dim = 1
        self.observation_dim = self.num_consumers * self.consumer_obs_dim
        self.action_dim = self.num_consumers * 2
        self.total_consumer_action_dim = self.num_consumers * self.consumer_action_dim
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.actor = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(
            self.observation_dim, 
            self.action_dim, 
            self.total_consumer_action_dim
        ).to(self.device)

        self.critic_target = Critic(
            self.observation_dim, 
            self.action_dim, 
            self.total_consumer_action_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.replay_buffer = deque(maxlen=replay_capacity)

        self.total_demand = 0.0
        self.trust_scores = np.ones(self.num_consumers, dtype=np.float32)
        self.last_state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards = deque(maxlen=50)

        # ---------- ADDED: ENCRYPTION CHANNEL ----------
        try:
            self.secure_channel = SecureChannel()
        except Exception as e:
            self.secure_channel = None
            print(f"[Aggregator {self.id}] Warning: SecureChannel init failed: {e}. Encryption disabled.")

    def __repr__(self):
        return f"AggregatorAgentDDPG(id={self.id}, consumers={self.num_consumers})"

    # --- get_observation ---
    def get_observation(self, consumer_observations):
        if not consumer_observations:
            return np.zeros(self.observation_dim, dtype=np.float32)
        try:
            state = np.concatenate(consumer_observations).astype(np.float32)
            if state.shape[0] != self.observation_dim:
                correct = np.zeros(self.observation_dim, dtype=np.float32)
                l = min(state.shape[0], self.observation_dim)
                correct[:l] = state[:l]
                state = correct
        except ValueError:
            state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_state = state
        return state

    # --- select_action ---
    def select_action(self, obs, noise_std=0.1):
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        noise = np.random.normal(0, noise_std, size=self.action_dim)
        action = np.clip(action + noise, 0.0, 1.0)
        self.last_actions = action.astype(np.float32)
        return self.last_actions

    # -------------- ADDED: ENCRYPTED SIGNAL OUTPUT --------------
    def get_encrypted_price_packet(self, obs):
        """
        Outputs encrypted aggregator action vector.
        Returned value: bytes (nonce + ciphertext) or plaintext fallback.
        """
        action = self.select_action(obs, noise_std=0.0)
        vec = [float(x) for x in action]

        if self.secure_channel is None:
            # Fallback plaintext mode
            plain = ",".join(map(str, vec)).encode("utf-8")
            return b"PLAINTEXT:" + plain

        try:
            packet = self.secure_channel.encrypt_vector(vec)
            return packet
        except Exception as e:
            print(f"[Aggregator {self.id}] Encryption error: {e}")
            plain = ",".join(map(str, vec)).encode("utf-8")
            return b"PLAINTEXT:" + plain

    # --- store_experience ---
    def store_experience(self, state, action, consumer_actions, reward, next_state, done):
        self.replay_buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(consumer_actions, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    # --- soft update ---
    def _soft_update_target_networks(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(t.data * (1. - self.tau) + s.data * self.tau)
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(t.data * (1. - self.tau) + s.data * self.tau)

    # --- learn ---
    def learn(self, all_next_consumer_actions_batch=None):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, cons_actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        cons_actions = torch.tensor(np.array(cons_actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        if all_next_consumer_actions_batch is None:
            all_next_consumer_actions_batch = cons_actions

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions, all_next_consumer_actions_batch)
            target_q = rewards + (1. - dones) * self.gamma * next_q
        
        current_q = self.critic(states, actions, cons_actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions, cons_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item()

    # --- trust ---
    def update_trust(self, consumer_id, participated):
        if 0 <= consumer_id < len(self.trust_scores):
            delta = 0.05 if participated else -0.1
            self.trust_scores[consumer_id] = np.clip(self.trust_scores[consumer_id] + delta, 0.0, 1.0)

    # --- reward ---
    def get_reward(self, regional_demands):
        if not regional_demands: return 0.0
        arr = np.asarray(regional_demands, dtype=np.float32)
        avg = np.mean(arr)
        if avg == 0: return 0.0
        reward = - (np.max(arr) / avg)
        reward += 0.1 * (np.mean(self.trust_scores) - 0.8)
        self.recent_rewards.append(reward)
        return reward

    # --- reset ---
    def reset_episode(self):
        self.last_state = np.zeros(self.observation_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards.clear()

    # --- save ---
    def save_models(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"agg_{self.id}_actor_ep{episode}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"agg_{self.id}_critic_ep{episode}.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(directory, f"agg_{self.id}_actor_target_ep{episode}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(directory, f"agg_{self.id}_critic_target_ep{episode}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(directory, f"agg_{self.id}_actor_optim_ep{episode}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(directory, f"agg_{self.id}_critic_optim_ep{episode}.pth"))

    # --- load ---
    def load_models(self, directory, episode):
        try:
            self.actor.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_actor_ep{episode}.pth")))
            self.critic.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_critic_ep{episode}.pth")))
            self.actor_target.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_actor_target_ep{episode}.pth")))
            self.critic_target.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_critic_target_ep{episode}.pth")))
            self.actor_optimizer.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_actor_optim_ep{episode}.pth")))
            self.critic_optimizer.load_state_dict(torch.load(os.path.join(directory, f"agg_{self.id}_critic_optim_ep{episode}.pth")))
            self.actor.train()
            self.critic.train()
            self.actor_target.train()
            self.critic_target.train()
            print(f"Loaded models for Aggregator {self.id} from episode {episode}")
        except FileNotFoundError:
            print(f"No checkpoint found for Aggregator {self.id} at episode {episode}, starting fresh.")
        except Exception as e:
            print(f"Error loading models for Aggregator {self.id}: {e}")
