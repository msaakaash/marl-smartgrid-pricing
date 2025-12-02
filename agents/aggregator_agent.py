# aggregator_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from secure_channel import SecureChannel


# ============================================================================
#                           AGGREGATOR NETWORKS
# ============================================================================

class Actor(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()    # output between -1 and 1
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_dim=90, action_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.fc(x)


# ============================================================================
#                            AGGREGATOR AGENT DDPG
# ============================================================================

class AggregatorAgentDDPG:
    """Secure Aggregator Agent"""

    def __init__(self, region_id: int, consumers: list,
                 key_path="security/keys/secret.key"):

        self.id = region_id
        self.consumers = consumers
        self.device = torch.device("cpu")

        # Dimensions
        self.state_dim = 90
        self.action_dim = 10

        # Learning params
        self.gamma = 0.98
        self.lr_actor = 1e-3
        self.lr_critic = 2e-3
        self.batch_size = 64
        self.memory = deque(maxlen=100000)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Security Layer
        self.secure = SecureChannel(key_path)

        # Track last state/action
        self.last_state = None
        self.last_actions = None

    # ============================================================================
    def reset_episode(self):
        pass

    # ============================================================================
    def get_observation(self, consumer_state_list):
        """Each consumer sends a 30-dim state → concatenated into 90 dim."""
        obs = np.concatenate(consumer_state_list)
        return obs.astype(np.float32)

    # ============================================================================
    def select_action(self, state90):
        """Deterministic actor output."""
        s = torch.tensor(state90, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        return action

    # ============================================================================
    def store_experience(self, s, a, next_c_actions, r, s2, done):
        """Standard DDPG experience."""
        self.memory.append((s, a, next_c_actions, r, s2, done))

    # ============================================================================
    def learn(self, all_next_consumer_actions_batch=None):
        """DDPG update."""
        if len(self.memory) < self.batch_size:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        s, a, next_c_actions, r, s2, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(a), dtype=torch.float32).to(self.device)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1).to(self.device)

        # ---------------- Critic update ----------------
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            q_target = r + self.gamma * q2 * (1 - done)

        q_pred = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_pred, q_target)

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # ---------------- Actor update ----------------
        a_pred = self.actor(s)
        actor_loss = -self.critic(s, a_pred).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        return actor_loss.item(), critic_loss.item()

    # ============================================================================
    def get_reward(self, regional_demand):
        return -abs(regional_demand[0])

    # ============================================================================
    def encrypt_signals_for_consumers(self, raw_action_10dim):
        """Splits 10D action into pairs → encrypt each → return dict."""
        encrypted_packets = {}

        for i, c in enumerate(self.consumers):
            vec = raw_action_10dim[i*2: i*2 + 2]
            packet = self.secure.encrypt(np.array(vec, dtype=np.float32))
            encrypted_packets[c.id] = packet

        return encrypted_packets

    # ============================================================================
    def save_models(self, path, episode):
        torch.save(self.actor.state_dict(), f"{path}/agg_actor_{self.id}_ep{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/agg_critic_{self.id}_ep{episode}.pth")

    def load_models(self, path, episode):
        try:
            self.actor.load_state_dict(torch.load(
                f"{path}/agg_actor_{self.id}_ep{episode}.pth", map_location="cpu"
            ))
            self.critic.load_state_dict(torch.load(
                f"{path}/agg_critic_{self.id}_ep{episode}.pth", map_location="cpu"
            ))
            print(f"Loaded Aggregator {self.id} models for episode {episode}")
        except:
            print(f"Failed loading Aggregator {self.id} episode {episode}")
