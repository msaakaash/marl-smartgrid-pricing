# agents/aggregator_agent.py

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from security.secure_channel import SecureChannel


# ============================================================================
#                           AGGREGATOR NETWORKS
# ============================================================================

class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
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
    def __init__(self, state_dim: int, action_dim: int):
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
                 key_path: str = "security/keys/secret.key"):

        self.id = region_id
        self.consumers = consumers
        self.device = torch.device("cpu")

        # Dimensions (state_dim will be inferred lazily)
        self.state_dim = None
        self.action_dim = 10  # 10D action, split into 5 pairs

        # Learning params
        self.gamma = 0.98
        self.lr_actor = 1e-3
        self.lr_critic = 2e-3
        self.batch_size = 64
        self.memory = deque(maxlen=100000)

        # Networks (lazy init once we see a real state)
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.optim_actor = None
        self.optim_critic = None

        # Security Layer
        self.secure = SecureChannel(key_path=key_path)

        # Track last state/action
        self.last_state = None
        self.last_actions = None

    # ========================================================================
    # lazy network construction
    # ========================================================================
    def _build_networks(self, state_dim: int):
        """Build actor/critic and their targets with the given state_dim."""
        if self.state_dim is not None and self.state_dim == state_dim and self.actor is not None:
            return  # already built

        self.state_dim = state_dim

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        print(f"[Aggregator {self.id}] Built networks with state_dim={self.state_dim}, action_dim={self.action_dim}")

    # ========================================================================
    def reset_episode(self):
        self.last_state = None
        self.last_actions = None

    # ========================================================================
    def get_observation(self, consumer_state_list):
        """Each consumer sends a state vector → concatenated into a regional state."""
        obs = np.concatenate(consumer_state_list)
        return obs.astype(np.float32)

    # ========================================================================
    def select_action(self, state_vec):
        """Deterministic actor output."""
        state_vec = np.asarray(state_vec, dtype=np.float32)
        # Build networks lazily to match actual state dimension
        if self.actor is None or self.state_dim != state_vec.shape[0]:
            self._build_networks(state_dim=state_vec.shape[0])

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(s).detach().cpu().tolist()[0]
        return action

    # ========================================================================
    def store_experience(self, s, a, next_c_actions, r, s2, done):
        """Standard DDPG experience."""
        self.memory.append((s, a, next_c_actions, r, s2, done))

    # ========================================================================
    def learn(self, all_next_consumer_actions_batch=None):
        """DDPG update."""
        if self.actor is None or self.critic is None:
            return None, None

        if len(self.memory) < self.batch_size:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        s, a, next_c_actions, r, s2, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a), dtype=torch.float32, device=self.device)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(done), dtype=torch.float32, device=self.device).unsqueeze(1)

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

    # ========================================================================
    def get_reward(self, regional_demand):
        """Simple reward: penalize magnitude of regional demand."""
        return -abs(regional_demand[0])

    # ========================================================================
    def encrypt_signals_for_consumers(self, raw_action_10dim):
        """Splits 10D action into pairs → encrypt each → return dict."""
        encrypted_packets = {}

        for i, c in enumerate(self.consumers):
            vec = raw_action_10dim[i*2: i*2 + 2]
            vec = np.array(vec, dtype=np.float32)

            # Use dict-based API (consumer can handle dicts via _process_agg_signal)
            packet = self.secure.encrypt(vec)
            encrypted_packets[c.id] = packet

        return encrypted_packets

    # ========================================================================
    def save_models(self, path, episode):
        if self.actor is not None:
            torch.save(self.actor.state_dict(), f"{path}/agg_actor_{self.id}_ep{episode}.pth")
        if self.critic is not None:
            torch.save(self.critic.state_dict(), f"{path}/agg_critic_{self.id}_ep{episode}.pth")

    def load_models(self, path, episode):
        try:
            actor_path = f"{path}/agg_actor_{self.id}_ep{episode}.pth"
            critic_path = f"{path}/agg_critic_{self.id}_ep{episode}.pth"

            # We need networks built before loading; use a dummy dim if unknown
            if self.state_dim is None:
                # This will be corrected later when real state appears,
                # but allows loading shapes from disk.
                self._build_networks(state_dim=90)

            self.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
            self.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
            print(f"Loaded Aggregator {self.id} models for episode {episode}")
        except Exception as e:
            print(f"Failed loading Aggregator {self.id} episode {episode}: {e}")
