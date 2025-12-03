# agents/aggregator_agent.py
import os
import random
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from security.secure_channel import SecureChannel

# -------------------------
# small MLP actor / critic
# -------------------------
class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
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


class AggregatorAgentDDPG:
    """
    Aggregator (DDPG-like). Lazy network build: it waits until it sees a real
    consumer-state concatenation and then builds actor/critic with matching input.
    """

    def __init__(self, region_id: int, consumers: List, key_path: str = "security/keys/secret.key", debug: bool = True):
        self.id = region_id
        self.consumers = consumers  # list of consumer agent objects
        self.device = torch.device("cpu")
        self.debug = debug

        self.state_dim = None   # will set when first real state is seen
        self.action_dim = len(consumers) * 2  # 2D signal per consumer

        # optimizer params
        self.gamma = 0.99
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.batch_size = 64
        self.memory = deque(maxlen=100_000)

        # nets will be created lazily
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.optim_actor = None
        self.optim_critic = None

        # secure channel (reads secret key)
        self.secure = SecureChannel(key_path=key_path, debug=self.debug)

        # book-keeping
        self.last_state = None
        self.last_actions = None

    def _build_networks(self, state_dim: int):
        if self.actor is not None and self.state_dim == state_dim:
            return
        self.state_dim = state_dim
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        if self.debug:
            print(f"[Aggregator {self.id}] Built networks state_dim={self.state_dim} action_dim={self.action_dim}")

    def reset_episode(self):
        self.last_state = None
        self.last_actions = None

    def get_observation(self, consumer_state_list):
        """Concatenate consumer states to create aggregator state."""
        obs = np.concatenate(consumer_state_list).astype(np.float32)
        return obs

    def select_action(self, state_vec: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        state_vec = np.asarray(state_vec, dtype=np.float32)
        if self.actor is None or self.state_dim != state_vec.shape[0]:
            self._build_networks(state_dim=state_vec.shape[0])

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(s).detach().cpu().tolist()[0]
        if noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        self.last_state = state_vec
        self.last_actions = action
        if self.debug:
            print(f"[Aggregator {self.id}] select_action -> {action.tolist()}")
        return action

    def encrypt_signals_for_consumers(self, raw_action_10dim: np.ndarray):
        """
        Splits the action vector into 2D signals (per consumer),
        encrypts each using SecureChannel.encrypt and returns dict consumer_id -> packet(bytes).
        """
        packets = {}
        for i, c in enumerate(self.consumers):
            start = i * 2
            vec = raw_action_10dim[start:start + 2]
            vec = np.asarray(vec, dtype=np.float32)
            packet = self.secure.encrypt(vec)
            packets[c.id] = packet
            if self.debug:
                import binascii
                print(f"[Aggregator {self.id}] Encrypted for Consumer {c.id}: nonce+ct_len={len(packet)} vec={vec.tolist()}")
        return packets

    def store_experience(self, s, a, next_c_actions, r, s2, done):
        self.memory.append((s, a, next_c_actions, r, s2, done))

    def learn(self, all_next_consumer_actions_batch=None):
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

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            q_target = r + self.gamma * q2 * (1 - done)

        q_pred = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_pred, q_target)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        a_pred = self.actor(s)
        actor_loss = -self.critic(s, a_pred).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        return actor_loss.item(), critic_loss.item()

    def get_reward(self, regional_demands):
        if not regional_demands:
            return 0.0
        arr = np.asarray(regional_demands, dtype=np.float32)
        return -float(np.abs(arr[0]))

    def save_models(self, path, episode):
        os.makedirs(path, exist_ok=True)
        if self.actor is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"agg_actor_{self.id}_ep{episode}.pth"))
        if self.critic is not None:
            torch.save(self.critic.state_dict(), os.path.join(path, f"agg_critic_{self.id}_ep{episode}.pth"))

    def load_models(self, path, episode):
        try:
            a_path = os.path.join(path, f"agg_actor_{self.id}_ep{episode}.pth")
            c_path = os.path.join(path, f"agg_critic_{self.id}_ep{episode}.pth")
            # lazy build if necessary
            if self.state_dim is None:
                self._build_networks(state_dim=90)
            if os.path.exists(a_path):
                self.actor.load_state_dict(torch.load(a_path, map_location="cpu"))
            if os.path.exists(c_path):
                self.critic.load_state_dict(torch.load(c_path, map_location="cpu"))
            if self.debug:
                print(f"[Aggregator {self.id}] Loaded models from episode {episode} (if present).")
        except Exception as e:
            print(f"[Aggregator {self.id}] load_models failed: {e}")
