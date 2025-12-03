# agents/consumer_agent.py
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from security.secure_channel import SecureChannel


# ================================================================
#   Q-Network
# ================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=(128, 128)):
        super().__init__()
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


# ================================================================
#          CONSUMER DQN AGENT (with secure decryption)
# ================================================================
class ConsumerAgentDQN:
    def __init__(self, building, building_id, metadata, action_space,
                 key_path="security/keys/secret.key",
                 debug=True):

        self.building = building
        self.id = building_id
        self.metadata = metadata
        self.action_space = action_space
        self.debug = debug

        # obs = 18-dim used in MARL pipeline
        self.observation_dim = 18

        # 3-action discrete control: Charge / Idle / Discharge
        self.action_values = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        self.action_dim = len(self.action_values)

        # RL params
        self.device = torch.device("cpu")
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.replay = deque(maxlen=50_000)

        # Q-networks
        self.q_network = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.target_update_freq = 1000
        self.train_steps = 0

        # SOC & reward history
        self.soc = 0.0
        self.reward_history = []

        # Secure-channel for decrypting aggregator signals
        self.secure = SecureChannel(key_path=key_path, debug=self.debug)

        # runtime
        self.last_state = None
        self.last_action_index = 1  # default = idle

        if self.debug:
            print(f"[Consumer {self.id}] Ready. Actions={self.action_values.tolist()}")

    # ================================================================
    #   DECRYPT aggregator signals
    # ================================================================
    def decrypt_agg_packet(self, packet_bytes: bytes):
        """Decrypt aggregator→consumer encrypted packet."""
        try:
            vec = self.secure.decrypt(packet_bytes)
        except Exception:
            # tampered packet — secure layer returned zero vector
            return np.zeros(2, dtype=np.float32)

        vec = np.asarray(vec, dtype=np.float32)

        if vec.size < 2:
            padded = np.zeros(2, dtype=np.float32)
            padded[:vec.size] = vec
            return padded
        return vec[:2]

    # ================================================================
    #   BUILD 18-dim OBSERVATION
    # ================================================================
    def get_observation(self, raw_obs, observation_names, agg_signal=None):
        obs = np.array(raw_obs, dtype=np.float32)

        # ensure length >= 10
        if obs.size < 10:
            padded = np.zeros(10, dtype=np.float32)
            padded[:obs.size] = obs
            obs = padded

        # aggregator signal
        if agg_signal is None:
            agg_signal = np.zeros(2, dtype=np.float32)
        else:
            agg_signal = np.asarray(agg_signal, dtype=np.float32)
            if agg_signal.size < 2:
                tmp = np.zeros(2, dtype=np.float32)
                tmp[:agg_signal.size] = agg_signal
                agg_signal = tmp

        # build final 18-dim vector
        final_obs = np.zeros(self.observation_dim, dtype=np.float32)

        final_obs[:10] = obs[:10]

        bt_map = {
            "residential": 0, "hospital": 1, "office": 2,
            "mall": 3, "industry": 4, "cheater": 5
        }

        final_obs[10] = bt_map.get(self.metadata.get("building_type", "residential"), 0)
        final_obs[11] = float(self.metadata.get("critical_load_fraction", 0.2))
        final_obs[12] = float(self.metadata.get("cheating_propensity", 0.0))
        final_obs[13] = 1.0 if self.metadata.get("emergency_flag", False) else 0.0
        final_obs[14] = 1.0 if int(obs[2]) in self.metadata.get("prime_hours", []) else 0.0

        final_obs[15] = float(agg_signal[0])
        final_obs[16] = float(agg_signal[1])
        final_obs[17] = 0.0  # padding

        return final_obs

    # ================================================================
    #   ACTION SELECTION
    # ================================================================
    def select_action(self, obs):
        if random.random() < self.epsilon:
            idx = random.randrange(self.action_dim)
        else:
            s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q_network(s)
            idx = int(torch.argmax(qvals, dim=1).item())

        action_value = float(self.action_values[idx])

        # ensure valid range
        try:
            low = float(np.atleast_1d(self.action_space.low)[0])
            high = float(np.atleast_1d(self.action_space.high)[0])
            action_value = float(np.clip(action_value, low, high))
        except Exception:
            pass

        self.last_state = obs
        self.last_action_index = idx

        return np.array([action_value], dtype=np.float32), idx

    # ================================================================
    #   REPLAY BUFFER
    # ================================================================
    def store_experience(self, s, a, r, s2, done):
        self.replay.append(
            (np.array(s, dtype=np.float32),
             int(a),
             float(r),
             np.array(s2, dtype=np.float32),
             float(done))
        )

    # ================================================================
    #   LEARNING
    # ================================================================
    def learn(self):
        if len(self.replay) < self.batch_size:
            return None

        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.long).unsqueeze(1)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)

        q = self.q_network(s).gather(1, a)

        with torch.no_grad():
            q_next = self.target_q_network(s2).max(1, keepdim=True)[0]
            target = r + (1.0 - done) * self.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    # ================================================================
    #   MODEL SAVE/LOAD (fixed)
    # ================================================================
    def save_models(self, path, episode):
        try:
            torch.save(self.q_network.state_dict(),
                       f"{path}/consumer_{self.id}_qnet_ep{episode}.pth")
            torch.save(self.target_q_network.state_dict(),
                       f"{path}/consumer_{self.id}_target_ep{episode}.pth")
        except Exception as e:
            print(f"[Consumer {self.id}] ERROR saving models: {e}")

    def load_models(self, path, episode):
        try:
            q_path = f"{path}/consumer_{self.id}_qnet_ep{episode}.pth"
            target_path = f"{path}/consumer_{self.id}_target_ep{episode}.pth"

            self.q_network.load_state_dict(torch.load(q_path, map_location="cpu"))
            self.target_q_network.load_state_dict(torch.load(target_path, map_location="cpu"))

            print(f"[Consumer {self.id}] Loaded models for episode {episode}")
        except Exception as e:
            print(f"[Consumer {self.id}] ERROR loading models: {e}")

    # ================================================================
    def get_reward(self, raw_reward, current_power_demand,
                   action_value, soc_before, electricity_pricing,
                   agg_signal=None):

        try:
            av = float(np.atleast_1d(action_value)[0])
        except:
            av = float(action_value)

        penalty = 0.0
        if (av < 0 and soc_before <= 0.0) or (av > 0 and soc_before >= 1.0):
            penalty -= 1.0

        final_r = float(raw_reward) + penalty
        self.reward_history.append(final_r)
        return final_r

    # ================================================================
    def reset_episode(self):
        self.last_state = None
        self.last_action_index = 1
        self.reward_history.clear()
