# consumer_agent.py
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import the secure channel implementation (expects security/secure_channel.py)
try:
    from security.secure_channel import SecureChannel
except Exception as e:
    # Best-effort fallback if import path differs
    raise ImportError("Failed to import SecureChannel from security/secure_channel.py: " + str(e))


# ---------------------------
# Q-network (simple MLP)
# ---------------------------
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


# ---------------------------
# ConsumerAgentDQN (secure-aware)
# ---------------------------
class ConsumerAgentDQN:
    """
    Consumer agent using DQN. Supports receiving agg signals as:
      - plaintext numpy array (shape (2,))
      - 'packet' bytes produced by SecureChannel.encrypt_vector (nonce + ciphertext)
      - 'PLAINTEXT:' prefixed bytes (fallback)
      - dict with keys 'nonce'/'ciphertext' (legacy)
    Decryption/verification logs are printed during runtime (secure_channel debug also prints).
    """

    def __init__(
        self,
        building,
        building_id: int,
        metadata: dict,
        action_space,
        key_path: str = "security/keys/shared_key.bin",
        device: str = "cpu",
        replay_capacity: int = 50000,
        lr: float = 5e-4,
        gamma: float = 0.95,
        batch_size: int = 64,
        target_update_freq: int = 1000,
    ):
        self.building = building
        self.id = building_id
        self.metadata = metadata or {}
        self.action_space = action_space

        # Agent meta
        self.type = self.metadata.get("building_type", "residential")
        self.critical_load_fraction = float(self.metadata.get("critical_load_fraction", 0.2))
        self.prime_hours = self.metadata.get("prime_hours", [])
        self.cheating_propensity = float(self.metadata.get("cheating_propensity", 0.0))
        self.emergency_flag = bool(self.metadata.get("emergency_flag", False))

        # Action discretization (keeps compatibility with earlier code)
        self.discrete_actions = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        self.action_dim = len(self.discrete_actions)

        # Observation/network will be created lazily to match environment obs length
        self.input_dim = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # Training hyperparams
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.lr = lr
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.train_steps = 0
        self.target_update_freq = target_update_freq

        # bookkeeping
        self.last_action = 0.0
        self.last_action_index = 1
        self.last_state = None
        self.soc = 0.0
        self.reward_history = []
        self.energy_history = []
        self.satisfaction_history = []

        # Secure channel (ChaCha20-Poly1305)
        # SecureChannel.encrypt_vector(packet) -> bytes (nonce + ciphertext)
        # SecureChannel.decrypt_vector(packet) -> np.array(dtype=float32)
        try:
            self.secure = SecureChannel(key_path=key_path, debug=True)
        except TypeError:
            # Older secure_channel signatures may not accept debug kwarg
            self.secure = SecureChannel(key_path=key_path)
        except Exception as e:
            print(f"[Consumer {self.id}] Warning: SecureChannel init failed: {e}. Continuing without secure channel.")
            self.secure = None

    # ---------------------------
    # lazy network creation
    # ---------------------------
    def _build_networks(self, input_dim: int):
        """Create policy & target nets given runtime observation length."""
        self.input_dim = input_dim
        self.policy_net = QNetwork(input_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(input_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        print(f"[Consumer {self.id}] Built networks with input_dim={input_dim}, action_dim={self.action_dim}")

    # ---------------------------
    # decrypt helper
    # ---------------------------
    def _process_agg_signal(self, agg_signal):
        """
        Accepts several formats for incoming aggregator signal:
         - None -> zeros
         - numpy array/list -> converted and returned
         - bytes -> passed to secure.decrypt_vector (or treated as PLAINTEXT:)
         - dict with nonce/ciphertext -> try decrypt that too
        Returns numpy array dtype float32
        """
        if agg_signal is None:
            return np.zeros(2, dtype=np.float32)

        # Already numerical array/list
        if isinstance(agg_signal, (np.ndarray, list, tuple)):
            arr = np.asarray(agg_signal, dtype=np.float32)
            if arr.size == 0:
                return np.zeros(2, dtype=np.float32)
            # ensure length 2 (truncate/pad)
            arr = arr.astype(np.float32)
            if arr.size >= 2:
                return arr[:2].astype(np.float32)
            else:
                out = np.zeros(2, dtype=np.float32)
                out[: arr.size] = arr
                return out

        # dict-like with nonce/ciphertext (some aggregator implementations use dict)
        if isinstance(agg_signal, dict):
            # try to convert to bytes packet if possible (nonce + ciphertext)
            if "nonce" in agg_signal and "ciphertext" in agg_signal:
                try:
                    packet = bytes(agg_signal["nonce"]) + bytes(agg_signal["ciphertext"])
                    agg_signal = packet
                except Exception:
                    # fallback: return zeros
                    print(f"[Consumer {self.id}] Received dict agg_signal but failed to pack bytes -> safe zeros")
                    return np.zeros(2, dtype=np.float32)

        # bytes-like packet — attempt decryption using secure channel if available
        if isinstance(agg_signal, (bytes, bytearray)):
            if self.secure is None:
                # If no secure channel, attempt plaintext decode
                try:
                    raw = agg_signal
                    if raw.startswith(b"PLAINTEXT:"):
                        raw = raw.replace(b"PLAINTEXT:", b"")
                    vec = np.array(list(map(float, raw.decode().split(","))), dtype=np.float32)
                    print(f"[Consumer {self.id}] Received PLAINTEXT packet -> {vec}")
                    return vec[:2] if vec.size >= 2 else np.pad(vec, (0, 2 - vec.size))
                except Exception:
                    return np.zeros(2, dtype=np.float32)

            # use secure.decrypt_vector if available
            try:
                # secure.decrypt_vector expects packet bytes (nonce + ciphertext) or PLAINTEXT fallback.
                vec = self.secure.decrypt_vector(bytes(agg_signal))
                # secure_channel prints debug info; add a concise one
                print(f"[Consumer {self.id}] Decrypted agg vector -> {vec}")
                # Ensure shape (2,)
                if vec.size >= 2:
                    return vec[:2].astype(np.float32)
                else:
                    out = np.zeros(2, dtype=np.float32)
                    out[: vec.size] = vec
                    return out
            except Exception as e:
                print(f"[Consumer {self.id}] Decryption error: {e} — returning safe zeros")
                return np.zeros(2, dtype=np.float32)

        # Unknown type
        print(f"[Consumer {self.id}] Unknown agg_signal type ({type(agg_signal)}). Using zeros.")
        return np.zeros(2, dtype=np.float32)

    # ---------------------------
    # public: get_observation
    # ---------------------------
    def get_observation(self, raw_obs, observation_names=None, agg_signal=None):
        """
        Build observation vector for DQN. raw_obs is environment observation (list/array).
        agg_signal may be encrypted bytes, dict, or plain array.
        """
        raw_arr = np.array(raw_obs, dtype=np.float32)
        decrypted = self._process_agg_signal(agg_signal)
        obs = np.concatenate([raw_arr, decrypted]).astype(np.float32)

        # Build networks lazily if not yet built or input size changed
        if self.policy_net is None or self.input_dim != obs.shape[0]:
            self._build_networks(input_dim=obs.shape[0])

        return obs

    # ---------------------------
    # action selection (epsilon-greedy)
    # ---------------------------
    def select_action(self, obs):
        # obs is numpy array
        if self.policy_net is None:
            # Defensive fallback (shouldn't happen)
            self._build_networks(input_dim=obs.shape[0])

        # Emergency: hold state
        if self.emergency_flag:
            action_index = 1
            action_value = float(self.discrete_actions[action_index])
            self.last_action = action_value
            self.last_action_index = action_index
            self.last_state = obs
            return np.array([action_value], dtype=np.float32), action_index

        # Cheating stochastic behavior (keeps compatibility)
        if random.random() < self.cheating_propensity:
            action_index = self.action_dim - 1  # cheat (charge)
            self.trust_score = max(0.0, getattr(self, "trust_score", 1.0) - 0.1)
            action_value = float(self.discrete_actions[action_index])
            action_value = float(np.clip(action_value, self.action_space.low, self.action_space.high))
            self.last_action = action_value
            self.last_action_index = int(action_index)
            self.last_state = obs
            return np.array([action_value], dtype=np.float32), action_index
        else:
            # small trust repair
            if hasattr(self, "trust_score"):
                self.trust_score = min(1.0, self.trust_score + 0.01)

        # epsilon-greedy
        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_dim)
        else:
            s_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                qvals = self.policy_net(s_t)
            action_index = int(torch.argmax(qvals, dim=1).item())

        action_value = float(self.discrete_actions[action_index])
        # clip to env action_space if available
        try:
            low = float(self.action_space.low) if np.isscalar(self.action_space.low) else float(np.atleast_1d(self.action_space.low)[0])
            high = float(self.action_space.high) if np.isscalar(self.action_space.high) else float(np.atleast_1d(self.action_space.high)[0])
            action_value = float(np.clip(action_value, low, high))
        except Exception:
            pass

        self.last_action = action_value
        self.last_action_index = int(action_index)
        self.last_state = obs

        return np.array([action_value], dtype=np.float32), action_index

    # ---------------------------
    # experience + learn
    # ---------------------------
    def store_experience(self, state, action_index, reward, next_state, done):
        self.replay_buffer.append((
            np.array(state, dtype=np.float32),
            int(action_index),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

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

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        if self.target_net is not None and self.policy_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------------------------
    # reward (kept simple)
    # ---------------------------
    def get_reward(self, raw_reward, current_power_demand, action_value, soc_before_action, electricity_pricing, agg_signal=None):
        try:
            action_scalar = float(np.asarray(action_value).item())
        except Exception:
            action_scalar = float(action_value)

        weights = {'env': 1.0, 'constraint': 50.0, 'emergency': 500.0, 'cheating': -20.0, 'shifting': 0.5, 'agg_follow': 5.0}

        reward_env = float(raw_reward)
        reward_constraint = 0.0
        if (action_scalar < 0 and soc_before_action <= 0.0) or (action_scalar > 0 and soc_before_action >= 1.0):
            reward_constraint = -1.0
        if self.emergency_flag:
            try:
                emergency_load = float(self.building.non_shiftable_load[-1]) + float(self.building.cooling_demand[-1])
            except Exception:
                emergency_load = float(current_power_demand)
            if current_power_demand <= emergency_load * 1.05:
                emergency_term = +1.0
            else:
                emergency_term = -1.0
            final_reward = weights['emergency'] * emergency_term
            final_reward += 0.1 * reward_env
            self.reward_history.append(final_reward)
            return final_reward

        trust_term = (1.0 - getattr(self, "trust_score", 1.0))
        reward_shifting = - (electricity_pricing * current_power_demand) * 0.01
        agg_follow = 0.0

        if agg_signal is not None:
            try:
                # If agg_signal is a packet, decrypt to compute agg_follow
                sig = self._process_agg_signal(agg_signal)
                agg_price_val = float(np.atleast_1d(sig)[0])
            except Exception:
                agg_price_val = 0.0
            if agg_price_val > 0 and action_scalar < 0:
                agg_follow = 1.0
            elif agg_price_val > 0 and action_scalar >= 0:
                agg_follow = -1.0

        final_reward = (
            weights['env'] * reward_env +
            weights['constraint'] * reward_constraint +
            weights['shifting'] * reward_shifting +
            weights['agg_follow'] * agg_follow +
            weights['cheating'] * trust_term
        )

        self.reward_history.append(final_reward)
        self.energy_history.append(current_power_demand)
        if hasattr(self.building, 'indoor_dry_bulb_temperature') and len(self.building.indoor_dry_bulb_temperature):
            self.satisfaction_history.append(1.0 - abs(self.building.indoor_dry_bulb_temperature[-1] - 22.0) / 10.0)
        else:
            self.satisfaction_history.append(1.0)

        return float(final_reward)

    # ---------------------------
    # checkpoints
    # ---------------------------
    def save_models(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.policy_net is not None:
            torch.save(self.policy_net.state_dict(), os.path.join(directory, f"consumer_{self.id}_policy_ep{episode}.pth"))
            torch.save(self.target_net.state_dict(), os.path.join(directory, f"consumer_{self.id}_target_ep{episode}.pth"))
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(directory, f"consumer_{self.id}_optim_ep{episode}.pth"))

    def load_models(self, directory, episode):
        """Load if networks were previously created with the same input dim."""
        try:
            # If model files exist we'll infer input dim from the model weights shape
            policy_path = os.path.join(directory, f"consumer_{self.id}_policy_ep{episode}.pth")
            if os.path.exists(policy_path):
                # load state dict to a CPU tensor to inspect
                state = torch.load(policy_path, map_location="cpu")
                # Infer input dim from first linear weight shape
                first_layer_weight = None
                for k, v in state.items():
                    if k.endswith("net.0.weight") or "fc.0.weight" in k:
                        first_layer_weight = v
                        break
                if first_layer_weight is not None:
                    inferred_input = first_layer_weight.shape[1]
                else:
                    inferred_input = None

                if inferred_input is not None and (self.policy_net is None or self.input_dim != inferred_input):
                    self._build_networks(inferred_input)

                self.policy_net.load_state_dict(torch.load(policy_path, map_location="cpu"))
                self.target_net.load_state_dict(torch.load(os.path.join(directory, f"consumer_{self.id}_target_ep{episode}.pth"), map_location="cpu"))
                try:
                    self.optimizer.load_state_dict(torch.load(os.path.join(directory, f"consumer_{self.id}_optim_ep{episode}.pth"), map_location="cpu"))
                except Exception:
                    pass
                print(f"[Consumer {self.id}] Loaded models for episode {episode}")
            else:
                print(f"[Consumer {self.id}] No checkpoint found for episode {episode}")
        except Exception as e:
            print(f"[Consumer {self.id}] Error loading models: {e}")

    # ---------------------------
    # utilities
    # ---------------------------
    def reset_episode(self):
        self.last_action = 0.0
        self.last_action_index = 1
        self.last_state = None
        self.reward_history.clear()
        self.energy_history.clear()
        self.satisfaction_history.clear()
        self.epsilon = 1.0

