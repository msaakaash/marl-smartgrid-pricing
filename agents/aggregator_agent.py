# agents/aggregator_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Dict, Any

# Import SecureChannel from security package (should provide encrypt(np.ndarray)->dict)
try:
    from security.secure_channel import SecureChannel
except Exception:
    SecureChannel = None  # graceful fallback


# ----------------------------
# Networks
# ----------------------------
class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 16), output_dim),
            nn.Tanh()  # bounded outputs in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 16), 1)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


# ----------------------------
# Aggregator Agent
# ----------------------------
class AggregatorAgentDDPG:
    """
    Aggregator agent that creates per-consumer signals and encrypts them via SecureChannel.
    Dimensions are derived from consumers list where possible.
    """

    def __init__(self,
                 region_id: int,
                 consumers: List[Any],
                 key_path: str = "security/keys/secret.key",
                 consumer_signal_dim: int = 2,
                 device: str = "cpu"):
        """
        consumers: list of consumer agent objects. If a consumer exposes `observation_dim` (or `state_dim`),
                   that will be used to determine consumer_state_dim. Otherwise 30 is assumed.
        consumer_signal_dim: number of signal floats per consumer (default 2).
        """
        self.id = region_id
        self.consumers = consumers if consumers is not None else []
        self.device = torch.device(device)

        # Derive consumer_state_dim from first consumer if possible
        self.consumer_state_dim = 30
        if len(self.consumers) > 0:
            c0 = self.consumers[0]
            if hasattr(c0, "observation_dim"):
                try:
                    self.consumer_state_dim = int(getattr(c0, "observation_dim"))
                except Exception:
                    pass
            elif hasattr(c0, "state_dim"):
                try:
                    self.consumer_state_dim = int(getattr(c0, "state_dim"))
                except Exception:
                    pass
            else:
                # try to call get_observation with a dummy to infer size (safe-guarded)
                try:
                    sample = getattr(c0, "last_state", None)
                    if sample is not None:
                        self.consumer_state_dim = int(np.asarray(sample).size)
                except Exception:
                    pass

        self.consumer_signal_dim = int(consumer_signal_dim)
        self.num_consumers = max(0, len(self.consumers))

        # Derived dims
        self.state_dim = self.num_consumers * self.consumer_state_dim
        self.action_dim = self.num_consumers * self.consumer_signal_dim

        # Learning hyperparams (kept conservative for CPU)
        self.gamma = 0.99
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.batch_size = 64
        self.tau = 0.005
        self.replay_capacity = 100000
        self.memory = deque(maxlen=self.replay_capacity)

        # Build models (handle zero-consumer case safely)
        if self.state_dim <= 0 or self.action_dim <= 0:
            # trivial placeholder networks to avoid runtime errors
            self.actor = Actor(1, 1).to(self.device)
            self.actor_target = Actor(1, 1).to(self.device)
            self.critic = Critic(1, 1).to(self.device)
            self.critic_target = Critic(1, 1).to(self.device)
        else:
            self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
            self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
            self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
            self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
            # copy weights
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # bookkeeping
        self.last_state = np.zeros(self.state_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards = deque(maxlen=50)

        # Secure channel initialization with graceful fallback
        self.secure = None
        self.secure_ok = False
        if SecureChannel is not None:
            try:
                self.secure = SecureChannel(key_path)
                self.secure_ok = True
                print(f"[Aggregator {self.id}] SecureChannel initialized (key: {key_path})")
            except Exception as e:
                self.secure = None
                self.secure_ok = False
                print(f"[Aggregator {self.id}] SecureChannel init failed: {e} — continuing without encryption.")
        else:
            print(f"[Aggregator {self.id}] SecureChannel module not available — continuing without encryption.")

    # -------------------------
    def __repr__(self):
        return f"AggregatorAgentDDPG(id={self.id}, consumers={self.num_consumers}, state_dim={self.state_dim}, action_dim={self.action_dim})"

    # -------------------------
    def reset_episode(self):
        self.last_state = np.zeros(self.state_dim, dtype=np.float32)
        self.last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self.recent_rewards.clear()

    # -------------------------
    def get_observation(self, consumer_state_list: List[np.ndarray]) -> np.ndarray:
        """
        Concatenates the list of consumer states into a single 1D numpy vector of length state_dim.
        It will pad or truncate if shapes mismatch.
        """
        if not consumer_state_list:
            return np.zeros(self.state_dim, dtype=np.float32)

        flat = []
        for s in consumer_state_list:
            arr = np.asarray(s, dtype=np.float32).ravel()
            # if arr has different length than consumer_state_dim, pad/truncate locally
            if arr.size != self.consumer_state_dim:
                tmp = np.zeros(self.consumer_state_dim, dtype=np.float32)
                l = min(arr.size, self.consumer_state_dim)
                tmp[:l] = arr[:l]
                arr = tmp
            flat.append(arr)
        cat = np.concatenate(flat, axis=0) if flat else np.zeros(self.state_dim, dtype=np.float32)

        # final pad/truncate to expected state_dim
        if cat.size != self.state_dim:
            tmp = np.zeros(self.state_dim, dtype=np.float32)
            l = min(cat.size, self.state_dim)
            tmp[:l] = cat[:l]
            cat = tmp

        self.last_state = cat
        return cat

    # -------------------------
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """
        Deterministic actor -> returns action vector length action_dim (values in [-1,1]).
        """
        if self.action_dim <= 0 or self.state_dim <= 0:
            return np.zeros(self.action_dim, dtype=np.float32)

        s = torch.tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).cpu().numpy()[0]
        if noise_std and noise_std > 0.0:
            a = a + np.random.normal(0.0, noise_std, size=a.shape)
        a = np.clip(a, -1.0, 1.0)
        self.last_actions = a.astype(np.float32)
        return self.last_actions

    # -------------------------
    def store_experience(self, s, a, consumer_actions, r, s2, done):
        self.memory.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(consumer_actions, dtype=np.float32),
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(done)
        ))

    # -------------------------
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float = None):
        if tau is None:
            tau = self.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    # -------------------------
    def learn(self, all_next_consumer_actions_batch: Any = None):
        if len(self.memory) < self.batch_size:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        s_b, a_b, c_b, r_b, s2_b, done_b = zip(*batch)

        s = torch.tensor(np.array(s_b), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a_b), dtype=torch.float32, device=self.device)
        r = torch.tensor(np.array(r_b), dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(s2_b), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(done_b), dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            q_target = r + (1.0 - done) * self.gamma * q2

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

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return float(actor_loss.item()), float(critic_loss.item())

    # -------------------------
    def get_reward(self, regional_demands: List[float]) -> float:
        if not regional_demands:
            return 0.0
        val = float(regional_demands[0])
        self.recent_rewards.append(val)
        return -abs(val)

    # -------------------------
    def encrypt_signals_for_consumers(self, raw_action_1d: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        raw_action_1d: 1D numpy array length == action_dim (or will be padded/truncated).
        Splits into per-consumer vectors of length consumer_signal_dim and encrypts each.
        Returns dict: consumer_id -> packet (dict). If encryption not available, returns {'plaintext': [..]}
        Also prints short logs so encryption is visible during training.
        """
        out = {}
        if self.action_dim <= 0:
            return out

        arr = np.asarray(raw_action_1d, dtype=np.float32).flatten()
        if arr.size != self.action_dim:
            tmp = np.zeros(self.action_dim, dtype=np.float32)
            l = min(arr.size, self.action_dim)
            tmp[:l] = arr[:l]
            arr = tmp

        for i, c in enumerate(self.consumers):
            start = i * self.consumer_signal_dim
            end = start + self.consumer_signal_dim
            vec = arr[start:end].astype(np.float32)

            if self.secure_ok and self.secure is not None:
                try:
                    packet = self.secure.encrypt(vec)
                    out[c.id] = packet
                    # print brief debug (nonce & ciphertext preview)
                    try:
                        n = packet.get("nonce", b"")
                        ct = packet.get("ciphertext", b"")
                        print(f"[Aggregator {self.id}] -> Consumer {c.id} | vec={vec.tolist()} | nonce={n.hex()[:16]}... | ct={ct.hex()[:16]}...")
                    except Exception:
                        print(f"[Aggregator {self.id}] -> Consumer {c.id} encrypted (packet keys present).")
                except Exception as e:
                    print(f"[Aggregator {self.id}] Encryption error for consumer {c.id}: {e}. Sending plaintext.")
                    out[c.id] = {"plaintext": vec.tolist()}
            else:
                out[c.id] = {"plaintext": vec.tolist()}
                print(f"[Aggregator {self.id}] (NO-SEC) -> Consumer {c.id} plaintext: {vec.tolist()}")

        return out

    # -------------------------
    def save_models(self, path: str, episode: int):
        if not path:
            return
        torch.save(self.actor.state_dict(), f"{path}/agg_actor_{self.id}_ep{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/agg_critic_{self.id}_ep{episode}.pth")

    # -------------------------
    def load_models(self, path: str, episode: int):
        try:
            self.actor.load_state_dict(torch.load(f"{path}/agg_actor_{self.id}_ep{episode}.pth", map_location="cpu"))
            self.critic.load_state_dict(torch.load(f"{path}/agg_critic_{self.id}_ep{episode}.pth", map_location="cpu"))
            print(f"Loaded Aggregator {self.id} models for episode {episode}")
        except FileNotFoundError:
            print(f"No checkpoint found for Aggregator {self.id} episode {episode}")
        except Exception as e:
            print(f"Error loading Aggregator {self.id} models: {e}")
