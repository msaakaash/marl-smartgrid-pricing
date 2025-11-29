import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


# ==========================================================
#              SECURE COMMUNICATION LAYER
# ==========================================================

class SecureChannel:
    def __init__(self, key_path="security/keys/secret.key"):
        with open(key_path, "rb") as f:
            self.key = f.read()
        self.aead = ChaCha20Poly1305(self.key)

    def encrypt(self, signal: np.ndarray) -> dict:
        """Encrypt the outgoing price/coordination vector."""
        nonce = np.random.bytes(12)
        plaintext = signal.astype(np.float32).tobytes()
        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        # For demonstration (mentor proof)
        print(f"[AGG→ENC] nonce={nonce.hex()[:8]}..., ct={ciphertext[:8].hex()}...")

        return {
            "nonce": nonce,
            "ciphertext": ciphertext,
        }

    def decrypt(self, packet: dict) -> np.ndarray:
        """Aggregator rarely decrypts, but we include method for completeness."""
        try:
            plaintext = self.aead.decrypt(
                packet["nonce"],
                packet["ciphertext"],
                None
            )
            arr = np.frombuffer(plaintext, dtype=np.float32)
            return arr
        except Exception:
            # tamper detected
            print("[AGG] Decryption failed — packet tampered.")
            return np.zeros(2, dtype=np.float32)


# ==========================================================
#                 ACTOR-CRITIC NETWORKS (DDPG)
# ==========================================================

class Actor(nn.Module):
    def __init__(self, state_dim=30, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # output between -1 and 1
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_dim=30, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.fc(x)


# ==========================================================
#                   AGGREGATOR DDPG AGENT
# ==========================================================

class AggregatorDDPG:
    def __init__(self, state_dim=30, action_dim=2,
                 key_path="security/keys/secret.key"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Exploration noise
        self.noise_scale = 0.1

        # Secure communication
        self.secure = SecureChannel(key_path)

    # ------------------------------------------------------
    #      GENERATE AGGREGATOR→CONSUMER SIGNAL
    # ------------------------------------------------------
    def compute_signal(self, agg_state):
        """DDPG forward pass to generate the 2-dim price/signal."""
        s = torch.tensor(agg_state, dtype=torch.float32).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]

        # Add exploration noise
        noise = np.random.normal(0, self.noise_scale, size=self.action_dim)
        final_action = action + noise
        final_action = np.clip(final_action, -1.0, 1.0)

        # For mentor demo
        print(f"[AGG] Raw signal: {final_action}")

        return final_action.astype(np.float32)

    # ------------------------------------------------------
    #            ENCRYPT THE OUTGOING SIGNAL
    # ------------------------------------------------------
    def encrypt_signal(self, signal):
        return self.secure.encrypt(signal)

    # ------------------------------------------------------
    #            TRAINING (OPTIONAL FOR NOW)
    # ------------------------------------------------------
    def update(self, replay_buffer):
        pass  # your MARL training logic here

    # ------------------------------------------------------
    #            MODEL SAVING / LOADING
    # ------------------------------------------------------
    def save_models(self, path, episode):
        torch.save(self.actor.state_dict(),
                   f"{path}/agg_actor_ep{episode}.pth")
        torch.save(self.critic.state_dict(),
                   f"{path}/agg_critic_ep{episode}.pth")

    def load_models(self, path, episode):
        try:
            self.actor.load_state_dict(torch.load(
                f"{path}/agg_actor_ep{episode}.pth",
                map_location="cpu"
            ))
            self.critic.load_state_dict(torch.load(
                f"{path}/agg_critic_ep{episode}.pth",
                map_location="cpu"
            ))
            print(f"[AGG] Loaded models for episode {episode}")
        except:
            print(f"[AGG] Failed loading episode {episode}")
