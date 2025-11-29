import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# NEW secure channel import
from secure_channel import SecureChannel


# ==========================================================
#                      DQN NETWORKS
# ==========================================================

class QNetwork(nn.Module):
    def __init__(self, input_dim=30, output_dim=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================================
#                SECURE CONSUMER DQN AGENT
# ==========================================================

class ConsumerAgentDQN:
    def __init__(self, building, building_id, metadata, action_space,
                 key_path="security/keys/secret.key"):

        self.id = building_id
        self.building = building
        self.metadata = metadata
        self.action_space = action_space
        self.device = torch.device("cpu")

        # RL components
        self.state_dim = 30
        self.action_dim = 5
        self.gamma = 0.99
        self.lr = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05

        self.memory = deque(maxlen=50000)
        self.batch_size = 64

        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # NEW secure communication channel
        # debug=True enables printing encryption/decryption details
        self.secure = SecureChannel(key_path, debug=True)

        # Track last action for training
        self.last_state = None
        self.last_action_index = None

    # -----------------------------------------------------
    #               SECURE COMMUNICATION
    # -----------------------------------------------------
    def decrypt_signal(self, encrypted_packet):
        """
        Decrypts the aggregator AES-AD encrypted packet OR plaintext fallback.
        Returns np.array([x, y], dtype=float32)
        """
        return self.secure.decrypt_vector(encrypted_packet)

    # -----------------------------------------------------
    #              BUILD 30-DIM OBSERVATION
    # -----------------------------------------------------
    def get_observation(self, raw_obs, obs_names, agg_signal):
        raw_obs = np.array(raw_obs, dtype=np.float32)

        if agg_signal is None:
            agg_signal = np.array([0.0, 0.0], dtype=np.float32)

        full = np.concatenate([raw_obs, agg_signal])
        return full

    # -----------------------------------------------------
    #                  REWARD FUNCTION
    # -----------------------------------------------------
    def get_reward(self, raw_reward, net_energy, action, soc_before,
                   price, agg_signal):
        r = raw_reward
        r -= abs(action) * 0.05
        if soc_before < 0.1:
            r -= 0.2
        if soc_before > 0.9:
            r -= 0.2
        return float(r)

    # -----------------------------------------------------
    #            EPSILON-GREEDY ACTION SELECTION
    # -----------------------------------------------------
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)

        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            action_index = int(torch.argmax(q_values).item())

        # map discrete index to action value
        action_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        action = action_values[action_index]

        self.last_state = state
        self.last_action_index = action_index

        return torch.tensor([action], dtype=torch.float32), action_index

    # -----------------------------------------------------
    #                   REPLAY BUFFER
    # -----------------------------------------------------
    def store_experience(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    # -----------------------------------------------------
    #                      LEARNING
    # -----------------------------------------------------
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)

        s = torch.from_numpy(np.array(s, dtype=np.float32))
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.from_numpy(np.array(s2, dtype=np.float32))
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_pred = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -----------------------------------------------------
    #                 EPISODE RESET
    # -----------------------------------------------------
    def reset_episode(self):
        pass

    # -----------------------------------------------------
    #                  SAVE MODELS
    # -----------------------------------------------------
    def save_models(self, path, episode):
        torch.save(self.policy_net.state_dict(),
                   f"{path}/consumer_policy_{self.id}_ep{episode}.pth")
        torch.save(self.target_net.state_dict(),
                   f"{path}/consumer_target_{self.id}_ep{episode}.pth")

    # -----------------------------------------------------
    #                  LOAD MODELS
    # -----------------------------------------------------
    def load_models(self, path, episode):
        try:
            self.policy_net.load_state_dict(torch.load(
                f"{path}/consumer_policy_{self.id}_ep{episode}.pth",
                map_location="cpu"
            ))
            self.target_net.load_state_dict(torch.load(
                f"{path}/consumer_target_{self.id}_ep{episode}.pth",
                map_location="cpu"
            ))
            print(f"Loaded Consumer {self.id} models for episode {episode}")
        except:
            print(f"Failed loading Consumer {self.id} episode {episode}")
