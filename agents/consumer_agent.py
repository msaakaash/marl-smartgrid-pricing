import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 128)):
        super(QNetwork, self).__init__()
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

class ConsumerAgentDQN:
    def __init__(self, building, building_id, metadata, action_space, device=None,
                 replay_capacity=50000, lr=5e-4, gamma=0.95, batch_size=64, target_update_freq=1000):
        self.building = building
        self.id = building_id
        self.type = metadata.get('building_type', 'residential')
        self.critical_load_fraction = float(metadata.get('critical_load_fraction', 0.2))
        self.prime_hours = metadata.get('prime_hours', [])
        self.cheating_propensity = float(metadata.get('cheating_propensity', 0.0))
        self.emergency_flag = bool(metadata.get('emergency_flag', False))
        self.trust_score = 1.0
        self.action_space = action_space
        self.discrete_actions = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        self.action_dim = len(self.discrete_actions)
        
        # <--- CHANGE 1: Observation dimension is now 18 (to include both signals)
        self.observation_dim = 18
        
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.train_steps = 0
        self.target_update_freq = target_update_freq
        self.last_action = 0.0
        self.last_action_index = 1
        self.last_state = None
        self.soc = 0.0
        self.reward_history = []
        self.energy_history = []
        self.satisfaction_history = []

    def get_observation(self, raw_obs, observation_names, agg_signal=None):
        obs_dict = dict(zip(observation_names, raw_obs))
        building_type_map = {"residential": 0, "hospital": 1, "office": 2, "mall": 3, "industry": 4, "cheater": 5}
        building_type_val = building_type_map.get(self.type, 0)
        is_prime_hour = 1.0 if obs_dict.get("hour") in self.prime_hours else 0.0

        # <--- CHANGE 2: Correctly parse both price and incentive signals
        agg_price_signal = 0.0
        agg_incentive_signal = 0.0
        if agg_signal is not None:
            agg_signal = np.atleast_1d(agg_signal)
            if len(agg_signal) > 0:
                agg_price_signal = float(agg_signal[0])
            if len(agg_signal) > 1:
                agg_incentive_signal = float(agg_signal[1])
        # --- End of Change 2 ---
        
        self.soc = float(obs_dict.get("electrical_storage_soc", self.soc))
        observation = np.array([
            obs_dict.get("month", 0),
            obs_dict.get("day_type", 0),
            obs_dict.get("hour", 0),
            obs_dict.get("outdoor_dry_bulb_temperature", 0.0),
            obs_dict.get("diffuse_solar_irradiance", 0.0),
            obs_dict.get("direct_solar_irradiance", 0.0),
            obs_dict.get("carbon_intensity", 0.0),
            obs_dict.get("non_shiftable_load", 0.0),
            obs_dict.get("solar_generation", 0.0),
            obs_dict.get("electrical_storage_soc", 0.0),
            obs_dict.get("electricity_pricing", 0.0),
            building_type_val,
            self.critical_load_fraction,
            self.cheating_propensity,
            1.0 if self.emergency_flag else 0.0,
            is_prime_hour,
            # <--- CHANGE 3: Add both signals to the state
            agg_price_signal,
            agg_incentive_signal
        ], dtype=np.float32)
        return observation

    def select_action(self, obs):
        if self.emergency_flag:
            action_index = 0
            action_value = float(self.discrete_actions[action_index])
            action_value = float(np.clip(action_value, self.action_space.low, self.action_space.high))
            self.last_action = action_value
            self.last_action_index = int(action_index)
            self.last_state = obs
            return np.array([action_value], dtype=np.float32), action_index
        if random.random() < self.cheating_propensity:
            action_index = self.action_dim - 1
            action_value = float(self.discrete_actions[action_index])
            action_value = float(np.clip(action_value, self.action_space.low, self.action_space.high))
            self.last_action = action_value
            self.last_action_index = int(action_index)
            self.last_state = obs
            return np.array([action_value], dtype=np.float32), action_index
        
        # Standard epsilon-greedy selection
        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_index = int(torch.argmax(q_values, dim=1).item())
        
        # Get the action value from the agent's decision
        action_value = float(self.discrete_actions[action_index])

        # <--- CHANGE 4: REMOVED the manual 'influence' logic.
        # The agent must learn to respond to the signals (which are now in its state)
        # on its own. This is what the 'agg_follow' reward is for.
        
        # Clip the chosen action to the environment's valid range
        low = float(self.action_space.low) if np.isscalar(self.action_space.low) else float(np.atleast_1d(self.action_space.low)[0])
        high = float(self.action_space.high) if np.isscalar(self.action_space.high) else float(np.atleast_1d(self.action_space.high)[0])
        action_value = float(np.clip(action_value, low, high))
        
        self.last_action = action_value
        self.last_action_index = int(action_index)
        self.last_state = obs
        return np.array([action_value], dtype=np.float32), action_index

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
        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()
        return loss.item()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

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
        trust_term = (self.trust_score - 1.0)
        reward_shifting = - (electricity_pricing * current_power_demand) * 0.01
        agg_follow = 0.0
        
        # This reward logic is now fine, as the agent is truly learning.
        # It rewards the agent for following the *price* signal (agg_val[0])
        if agg_signal is not None:
            # We still only use the price signal for this reward component.
            # The incentive signal's value will be learned implicitly
            # as it affects the agent's state and future rewards.
            agg_price_val = float(np.atleast_1d(agg_signal)[0])
            if agg_price_val > 0 and action_scalar < 0: # High price -> discharge
                agg_follow = 1.0
            elif agg_price_val > 0 and action_scalar >= 0: # High price -> NOT discharge
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

    def report_response(self, expected_reduction, actual_reduction):
        if expected_reduction <= 0:
            return
        ratio = actual_reduction / expected_reduction
        if ratio < 0.5:
            self.trust_score = max(0.0, self.trust_score - 0.1)
            self.cheating_propensity = min(1.0, self.cheating_propensity + 0.05)
        else:
            self.trust_score = min(1.0, self.trust_score + 0.02)
            self.cheating_propensity = max(0.0, self.cheating_propensity - 0.01)

    def set_emergency(self, flag=True):
        self.emergency_flag = bool(flag)

    def reset_episode(self):
        self.last_action = 0.0
        self.last_action_index = 1
        self.last_state = None
        self.reward_history.clear()
        self.energy_history.clear()
        self.satisfaction_history.clear()
        self.epsilon = 1.0

# ----------------------------------------------------
# New Rule-Based Agent Class for Comparison
# ----------------------------------------------------
class ConsumerAgentRuleBased:
    def __init__(self, building, building_id, metadata, action_space):
        self.building = building
        self.id = building_id
        self.type = metadata.get('building_type', 'residential')
        self.critical_load_fraction = float(metadata.get('critical_load_fraction', 0.2))
        self.prime_hours = metadata.get('prime_hours', [])
        self.cheating_propensity = float(metadata.get('cheating_propensity', 0.0))
        self.emergency_flag = bool(metadata.get('emergency_flag', False))
        self.action_space = action_space
        self.discrete_actions = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        self.action_dim = len(self.discrete_actions)
        self.soc = 0.0
        
        # <--- CHANGE 5: Observation dimension updated for consistency
        self.observation_dim = 18
        
        self.last_action = 0.0
        self.last_action_index = 1
        self.reward_history = []
        self.energy_history = []
        self.satisfaction_history = []

    def get_observation(self, raw_obs, observation_names, agg_signal=None):
        obs_dict = dict(zip(observation_names, raw_obs))
        building_type_map = {"residential": 0, "hospital": 1, "office": 2, "mall": 3, "industry": 4, "cheater": 5}
        building_type_val = building_type_map.get(self.type, 0)
        is_prime_hour = 1.0 if obs_dict.get("hour") in self.prime_hours else 0.0
        
        # Also update the rule-based agent to "see" both signals
        agg_price_signal = 0.0
        agg_incentive_signal = 0.0
        if agg_signal is not None:
            agg_signal = np.atleast_1d(agg_signal)
            if len(agg_signal) > 0:
                agg_price_signal = float(agg_signal[0])
            if len(agg_signal) > 1:
                agg_incentive_signal = float(agg_signal[1])

        self.soc = float(obs_dict.get("electrical_storage_soc", self.soc))
        observation = np.array([
            obs_dict.get("month", 0),
            obs_dict.get("day_type", 0),
            obs_dict.get("hour", 0),
            obs_dict.get("outdoor_dry_bulb_temperature", 0.0),
            obs_dict.get("diffuse_solar_irradiance", 0.0),
            obs_dict.get("direct_solar_irradiance", 0.0),
            obs_dict.get("carbon_intensity", 0.0),
            obs_dict.get("non_shiftable_load", 0.0),
            obs_dict.get("solar_generation", 0.0),
            obs_dict.get("electrical_storage_soc", 0.0),
            obs_dict.get("electricity_pricing", 0.0),
            building_type_val,
            self.critical_load_fraction,
            self.cheating_propensity,
            1.0 if self.emergency_flag else 0.0,
            is_prime_hour,
            agg_price_signal,      # Add price
            agg_incentive_signal   # Add incentive
        ], dtype=np.float32)
        return observation

    def select_action(self, obs):
        # The rule-based agent can now also use the new signals
        agg_price_signal = obs[16]
        agg_incentive_signal = obs[17]
        electricity_pricing = obs[10]
        soc = obs[9]
        hour = obs[2]

        action_index = 1  # Default to no action (0.0)

        # Rule 1: Respond to aggregator PRICE signal
        if agg_price_signal > 0.5:
            action_index = 0  # Reduce consumption
        
        # Rule 2: Respond to high electricity pricing
        elif electricity_pricing > 0.2:
            action_index = 0  # Reduce consumption
        
        # Rule 3: Charge battery during low-cost/high-solar hours if SoC is low
        elif soc < 0.2 and (electricity_pricing < 0.05 or obs[8] > 0.5):
            action_index = 2  # Increase consumption (for charging)

        # Rule 4: Discharge battery during peak hours if SoC is high
        elif soc > 0.8 and hour in self.prime_hours:
            action_index = 0  # Reduce consumption (by discharging)
        
        # Rule 5: Don't charge if battery is full or discharge if empty
        if action_index == 2 and soc >= 0.95:
            action_index = 1
        elif action_index == 0 and soc <= 0.05:
            action_index = 1

        action_value = float(self.discrete_actions[action_index])
        
        self.last_action = action_value
        self.last_action_index = int(action_index)
        
        return np.array([action_value], dtype=np.float32), action_index

    # ... all other methods for RuleBased (learn, store, etc.) remain empty ...
    def store_experience(self, *args):
        pass
    def learn(self):
        return None
    def update_target_network(self):
        pass
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
        trust_term = (self.trust_score - 1.0) if hasattr(self, 'trust_score') else 0.0
        reward_shifting = - (electricity_pricing * current_power_demand) * 0.01
        agg_follow = 0.0
        if agg_signal is not None:
            agg_price_val = float(np.atleast_1d(agg_signal)[0])
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
    def report_response(self, expected_reduction, actual_reduction):
        pass
    def set_emergency(self, flag=True):
        pass
    def reset_episode(self):
        self.last_action = 0.0
        self.last_action_index = 1
        self.reward_history.clear()
        self.energy_history.clear()
        self.satisfaction_history.clear()