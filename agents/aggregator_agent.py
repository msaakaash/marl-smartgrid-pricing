import numpy as np
from collections import deque

class AggregatorAgent:
    def __init__(self, region_id, consumers=None, aggregator_type="regional"):
        self.id = region_id
        self.type = aggregator_type
        self.consumers = consumers if consumers is not None else []
        self.total_demand = 0.0
        self.history = deque(maxlen=100)
        self.last_signal = None

        # Use NumPy arrays for faster computation
        self.trust_scores = np.ones(len(self.consumers), dtype=np.float32)

        # Learning and control parameters
        self.dynamic_threshold = 1500.0
        self.alpha = 0.1
        self.beta = 0.05
        self.recent_rewards = deque(maxlen=50)
        self.last_reward = 0.0

    def __repr__(self):
        return (f"AggregatorAgent(id={self.id}, type={self.type}, "
                f"num_consumers={len(self.consumers)}, "
                f"last_demand={self.total_demand:.2f}, "
                f"last_signal={self.last_signal})")

    def add_consumer(self, consumer):
        self.consumers.append(consumer)
        self.trust_scores = np.append(self.trust_scores, 1.0)

    def get_observation(self, regional_observations, regional_obs_names):
        num_consumers = len(self.consumers)
        if num_consumers == 0:
            return np.zeros(7, dtype=np.float32)

        regional_info = np.zeros(7, dtype=np.float32)  # total_demand, total_solar, avg_soc, num_emergency, avg_trust, hour, electricity_pricing
        emergency_count = 0

        # Use vectorized-like accumulation to minimize dict ops
        for i, consumer in enumerate(self.consumers):
            obs_dict = dict(zip(regional_obs_names[i], regional_observations[i]))
            regional_info[0] += obs_dict.get("non_shiftable_load", 0.0)
            regional_info[1] += obs_dict.get("solar_generation", 0.0)
            regional_info[2] += obs_dict.get("electrical_storage_soc", 0.0)
            if getattr(consumer, "emergency_flag", False):
                emergency_count += 1

        shared_dict = dict(zip(regional_obs_names[0], regional_observations[0]))
        regional_info[3] = emergency_count
        regional_info[4] = np.mean(self.trust_scores) if len(self.trust_scores) else 1.0
        regional_info[5] = shared_dict.get("hour", 0.0)
        regional_info[6] = shared_dict.get("electricity_pricing", 0.0)

        # Average values
        regional_info[2] /= num_consumers

        self.total_demand = regional_info[0]
        self.history.append(self.total_demand)
        return regional_info

    def _update_dynamic_threshold(self):
        """Adaptive threshold update (only when enough history exists)."""
        if len(self.history) > 10:
            avg_demand = np.mean(self.history)
            self.dynamic_threshold += self.alpha * ((avg_demand * 1.1) - self.dynamic_threshold)

    def select_action(self, obs, is_dr_needed):
        total_demand, total_solar, avg_soc, num_emergency, avg_trust, hour, price = obs

        # Update threshold occasionally (reduces redundant calls)
        if len(self.history) % 5 == 0:
            self._update_dynamic_threshold()

        # Fast math via NumPy scalar ops
        base_signal = 0.1
        if is_dr_needed or total_demand > self.dynamic_threshold:
            price_signal = base_signal + 0.5 * np.tanh((total_demand - self.dynamic_threshold) / 1000)
        else:
            price_signal = base_signal * (1 + np.sin(hour * 0.2618))  # 2π/24 ≈ 0.2618

        incentive_signal = 0.05 + 0.05 * (avg_trust - 0.8)
        if num_emergency > 0:
            incentive_signal += 0.2
        if avg_trust < 0.7:
            incentive_signal -= 0.1

        # Reward-guided adjustment
        if self.recent_rewards:
            incentive_signal += self.beta * np.mean(self.recent_rewards)

        self.last_signal = np.clip([price_signal, incentive_signal], 0.0, 1.0).astype(np.float32)
        return self.last_signal

    def update_trust(self, consumer_id, participated):
        # Use direct array index access instead of dict lookups
        if 0 <= consumer_id < len(self.trust_scores):
            self.trust_scores[consumer_id] = np.clip(
                self.trust_scores[consumer_id] + (0.05 if participated else -0.1),
                0.0, 1.0
            )

    def get_reward(self, all_regional_demands):
        if not all_regional_demands:
            return 0.0
        arr = np.asarray(all_regional_demands, dtype=np.float32)
        avg = np.mean(arr)
        if avg == 0:
            return 0.0

        reward = -(np.max(arr) / avg)
        reward += 0.1 * (np.mean(self.trust_scores) - 0.8)
        self.last_reward = reward
        self.recent_rewards.append(reward)
        return reward
