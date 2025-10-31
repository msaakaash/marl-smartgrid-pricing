import numpy as np
from collections import deque

class AggregatorAgent:
    def __init__(self, region_id, consumers=None, aggregator_type="regional"):
        self.id = region_id
        self.type = aggregator_type
        self.consumers = consumers if consumers is not None else []
        self.total_demand = 0.0
        self.history = deque(maxlen=100)  # for dynamic threshold learning
        self.last_signal = None
        self.trust_scores = {c.id: 1.0 for c in self.consumers}

        # Learning and control parameters
        self.dynamic_threshold = 1500.0
        self.alpha = 0.1  # learning rate for threshold update
        self.beta = 0.05  # learning rate for incentive adaptation
        self.recent_rewards = deque(maxlen=50)
        self.last_reward = 0.0

    def __repr__(self):
        return (f"AggregatorAgent(id={self.id}, type={self.type}, "
                f"num_consumers={len(self.consumers)}, "
                f"last_demand={self.total_demand:.2f}, "
                f"last_signal={self.last_signal})")

    def add_consumer(self, consumer):
        self.consumers.append(consumer)
        self.trust_scores[consumer.id] = 1.0

    def get_observation(self, regional_observations, regional_obs_names):
        regional_info = {
            "total_demand": 0.0,
            "total_solar": 0.0,
            "avg_soc": 0.0,
            "num_emergency": 0.0,
            "avg_trust": 0.0,
            "hour": 0.0,
            "electricity_pricing": 0.0
        }

        num_consumers = len(self.consumers)
        if num_consumers > 0:
            for i in range(num_consumers):
                obs_dict = dict(zip(regional_obs_names[i], regional_observations[i]))
                regional_info["total_demand"] += obs_dict.get("non_shiftable_load", 0.0)
                regional_info["total_solar"] += obs_dict.get("solar_generation", 0.0)
                regional_info["avg_soc"] += obs_dict.get("electrical_storage_soc", 0.0)
                regional_info["avg_trust"] += self.trust_scores.get(self.consumers[i].id, 1.0)
                if getattr(self.consumers[i], "emergency_flag", False):
                    regional_info["num_emergency"] += 1

            regional_info["avg_soc"] /= num_consumers
            regional_info["avg_trust"] /= num_consumers
            shared_dict = dict(zip(regional_obs_names[0], regional_observations[0]))
            regional_info["hour"] = shared_dict.get("hour", 0.0)
            regional_info["electricity_pricing"] = shared_dict.get("electricity_pricing", 0.0)

        self.total_demand = regional_info["total_demand"]
        self.history.append(self.total_demand)
        return np.array(list(regional_info.values()), dtype=np.float32)

    def _update_dynamic_threshold(self):
        """Adaptive threshold update based on recent load history."""
        if len(self.history) > 10:
            avg_demand = np.mean(self.history)
            # smooth update
            self.dynamic_threshold = (1 - self.alpha) * self.dynamic_threshold + self.alpha * (avg_demand * 1.1)

    def select_action(self, obs, is_dr_needed):
        # Expected order: [total_demand, total_solar, avg_soc, num_emergency, avg_trust, hour, electricity_pricing]
        total_demand, total_solar, avg_soc, num_emergency, avg_trust, hour, price = obs

        # Update internal learning
        self._update_dynamic_threshold()

        # Adaptive price control
        base_signal = 0.1
        if is_dr_needed or total_demand > self.dynamic_threshold:
            price_signal = base_signal + 0.5 * np.tanh((total_demand - self.dynamic_threshold) / 1000)
        else:
            price_signal = base_signal * (1 + np.sin(hour / 24 * 2 * np.pi))  # smooth diurnal variation

        # Incentive control
        incentive_signal = 0.05
        if num_emergency > 0:
            incentive_signal += 0.2
        if avg_trust < 0.7:
            incentive_signal -= 0.1
        # trust-weighted adjustment
        incentive_signal += 0.05 * (avg_trust - 0.8)

        # Reward-guided fine-tuning (reinforcement-like)
        if self.recent_rewards:
            reward_trend = np.mean(self.recent_rewards)
            incentive_signal += self.beta * reward_trend

        # Clip and save
        price_signal = np.clip(price_signal, 0.0, 1.0)
        incentive_signal = np.clip(incentive_signal, 0.0, 1.0)
        self.last_signal = np.array([price_signal, incentive_signal], dtype=np.float32)
        return self.last_signal

    def update_trust(self, consumer_id, participated):
        if consumer_id in self.trust_scores:
            if participated:
                self.trust_scores[consumer_id] = min(1.0, self.trust_scores[consumer_id] + 0.05)
            else:
                self.trust_scores[consumer_id] = max(0.0, self.trust_scores[consumer_id] - 0.1)

    def get_reward(self, all_regional_demands):
        if len(all_regional_demands) == 0:
            return 0.0
        peak = np.max(all_regional_demands)
        avg = np.mean(all_regional_demands)
        if avg == 0:
            return 0.0
        reward = -(peak / avg)
        # Reward adjustment: higher trust improves reward slightly
        avg_trust = np.mean(list(self.trust_scores.values()))
        reward += 0.1 * (avg_trust - 0.8)
        self.last_reward = reward
        self.recent_rewards.append(reward)
        return reward
