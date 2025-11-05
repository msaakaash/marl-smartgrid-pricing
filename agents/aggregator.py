import numpy as np
from collections import deque, namedtuple
import random
import math
from typing import List

Experience = namedtuple("Experience", ["obs", "action", "reward", "next_obs"])

class AggregatorAgent:
    def __init__(
        self,
        region_id,
        consumers=None,
        aggregator_type="regional",
        hist_len=100,
        reward_buf_len=200,
        exp_buffer_len=1000,
        batch_size=32,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.id = region_id
        self.type = aggregator_type
        self.consumers = consumers if consumers is not None else []
        self.total_demand = 0.0
        self.history = deque(maxlen=hist_len)  # demand history
        self.last_signal = None
        self.trust_scores = {c.id: 1.0 for c in self.consumers}

        # Adaptive control params
        self.dynamic_threshold = 1500.0
        self.alpha = 0.1  # learning rate for threshold update (adaptive)
        self.beta = 0.05  # learning rate for incentive adaptation (adaptive)

        # Reward tracking & experience replay
        self.recent_rewards = deque(maxlen=reward_buf_len)
        self.recent_reward_raw = deque(maxlen=reward_buf_len)  # raw for normalization baseline
        self.experience_buffer = deque(maxlen=exp_buffer_len)
        self.batch_size = batch_size

        # bookkeeping
        self.last_reward = 0.0
        self.step_count = 0
        self.learn_every = 10  # call learn_from_experience every N steps
        self.theta = {"thr_scale": 1.0, "incentive_bias": 0.0}  # small learned params

        # observation normalization stats (running mean/std)
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 0

    def __repr__(self):
        return (
            f"AggregatorAgent(id={self.id}, type={self.type}, "
            f"num_consumers={len(self.consumers)}, last_demand={self.total_demand:.2f}, "
            f"last_signal={self.last_signal})"
        )

    def add_consumer(self, consumer):
        self.consumers.append(consumer)
        self.trust_scores[consumer.id] = 1.0

    # ---------- Observation handling ----------
    def _running_mean_var_update(self, x: np.ndarray):
        """Welford's algorithm for running mean and variance over obs vector."""
        x = np.asarray(x, dtype=np.float64)
        if self.obs_mean is None:
            self.obs_mean = np.zeros_like(x)
            self.obs_var = np.zeros_like(x)
            self.obs_count = 0

        self.obs_count += 1
        if self.obs_count == 1:
            self.obs_mean = x.copy()
            self.obs_var = np.zeros_like(x)
        else:
            delta = x - self.obs_mean
            self.obs_mean += delta / self.obs_count
            delta2 = x - self.obs_mean
            self.obs_var += delta * delta2

    def _normalize_obs(self, x: np.ndarray):
        """Return normalized observation using running mean/var."""
        if self.obs_mean is None or self.obs_count < 2:
            return x.astype(np.float32)
        std = np.sqrt(self.obs_var / (self.obs_count - 1) + 1e-8)
        return ((x - self.obs_mean) / std).astype(np.float32)

    def get_observation(self, regional_observations: List[List[float]], regional_obs_names: List[List[str]]):
        """
        Build observation vector and normalize it on the fly.
        Expected regional_obs_names[i] maps to regional_observations[i].
        Adds a delta demand feature (current - previous).
        """
        regional_info = {
            "total_demand": 0.0,
            "total_solar": 0.0,
            "avg_soc": 0.0,
            "num_emergency": 0.0,
            "avg_trust": 0.0,
            "hour": 0.0,
            "electricity_pricing": 0.0,
            # extended
            "delta_demand": 0.0,
            "demand_std": 0.0,
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

        # bookkeeping
        prev_total = self.total_demand
        self.total_demand = regional_info["total_demand"]
        regional_info["delta_demand"] = self.total_demand - prev_total

        # history based std
        hist_list = list(self.history)
        if len(hist_list) > 1:
            regional_info["demand_std"] = float(np.std(hist_list))
        else:
            regional_info["demand_std"] = 0.0

        self.history.append(self.total_demand)

        # create vector and update normalization stats
        obs_vec = np.array(list(regional_info.values()), dtype=np.float32)
        self._running_mean_var_update(obs_vec)
        norm_obs = self._normalize_obs(obs_vec)
        return norm_obs

    # ---------- Adaptive learning helpers ----------
    def _update_learning_rates(self):
        """Adjust alpha and beta based on reward variability to speed up or slow learning."""
        if len(self.recent_rewards) < 8:
            return
        std = float(np.std(self.recent_rewards))
        mean = float(np.mean(self.recent_rewards))
        # If reward variance is low -> reduce learning rate (converging), else increase
        if std < max(1e-3, abs(mean) * 0.1):
            self.alpha = max(0.01, self.alpha * 0.95)
            self.beta = max(0.01, self.beta * 0.95)
        else:
            self.alpha = min(0.5, self.alpha * 1.05)
            self.beta = min(0.2, self.beta * 1.05)

    def _update_dynamic_threshold(self):
        """Adaptive threshold update based on recent load history and learned scale theta."""
        if len(self.history) > 5:
            avg_demand = np.mean(self.history)
            # combine smoothing + small learned multiplier
            target = avg_demand * (1.0 + 0.1 * self.theta["thr_scale"])
            self.dynamic_threshold = (1 - self.alpha) * self.dynamic_threshold + self.alpha * target

    # ---------- Action selection ----------
    def select_action(self, obs: np.ndarray, is_dr_needed: bool):
        """
        Return continuous action: [price_signal, incentive_signal]
        Uses a mixture of parametric signals + small learned offsets (theta).
        obs expected order (same as get_observation values)
        """
        total_demand_norm = float(obs[0])
        total_solar_norm = float(obs[1])
        avg_soc_norm = float(obs[2])
        num_emergency_norm = float(obs[3])
        avg_trust_norm = float(obs[4])
        hour_norm = float(obs[5])
        price_norm = float(obs[6])

        self._update_learning_rates()
        self._update_dynamic_threshold()

        base_price = 0.1
        demand_gap = (self.total_demand - self.dynamic_threshold) / max(1.0, self.dynamic_threshold)
        price_signal = base_price + 0.5 * math.tanh(demand_gap * 2.0)

        if not is_dr_needed and demand_gap < 0:
            try:
                hour_frac = (hour_norm + 3.0) % 1.0
                price_signal = base_price * (1 + math.sin(hour_frac * 2 * math.pi))
            except Exception:
                pass

        incentive_signal = 0.05
        if num_emergency_norm > 0.1:
            incentive_signal += 0.2

        trust_anchor = np.tanh(avg_trust_norm)
        if trust_anchor < -0.5:
            incentive_signal -= 0.08
        incentive_signal += 0.05 * trust_anchor
        incentive_signal += 0.01 * self.theta["incentive_bias"]

        if len(self.recent_rewards) > 0:
            reward_trend = float(np.mean(self.recent_rewards))
            incentive_signal += self.beta * reward_trend

        price_signal = float(np.clip(price_signal, 0.0, 1.0))
        incentive_signal = float(np.clip(incentive_signal, 0.0, 1.0))

        self.last_signal = np.array([price_signal, incentive_signal], dtype=np.float32)
        return self.last_signal

    # ---------- Experience replay ----------
    def store_experience(self, obs, action, reward, next_obs):
        self.experience_buffer.append(Experience(obs.copy(), action.copy(), float(reward), next_obs.copy()))

    def learn_from_experience(self):
        """Sample mini-batch and perform simple gradient-free parameter updates."""
        if len(self.experience_buffer) < max(self.batch_size, 10):
            return

        batch = random.sample(self.experience_buffer, k=self.batch_size)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)

        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        trend = float(np.mean(rewards_norm))

        self.theta["thr_scale"] += 0.01 * trend
        self.theta["thr_scale"] = float(np.clip(self.theta["thr_scale"], -2.0, 2.0))

        incentives = np.array([b.action[1] for b in batch], dtype=np.float32)
        corr = np.cov(incentives, rewards_norm)[0, 1] if len(incentives) > 1 else 0.0
        self.theta["incentive_bias"] += 0.05 * corr
        self.theta["incentive_bias"] = float(np.clip(self.theta["incentive_bias"], -1.0, 1.0))

        self.dynamic_threshold += self.alpha * trend * max(1.0, self.dynamic_threshold * 0.01)

        for r in rewards:
            self.recent_reward_raw.append(float(r))
        self.recent_rewards.append(float(trend))

    # ---------- Trust and Reward ----------
    def update_trust(self, consumer_id, participated: bool):
        if consumer_id in self.trust_scores:
            if participated:
                self.trust_scores[consumer_id] = min(1.0, self.trust_scores[consumer_id] + 0.05)
            else:
                self.trust_scores[consumer_id] = max(0.0, self.trust_scores[consumer_id] - 0.1)

    def get_reward(self, all_regional_demands: List[float]):
        if len(all_regional_demands) == 0:
            return 0.0
        peak = float(np.max(all_regional_demands))
        avg = float(np.mean(all_regional_demands))
        if avg == 0:
            return 0.0
        reward = -(peak / avg)

        if len(self.trust_scores) > 0:
            avg_trust = float(np.mean(list(self.trust_scores.values())))
        else:
            avg_trust = 1.0
        reward += 0.1 * (avg_trust - 0.8)

        self.last_reward = float(reward)
        self.recent_reward_raw.append(float(reward))

        if len(self.recent_reward_raw) >= 4:
            mean_r = np.mean(self.recent_reward_raw)
            std_r = np.std(self.recent_reward_raw) + 1e-8
            norm_r = (reward - mean_r) / std_r
        else:
            norm_r = reward

        self.recent_rewards.append(float(norm_r))
        return float(reward)

    # ---------- High-level training step ----------
    def train_step(self, regional_observations, regional_obs_names, is_dr_needed, all_regional_demands, next_regional_observations, next_regional_obs_names):
        obs = self.get_observation(regional_observations, regional_obs_names)
        action = self.select_action(obs, is_dr_needed)
        reward = self.get_reward(all_regional_demands)
        next_obs = self.get_observation(next_regional_observations, next_regional_obs_names)
        self.store_experience(obs, action, reward, next_obs)
        self.step_count += 1
        if self.step_count % self.learn_every == 0:
            self.learn_from_experience()
        return action, reward

    # ---------- Utilities ----------
    def save_state(self):
        return {
            "dynamic_threshold": self.dynamic_threshold,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": dict(self.theta),
            "trust_scores": dict(self.trust_scores),
        }

    def load_state(self, state: dict):
        self.dynamic_threshold = float(state.get("dynamic_threshold", self.dynamic_threshold))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.theta.update(state.get("theta", self.theta))
        self.trust_scores.update(state.get("trust_scores", self.trust_scores))


# ------------------------------
# Example/Minimal training loop
# ------------------------------
if __name__ == "__main__":
    class Consumer:
        def __init__(self, cid, emergency=False):
            self.id = cid
            self.emergency_flag = emergency

    consumers = [Consumer(f"C{i}") for i in range(5)]
    agent = AggregatorAgent(region_id="R1", consumers=consumers)

    def make_obs(num_consumers):
        names = []
        obs = []
        for i in range(num_consumers):
            names.append(["non_shiftable_load", "solar_generation", "electrical_storage_soc", "hour", "electricity_pricing"])
            non_shift = random.uniform(200, 600)
            solar = random.uniform(0, 200)
            soc = random.uniform(0.2, 0.9)
            hour = random.uniform(0, 23)
            price = random.uniform(0.05, 0.25)
            obs.append([non_shift, solar, soc, hour, price])
        return obs, names

    for episode in range(50):
        obs, names = make_obs(len(consumers))
        next_obs, next_names = make_obs(len(consumers))
        is_dr = random.random() < 0.2
        all_regional_demands = [random.uniform(1000, 2500) for _ in range(6)]
        action, reward = agent.train_step(obs, names, is_dr, all_regional_demands, next_obs, next_names)

        if episode % 5 == 0:
            print(f"Ep {episode:02d} | action={action} reward={reward:.4f} threshold={agent.dynamic_threshold:.1f} alpha={agent.alpha:.3f}")

    print("Final theta:", agent.theta)
    print("Done.")
