import numpy as np
import random
import copy
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from citylearn.citylearn import CityLearnEnv

from profiles import DEFAULT_PROFILE_TEMPLATE, normalize_profile
from rewards import consumer_reward, aggregator_reward, grid_reward


class CityLearnParallelEnv(ParallelEnv):
    metadata = {"render_modes": []}

    def __init__(self, schema_path: str, n_consumers: int = 5):
        self.env = CityLearnEnv(schema=schema_path)
        self.n_consumers = n_consumers
        self.agents = [f"consumer_{i}" for i in range(n_consumers)] + ["aggregator", "grid"]

        # spaces (rough estimates, update dynamically in reset)
        obs_dim = 10
        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {
            **{f"consumer_{i}": spaces.Box(-1.0, 1.0, (1,), dtype=np.float32) for i in range(n_consumers)},
            "aggregator": spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32),
            "grid": spaces.Box(low=np.array([0.5]), high=np.array([2.0]), dtype=np.float32),
        }

        self.profiles = []
        self.current_price = 0.5
        self.current_grid_limit = 5.0
        self.last_incentive_budget = 0.0

    def _init_profiles(self, seed=None):
        rng = np.random.RandomState(seed)
        self.profiles = []
        for i in range(self.n_consumers):
            p = copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)
            p["type"] = "commercial" if rng.rand() < 0.3 else "residential"
            p["flexibility"] = float(max(0.05, min(1.0, rng.normal(0.5, 0.15))))
            p["priority"] = int(rng.randint(1, 5))
            p["willingness_to_shift"] = float(rng.rand())
            p["emergency_status"] = 0
            self.profiles.append(p)

    def _update_dynamic_profiles(self, incentives, comfort_devs=None,
                                 emergency_prob=0.01, emergency_clear_prob=0.2,
                                 willingness_gain=0.05, willingness_decay=0.995,
                                 flex_jitter=0.05):
        if comfort_devs is None:
            comfort_devs = [0.0] * self.n_consumers

        for i, prof in enumerate(self.profiles):
            if random.random() < emergency_prob:
                prof["emergency_status"] = 1
            elif prof["emergency_status"] == 1 and random.random() < emergency_clear_prob:
                prof["emergency_status"] = 0

            prof["willingness_to_shift"] = prof["willingness_to_shift"] * willingness_decay + willingness_gain * incentives[i]
            if comfort_devs[i] > 0.1:
                prof["willingness_to_shift"] *= 0.9
            prof["willingness_to_shift"] = float(np.clip(prof["willingness_to_shift"], 0.0, 1.0))

            base = prof["flexibility"]
            jitter = np.random.uniform(-flex_jitter, flex_jitter)
            prof["flexibility"] = float(np.clip(base + jitter, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        self._init_profiles(seed)

        obs_dict = {}
        incentives = [0.0] * self.n_consumers
        for i in range(self.n_consumers):
            profile_vec = normalize_profile(self.profiles[i])
            obs_dict[f"consumer_{i}"] = np.concatenate([
                obs[i].astype(np.float32),
                np.array([self.current_price, self.current_grid_limit, incentives[i]], dtype=np.float32),
                profile_vec,
            ])

        obs_dict["aggregator"] = np.array([0.0, 0.0], dtype=np.float32)
        obs_dict["grid"] = np.array([self.current_grid_limit], dtype=np.float32)
        return obs_dict, {}

    def step(self, actions):
        # Extract aggregator action
        agg_action = actions.get("aggregator", [0.5, 0.0])
        self.current_price = float(agg_action[0])
        self.last_incentive_budget = float(agg_action[1])

        # Distribute incentives
        weights = np.array([p["flexibility"] * p["willingness_to_shift"] for p in self.profiles])
        if weights.sum() <= 1e-8:
            weights = np.ones_like(weights)
        weights = weights / (weights.sum() + 1e-12)
        incentives = list((self.last_incentive_budget * weights).astype(float))

        # Grid action
        grid_action = actions.get("grid", [self.current_grid_limit])
        self.current_grid_limit = float(grid_action[0])

        # Step consumers
        consumer_actions = [actions.get(f"consumer_{i}", [0.0])[0] for i in range(self.n_consumers)]
        next_obs, _, done, _, _ = self.env.step(consumer_actions)

        # Update dynamic profiles
        self._update_dynamic_profiles(incentives)

        # Build obs, rewards
        obs_dict, rewards_dict = {}, {}
        total_consumption = float(np.sum(self.env.net_electricity_consumption))
        peak_demand = float(np.max(self.env.net_electricity_consumption))

        for i in range(self.n_consumers):
            profile_vec = normalize_profile(self.profiles[i])
            obs_dict[f"consumer_{i}"] = np.concatenate([
                next_obs[i].astype(np.float32),
                np.array([self.current_price, self.current_grid_limit, incentives[i]], dtype=np.float32),
                profile_vec,
            ])

            consumption = float(self.env.net_electricity_consumption[i])
            comfort_dev = 0.0
            emergency = self.profiles[i]["emergency_status"]
            rewards_dict[f"consumer_{i}"] = consumer_reward(consumption, self.current_price, comfort_dev, emergency, incentives[i])

        rewards_dict["aggregator"] = aggregator_reward(peak_demand, self.last_incentive_budget)
        rewards_dict["grid"] = grid_reward(total_consumption, self.current_grid_limit)

        obs_dict["aggregator"] = np.array([peak_demand, self.last_incentive_budget], dtype=np.float32)
        obs_dict["grid"] = np.array([self.current_grid_limit], dtype=np.float32)

        truncs = {a: False for a in self.agents}
        terms = {a: done for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs_dict, rewards_dict, terms, truncs, infos

    def render(self):
        pass

    def close(self):
        self.env.close()
