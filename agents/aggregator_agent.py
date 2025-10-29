import numpy as np

class AggregatorAgent:
    def __init__(self, region_id, consumers=None, aggregator_type="regional"):
        self.id = region_id
        self.type = aggregator_type
        self.consumers = consumers if consumers is not None else []
        self.total_demand = 0.0
        self.history = []
        self.last_signal = None
        self.trust_scores = {c.id: 1.0 for c in self.consumers}

    def __repr__(self):
        return (f"AggregatorAgent(id={self.id}, type={self.type}, "
                f"num_consumers={len(self.consumers)}, "
                f"last_demand={self.total_demand}, "
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
                consumer_obs_dict = dict(zip(regional_obs_names[i], regional_observations[i]))

                regional_info["total_demand"] += consumer_obs_dict.get("non_shiftable_load", 0.0)
                regional_info["total_solar"] += consumer_obs_dict.get("solar_generation", 0.0)
                regional_info["avg_soc"] += consumer_obs_dict.get("electrical_storage_soc", 0.0)
                regional_info["avg_trust"] += self.trust_scores.get(self.consumers[i].id, 1.0)

                if getattr(self.consumers[i], "emergency_flag", False):
                    regional_info["num_emergency"] += 1

            regional_info["avg_soc"] /= num_consumers
            regional_info["avg_trust"] /= num_consumers

            shared_obs_dict = dict(zip(regional_obs_names[0], regional_observations[0]))
            regional_info["hour"] = shared_obs_dict.get("hour", 0.0)
            regional_info["electricity_pricing"] = shared_obs_dict.get("electricity_pricing", 0.0)

        self.total_demand = regional_info["total_demand"]
        self.history.append(self.total_demand)

        return np.array(list(regional_info.values()), dtype=np.float32)

    def select_action(self, obs, is_dr_needed):
        total_demand, num_emergency, avg_trust = obs
        
        if is_dr_needed:
            price_signal = 0.6
        else:
            threshold = 1500.0
            if total_demand > threshold:
                price_signal = 0.4
            else:
                price_signal = 0.1
        
        incentive_signal = 0.0
        if num_emergency > 0:
            incentive_signal = 0.2
        
        if avg_trust < 0.7:
            incentive_signal -= 0.1  
        
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
        peak_demand = np.max(all_regional_demands)
        avg_demand = np.mean(all_regional_demands)
        if avg_demand == 0:
            return 0.0
        return -(peak_demand / avg_demand)