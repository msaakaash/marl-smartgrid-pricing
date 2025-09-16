from citylearn.citylearn import CityLearnEnv
import numpy as np
import os
from profiles import generate_profiles
from rewards import calculate_consumer_reward, calculate_grid_reward

class MARLWrapper:
    def __init__(self):
        # Using a dummy CityLearn dataset for the prototype
        # The folder 'citylearn_challenge_2022_phase_1' should exist inside the citylearn package folder
        data_path = "citylearn_challenge_2022_phase_1"
        self.env = CityLearnEnv(data_path)
        
        # Dynamically get the number of buildings from the environment
        self.num_consumers = len(self.env.buildings)
        
        # Generate profiles for all consumers
        self.consumer_profiles = generate_profiles(self.num_consumers)
        self.state = self.env.reset()
        self.total_consumption = 0
        self.step_count = 0

    def step(self):
        # Prototype of the three-layer agent interaction

        # 1. Grid Controller observes the environment and decides on DR signals
        aggregated_demand = np.sum(self.state[0])
        dr_signal_type = "hybrid" if aggregated_demand > 20 else "none"

        # 2. Aggregator Agents (simulated) process the signal and decide on prices/incentives
        custom_dr_offers = {}
        for profile in self.consumer_profiles:
            if dr_signal_type == "hybrid":
                # A simple rule for the prototype
                price_signal = 0.5 + (0.5 * profile.flexibility)
                incentive = 1.0 - profile.willingness
                custom_dr_offers[profile.id] = (price_signal, incentive)
            else:
                custom_dr_offers[profile.id] = (0, 0)

        # 3. Consumer Agents take actions based on the custom offers and their profiles
        actions = []
        for profile in self.consumer_profiles:
            price_signal, incentive = custom_dr_offers[profile.id]
            # Simplified agent logic for the prototype
            action = -1 * (price_signal + incentive) * profile.willingness
            # WRAP THE ACTION IN A LIST
            actions.append([action]) 
        
        # The wrapper passes these actions to the CityLearn environment
        next_state, old_cost, _, _ = self.env.step(np.array(actions))
        new_state, new_cost, _, _ = self.env.step(np.array(actions))
        
        # Calculate rewards for agents
        consumer_rewards = [calculate_consumer_reward(old_cost[i], new_cost[i], 1.0) for i in range(self.num_consumers)]
        
        # Calculate grid reward based on total demand reduction
        old_peak_demand = np.sum(old_cost)
        new_peak_demand = np.sum(new_cost)
        grid_reward = calculate_grid_reward(old_peak_demand, new_peak_demand)
        
        self.state = next_state
        self.step_count += 1
        
        return new_state, consumer_rewards, grid_reward, dr_signal_type, new_peak_demand

    def reset(self):
        self.state = self.env.reset()
        self.step_count = 0
        self.total_consumption = 0
        return self.state