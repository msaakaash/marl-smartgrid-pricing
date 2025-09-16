import numpy as np

def calculate_reward(cost_savings, comfort_level, peak_load_reduction):
   
    w1_cost = 0.5
    w2_comfort = 0.3
    w3_peak = 0.2

    # A simple, linear combination of objectives
    return (w1_cost * cost_savings) + (w2_comfort * comfort_level) + (w3_peak * peak_load_reduction)

def calculate_consumer_reward(old_cost, new_cost, comfort_factor):
    """Calculates the reward for a single consumer."""
    cost_savings = (old_cost - new_cost) / old_cost if old_cost > 0 else 0
    comfort_level = comfort_factor
    return calculate_reward(cost_savings, comfort_level, 0) 

def calculate_grid_reward(old_peak_demand, new_peak_demand):
    """Calculates the reward for the grid controller."""
    peak_load_reduction = (old_peak_demand - new_peak_demand) / old_peak_demand if old_peak_demand > 0 else 0
    return calculate_reward(0, 0, peak_load_reduction) 