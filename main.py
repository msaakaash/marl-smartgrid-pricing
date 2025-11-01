import numpy as np
import random
import torch
from tqdm import tqdm
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building
from typing import List, Mapping, Any, Union
import json

from agents.consumer_agent import ConsumerAgentDQN
from agents.aggregator_agent import AggregatorAgent


def run_training_and_simulation():
    schema_paths = [
        'citylearn_challenge_2022_phase_1',
        'citylearn_challenge_2022_phase_2',
        'citylearn_challenge_2022_phase_3'
    ]

    buildings: List[Building] = []
    building_obs_names: Mapping[int, Union[List[str], Any]] = {}

    for schema_path in schema_paths:
        env_schema = CityLearnEnv(schema=schema_path)
        for i, bld in enumerate(env_schema.buildings):
            if len(buildings) >= 15:
                break
            buildings.append(bld)
            
            obs_names = env_schema.observation_names[i]
            if obs_names is None or not isinstance(obs_names, list):
                obs_names = [f"obs_{j}" for j in range(len(bld.observation_space.sample()))]

            building_obs_names[len(buildings) - 1] = obs_names
        
        if len(buildings) >= 15:
            break
    
    env = CityLearnEnv(schema=schema_paths[0], buildings=buildings)
    print(f"Combined environment loaded with {len(env.buildings)} buildings.")

    NUM_EPISODES = 10
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    GRID_VIRTUAL_CAPACITY = 10.0
    DR_THRESHOLD_FACTOR = 0.85
    
    marl_results = {
        'total_reward': [],
        'total_energy': [],
        'peak_to_average_ratio': []
    }
    
    marl_load_profile = []

    consumer_agents = []
    for i, building in enumerate(env.buildings):
        metadata = {
            "building_type": random.choice(["residential", "office", "mall"]),
            "critical_load_fraction": random.uniform(0.1, 0.3),
            "prime_hours": random.sample(range(8, 20), 3),
            "cheating_propensity": random.uniform(0.0, 0.3),
            "emergency_flag": False
        }
        agent = ConsumerAgentDQN(
            building=building,
            building_id=i,
            metadata=metadata,
            action_space=building.action_space
        )
        consumer_agents.append(agent)

    aggregator_agents = []
    for i in range(3):
        start_index = i * 5
        end_index = start_index + 5
        aggregator_consumers = consumer_agents[start_index:end_index]
        aggregator = AggregatorAgent(region_id=i, consumers=aggregator_consumers)
        aggregator_agents.append(aggregator)
        
    for episode in range(NUM_EPISODES):
        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}
        for agent in consumer_agents:
            agent.reset_episode()
        
        episode_demands = []
        episode_total_reward = 0.0
        
        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}...")

        for t in tqdm(range(env.time_steps - 1), desc="MARL-iDR Training"):
            all_actions = []
            total_grid_demand = sum(agg.total_demand for agg in aggregator_agents)
            is_dr_needed = total_grid_demand > (GRID_VIRTUAL_CAPACITY * DR_THRESHOLD_FACTOR)

            for agg in aggregator_agents:
                regional_consumer_obs = [obs_dict[c.id] for c in agg.consumers]
                regional_obs_names = [building_obs_names[c.id] for c in agg.consumers]
                agg_signal = agg.select_action(
                    agg.get_observation(regional_consumer_obs, regional_obs_names), is_dr_needed
                )
                for consumer_agent in agg.consumers:
                    current_state = consumer_agent.get_observation(
                        obs_dict[consumer_agent.id], building_obs_names[consumer_agent.id], agg_signal=agg_signal
                    )
                    action_value, action_index = consumer_agent.select_action(current_state)
                    all_actions.append(action_value)

            next_observations, rewards, done, info = env.step(all_actions)
            next_obs_dict = {i: next_observations[i] for i in range(len(next_observations))}
            
            terminated = done
            truncated = False
            
            current_total_demand = sum(c.net_electricity_consumption[-1] for c in env.buildings)
            episode_demands.append(current_total_demand)
            if episode == NUM_EPISODES - 1:
                marl_load_profile.append(current_total_demand)

            for i, consumer_agent in enumerate(consumer_agents):
                soc_before_action = float(consumer_agent.building.electrical_storage.soc[-1] if len(consumer_agent.building.electrical_storage.soc) > 0 else 0.0)
                current_price = obs_dict[i][10]
                agg_signal_val = np.array([0.0, 0.0])
                for agg in aggregator_agents:
                    if consumer_agent in agg.consumers:
                        agg_signal_val = agg.last_signal
                        break
                custom_reward = consumer_agent.get_reward(
                    raw_reward=rewards[i], current_power_demand=consumer_agent.building.net_electricity_consumption[-1],
                    action_value=all_actions[i], soc_before_action=soc_before_action,
                    electricity_pricing=current_price, agg_signal=agg_signal_val
                )
                next_state_for_store = consumer_agent.get_observation(
                    next_obs_dict[consumer_agent.id], building_obs_names[consumer_agent.id], agg_signal=agg_signal_val
                )
                consumer_agent.store_experience(
                    consumer_agent.last_state, consumer_agent.last_action_index, custom_reward, next_state_for_store, done
                )
                loss = consumer_agent.learn()
                if loss is not None and t % 500 == 0:
                    print(f"  --> Agent {i} at t={t}: Loss={loss:.4f}")
                episode_total_reward += custom_reward

            if t % TARGET_UPDATE_FREQ == 0:
                for agent in consumer_agents:
                    agent.update_target_network()

            observations = next_observations
            obs_dict = next_obs_dict

        final_demands = np.array(episode_demands)
        if len(final_demands) > 0:
            peak_demand = np.max(final_demands)
            avg_demand = np.mean(final_demands)
            peak_to_average_ratio = peak_demand / avg_demand
        else:
            peak_to_average_ratio = 0.0

        marl_results['total_reward'].append(episode_total_reward)
        marl_results['total_energy'].append(np.sum(final_demands))
        marl_results['peak_to_average_ratio'].append(peak_to_average_ratio)

        print(f"  Episode {episode + 1} finished.")
        print(f"  Final Epsilon (Agent 0): {consumer_agents[0].epsilon:.4f}")
        print(f"  Total Episode Reward: {episode_total_reward:.2f}")
        print(f"  Total Episode Energy Consumption: {np.sum(final_demands):.2f}")
        print(f"  Peak-to-Average Ratio: {peak_to_average_ratio:.2f}")

    print("\nRL training complete.")
    
    final_results = {
        "marl_training_history": marl_results
    }
    with open('simulation_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)


if __name__ == "__main__":
    run_training_and_simulation()