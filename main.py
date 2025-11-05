import numpy as np
import random
import torch
from tqdm import tqdm
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building
from typing import List, Mapping, Any, Union
import json
import pandas as pd

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

    # ✅ Load up to 15 buildings
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
    print(f"✅ Combined environment loaded with {len(env.buildings)} buildings.\n")

    # ✅ Training hyperparameters
    NUM_EPISODES = 5
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = 200
    GRID_VIRTUAL_CAPACITY = 10.0
    DR_THRESHOLD_FACTOR = 0.85

    marl_results = {'total_reward': [], 'total_energy': [], 'peak_to_average_ratio': []}
    marl_load_profile = []
    step_data = []  # For CSV output

    # ✅ Initialize Consumer Agents
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

    # ✅ Initialize Aggregator Agents (3 regions × 5 consumers each)
    aggregator_agents = []
    for i in range(3):
        start_index = i * 5
        end_index = start_index + 5
        aggregator_consumers = consumer_agents[start_index:end_index]
        aggregator = AggregatorAgent(region_id=i, consumers=aggregator_consumers)
        aggregator_agents.append(aggregator)

    # ✅ Main training loop
    for episode in range(NUM_EPISODES):
        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}
        for agent in consumer_agents:
            agent.reset_episode()

        episode_demands = []
        episode_total_reward = 0.0

        print(f"\n🚀 Episode {episode + 1}/{NUM_EPISODES} starting...")

        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode+1} Training"):
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
            current_total_demand = sum(c.net_electricity_consumption[-1] for c in env.buildings)
            episode_demands.append(current_total_demand)
            if episode == NUM_EPISODES - 1:
                marl_load_profile.append(current_total_demand)

            for i, consumer_agent in enumerate(consumer_agents):
                # ✅ Safety patch — in case metadata missing
                if not hasattr(consumer_agent, "metadata"):
                    consumer_agent.metadata = {"building_type": "unknown"}

                soc_before_action = float(
                    consumer_agent.building.electrical_storage.soc[-1]
                    if len(consumer_agent.building.electrical_storage.soc) > 0 else 0.0
                )
                current_price = obs_dict[i][10]
                agg_signal_val = np.array([0.0, 0.0])
                for agg in aggregator_agents:
                    if consumer_agent in agg.consumers:
                        agg_signal_val = agg.last_signal
                        break
                custom_reward = consumer_agent.get_reward(
                    raw_reward=rewards[i],
                    current_power_demand=consumer_agent.building.net_electricity_consumption[-1],
                    action_value=all_actions[i],
                    soc_before_action=soc_before_action,
                    electricity_pricing=current_price,
                    agg_signal=agg_signal_val
                )

                next_state_for_store = consumer_agent.get_observation(
                    next_obs_dict[consumer_agent.id],
                    building_obs_names[consumer_agent.id],
                    agg_signal=agg_signal_val
                )

                consumer_agent.store_experience(
                    consumer_agent.last_state,
                    consumer_agent.last_action_index,
                    custom_reward,
                    next_state_for_store,
                    done
                )
                loss = consumer_agent.learn()
                episode_total_reward += custom_reward

                # ✅ Record for CSV
                step_data.append({
                    "episode": episode + 1,
                    "time_step": t,
                    "consumer_id": consumer_agent.id,
                    "consumer_type": consumer_agent.metadata["building_type"],
                    "total_demand": float(current_total_demand),
                    "net_electricity_consumption": float(consumer_agent.building.net_electricity_consumption[-1]),
                    "soc_after_action": soc_before_action,
                    "agg_signal": agg_signal_val.tolist(),
                    "action": float(all_actions[i]),
                    "reward": float(custom_reward)
                })

            if t % TARGET_UPDATE_FREQ == 0:
                for agent in consumer_agents:
                    agent.update_target_network()

            observations = next_observations
            obs_dict = next_obs_dict

        final_demands = np.array(episode_demands)
        peak_demand = np.max(final_demands) if len(final_demands) > 0 else 0.0
        avg_demand = np.mean(final_demands) if len(final_demands) > 0 else 1.0
        peak_to_average_ratio = peak_demand / avg_demand

        marl_results['total_reward'].append(episode_total_reward)
        marl_results['total_energy'].append(np.sum(final_demands))
        marl_results['peak_to_average_ratio'].append(peak_to_average_ratio)

        # ✅ Episode Summary
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   • Total Reward: {episode_total_reward:.2f}")
        print(f"   • Total Energy: {np.sum(final_demands):.2f}")
        print(f"   • Peak/Avg Ratio: {peak_to_average_ratio:.2f}")
        print(f"   • Final Epsilon (Agent 0): {consumer_agents[0].epsilon:.4f}")

    print("\n🎯 MARL training complete.")

    # ✅ Save JSON Results
    final_results = {"marl_training_history": marl_results}
    with open('simulation_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    # ✅ Save Step CSV
    df = pd.DataFrame(step_data)
    df.to_csv("training_step_data.csv", index=False)
    print("\n📁 Saved 'simulation_results.json' and 'training_step_data.csv' successfully.")


if __name__ == "__main__":
    run_training_and_simulation()
