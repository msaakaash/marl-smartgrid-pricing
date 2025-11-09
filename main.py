# main.py
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Mapping, Any, Union, Dict
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building

# --- AGENT IMPORTS ---
from consumer_agent import ConsumerAgentDQN, ConsumerAgentRuleBased

# --- MADDPG CHANGE: Import the new DDPG Aggregator ---
try:
    from aggregator_agent import AggregatorAgentDDPG as AggregatorAgentImpl
    AGGREGATOR_IS_DDPG = True
    print("Successfully imported AggregatorAgentDDPG (MADDPG compatible).")
except ImportError as e:
    print(f"CRITICAL: Could not import AggregatorAgentDDPG ({e}). Exiting.")
    exit()

# (CSV Columns and helper functions are unchanged)
CSV_COLUMNS = [
    "episode", "time_step", "consumer_id", "consumer_type", "total_demand",
    "net_electricity_consumption", "soc_after_action", "agg_signal", "action", "reward"
]
def build_consumers_from_env(env, building_obs_names):
    # (This function is unchanged)
    consumer_agents = []
    consumer_metadata_map = {}
    for i, building in enumerate(env.buildings):
        metadata = {
            "building_type": random.choice(["residential", "office", "mall", "hospital", "school"]),
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
        agent.building_type = metadata["building_type"]
        consumer_agents.append(agent)
        consumer_metadata_map[i] = metadata
    print(f"Built {len(consumer_agents)} consumer agents.")
    return consumer_agents, consumer_metadata_map

def build_aggregators(consumer_agents, use_ddpg=True):
    # (This function is unchanged from our last version)
    aggregators = []
    num_regions = max(1, (len(consumer_agents) + 4) // 5)
    print(f"Building {num_regions} aggregator regions...")
    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5
        consumers_slice = consumer_agents[start:end]
        if AGGREGATOR_IS_DDPG and use_ddpg:
            agg = AggregatorAgentImpl(region_id=region_id, consumers=consumers_slice)
        else:
            print("CRITICAL: DDPG agent requested or fallback not available. Exiting.")
            exit()
        aggregators.append(agg)
    return aggregators
def record_step_csv(step_rows, csv_path):
    df = pd.DataFrame(step_rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
# (End of unchanged helper functions)


def run_training_and_simulation(
    schema_paths=None,
    num_episodes: int = 50, # Set back to 50
    use_ddpg_aggregator: bool = True,
    grid_virtual_capacity: float = 10.0,
    dr_threshold_factor: float = 0.85,
    save_dir: str = ".",
):
    # (Environment loading is unchanged)
    if schema_paths is None:
        schema_paths = ["citylearn_challenge_2022_phase_1", "citylearn_challenge_2022_phase_2", "citylearn_challenge_2022_phase_3"]
    buildings: List[Building] = []
    building_obs_names: Mapping[int, Union[List[str], Any]] = {}
    for schema_path in schema_paths:
        env_schema = CityLearnEnv(schema=schema_path)
        for i, bld in enumerate(env_schema.buildings):
            if len(buildings) >= 15: break
            buildings.append(bld)
            obs_names = env_schema.observation_names[i]
            if obs_names is None or not isinstance(obs_names, list):
                obs_names = [f"obs_{j}" for j in range(len(bld.observation_space.sample()))]
            building_obs_names[len(buildings) - 1] = obs_names
        if len(buildings) >= 15: break
    env = CityLearnEnv(schema=schema_paths[0], buildings=buildings)
    print("Environment loaded: {} buildings".format(len(env.buildings)))
    
    if use_ddpg_aggregator and not AGGREGATOR_IS_DDPG:
        print("CRITICAL: Requested DDPG, but import failed.")
        st.stop()
    print("Aggregator implementation: {}".format("DDPG (MADDPG)" if use_ddpg_aggregator else "Heuristic/Other"))
    
    # (Build agents is unchanged)
    consumer_agents, consumer_metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents, use_ddpg=use_ddpg_aggregator)

    step_records = []
    episodes_summary = []

    # --- Main Training Loop ---
    for episode in range(1, num_episodes + 1):
        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        for c in consumer_agents: c.reset_episode()
        for agg in aggregator_agents: agg.reset_episode()

        episode_demands = []
        episode_reward_sum = 0.0
        
        last_agg_signals_dict: Dict[int, np.ndarray] = {c.id: np.array([0.0, 0.0]) for c in consumer_agents}
        current_consumer_states_dict: Dict[int, np.ndarray] = {}
        
        # --- MADDPG CHANGE: Dictionaries to store consumer actions ---
        current_consumer_actions: Dict[int, float] = {c.id: 0.0 for c in consumer_agents}
        next_consumer_actions: Dict[int, float] = {c.id: 0.0 for c in consumer_agents}

        consumer_episode_stats = {c.id: {"reward_sum": 0.0, "steps": 0, "energy_sum": 0.0} for c in consumer_agents}
        aggregator_episode_stats = {agg.id: {"reward_sum": 0.0, "steps": 0, "total_demand": 0.0} for agg in aggregator_agents}

        print("Starting episode {} / {}".format(episode, num_episodes))

        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}"):
            
            # --- STEP 1: GATHER ALL CONSUMER STATES (Unchanged) ---
            current_consumer_states_dict.clear()
            for c in consumer_agents:
                last_sig = last_agg_signals_dict.get(c.id, np.array([0.0, 0.0]))
                obs_18dim = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=last_sig
                )
                current_consumer_states_dict[c.id] = obs_18dim

            # --- STEP 2: AGGREGATORS OBSERVE AND ACT (Unchanged) ---
            current_agg_signals_dict: Dict[int, np.ndarray] = {}
            for agg in aggregator_agents:
                consumer_states_list = [current_consumer_states_dict[c.id] for c in agg.consumers]
                agg_state_90dim = agg.get_observation(consumer_states_list)
                agg_action_10dim = agg.select_action(agg_state_90dim)
                agg.last_state = agg_state_90dim
                agg.last_actions = agg_action_10dim
                for i, consumer in enumerate(agg.consumers):
                    targeted_signal = agg_action_10dim[i*2 : i*2 + 2]
                    current_agg_signals_dict[consumer.id] = targeted_signal

            # --- STEP 3: CONSUMERS OBSERVE AND ACT (UPGRADED) ---
            all_actions = []
            current_consumer_actions.clear() # Clear for this step
            for c in consumer_agents:
                new_targeted_signal = current_agg_signals_dict[c.id]
                obs_18dim_new = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=new_targeted_signal
                )
                action_value, action_index = c.select_action(obs_18dim_new)
                
                all_actions.append(action_value)
                # --- MADDPG CHANGE: Store the consumer's chosen action ---
                current_consumer_actions[c.id] = float(action_value.item())

            # --- STEP 4: STEP THE ENVIRONMENT (Unchanged) ---
            next_observations, raw_rewards, done, info = env.step(all_actions)
            next_obs_dict = {i: next_observations[i] for i in range(len(next_observations))}
            current_total_demand = sum(b.net_electricity_consumption[-1] for b in env.buildings)
            episode_demands.append(current_total_demand)
            
            # --- MADDPG CHANGE: STEP 4.5: GET *NEXT* CONSUMER ACTIONS ---
            # We need these for the Critic's target calculation.
            next_consumer_actions.clear()
            for c in consumer_agents:
                signal_for_next_state = current_agg_signals_dict[c.id]
                next_state_obs = c.get_observation(
                    next_obs_dict[c.id],
                    building_obs_names[c.id],
                    agg_signal=signal_for_next_state
                )
                
                # Get deterministic action (no exploration) for the target
                original_epsilon = c.epsilon
                c.epsilon = 0.0 # Turn off exploration
                next_action_val, _ = c.select_action(next_state_obs)
                c.epsilon = original_epsilon # Turn exploration back on
                
                next_consumer_actions[c.id] = float(next_action_val.item())

            # --- STEP 5: CONSUMER LEARNING (Unchanged) ---
            action_idx = 0
            for agg in aggregator_agents:
                for c in agg.consumers:
                    # (This whole block is unchanged)
                    c_type = c.building_type
                    soc_before = float(c.building.electrical_storage.soc[-1]) if len(c.building.electrical_storage.soc) > 0 else 0.0
                    current_price = float(obs_dict[c.id][10]) if len(obs_dict[c.id]) > 10 else 0.0
                    agg_signal_val = current_agg_signals_dict[c.id]
                    action_value = all_actions[action_idx]
                    raw_reward = raw_rewards[action_idx]
                    custom_reward = c.get_reward(
                        raw_reward, c.building.net_electricity_consumption[-1],
                        action_value, soc_before, current_price, agg_signal_val
                    )
                    next_state = c.get_observation(
                        next_obs_dict[c.id], building_obs_names[c.id], agg_signal=agg_signal_val
                    )
                    c.store_experience(c.last_state, c.last_action_index, custom_reward, next_state, done)
                    c.learn()
                    
                    consumer_episode_stats[c.id]["reward_sum"] += float(custom_reward)
                    consumer_episode_stats[c.id]["steps"] += 1
                    consumer_episode_stats[c.id]["energy_sum"] += float(c.building.net_electricity_consumption[-1])

                    row = {
                        "episode": episode, "time_step": t, "consumer_id": c.id, "consumer_type": c_type,
                        "total_demand": float(current_total_demand),
                        "net_electricity_consumption": float(c.building.net_electricity_consumption[-1]),
                        "soc_after_action": soc_before,
                        "agg_signal": agg_signal_val.tolist(),
                        "action": float(np.atleast_1d(action_value).item()),
                        "reward": float(custom_reward)
                    }
                    step_records.append(row)
                    action_idx += 1

            # --- STEP 6: AGGREGATOR LEARNING (UPGRADED) ---
            regional_demands_by_agg = []
            for agg in aggregator_agents:
                regs = [c.building.net_electricity_consumption[-1] for c in agg.consumers]
                regional_demands_by_agg.append(np.sum(regs))
            
            for agg_idx, agg in enumerate(aggregator_agents):
                agg_reward = agg.get_reward([regional_demands_by_agg[agg_idx]])
                
                aggregator_episode_stats[agg.id]["reward_sum"] += float(agg_reward)
                aggregator_episode_stats[agg.id]["steps"] += 1
                aggregator_episode_stats[agg.id]["total_demand"] += float(regional_demands_by_agg[agg_idx])

                if use_ddpg_aggregator:
                    # --- MADDPG CHANGE: Collect the 5 consumer actions for this agg ---
                    consumer_actions_list = [current_consumer_actions[c.id] for c in agg.consumers]
                    
                    # Build the next 90-dim state (unchanged)
                    next_consumer_states_list = []
                    for c in agg.consumers:
                        new_sig = current_agg_signals_dict[c.id]
                        next_obs = c.get_observation(
                            next_obs_dict[c.id], building_obs_names[c.id], agg_signal=new_sig
                        )
                        next_consumer_states_list.append(next_obs)
                    next_agg_state_90dim = agg.get_observation(next_consumer_states_list)
                    
                    # --- MADDPG CHANGE: Store the consumer actions ---
                    agg.store_experience(
                        agg.last_state, 
                        agg.last_actions, 
                        consumer_actions_list, # <-- NEW
                        agg_reward, 
                        next_agg_state_90dim, 
                        done
                    )
                    
                    # --- MADDPG CHANGE: Pass next actions to learn() ---
                    next_consumer_actions_list = [next_consumer_actions[c.id] for c in agg.consumers]
                    
                    # Create a "batch" of these next actions for the Critic
                    next_consumer_actions_batch = torch.tensor(
                        np.array([next_consumer_actions_list] * agg.batch_size), 
                        dtype=torch.float32, device=agg.device
                    )

                    actor_loss, critic_loss = agg.learn(
                        all_next_consumer_actions_batch=next_consumer_actions_batch
                    )
                    
                    if t % 1000 == 0 and actor_loss is not None:
                        print(f"Step {t} | Agg {agg.id} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

            # --- STEP 7: TRANSITION (Unchanged) ---
            obs_dict = next_obs_dict
            last_agg_signals_dict = current_agg_signals_dict

        # --- End of Episode Summary (Unchanged) ---
        episode_total_energy = float(np.sum(episode_demands)) if episode_demands else 0.0
        episode_total_reward = sum(v["reward_sum"] for v in consumer_episode_stats.values())
        consumers_summary = []
        for cid, stats in consumer_episode_stats.items():
            avg_reward = float(stats["reward_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            avg_energy = float(stats["energy_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            consumers_summary.append({
                "consumer_id": cid,
                "building_type": consumer_metadata_map.get(cid, {}).get("building_type", "unknown"),
                "avg_reward": avg_reward, "avg_energy": avg_energy
            })
        aggregators_summary = []
        for aid, stats in aggregator_episode_stats.items():
            avg_reward = float(stats["reward_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            avg_demand = float(stats["total_demand"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            aggregators_summary.append({
                "aggregator_id": aid, "avg_reward": avg_reward,
                "avg_demand": avg_demand, "dr_triggers": 0
            })
        episode_record = {
            "episode": episode, "total_energy": episode_total_energy, "total_reward": episode_total_reward,
            "consumers": consumers_summary, "aggregators": aggregators_summary
        }
        episodes_summary.append(episode_record)
        print("\nEpisode {} summary: total_energy={:.2f}, total_reward={:.2f}".format(
            episode, episode_total_energy, episode_total_reward
        ))
        for a in aggregators_summary:
            print(" - Aggregator {:02d}: avg_reward={:.4f}, avg_demand={:.2f}".format(
                a["aggregator_id"], a["avg_reward"], a["avg_demand"]
            ))

    # --- After all episodes: save CSV and JSON (Unchanged) ---
    csv_path = os.path.join(save_dir, "training_step_data.csv")
    json_path = os.path.join(save_dir, "training_summary.json")
    if len(step_records) > 0:
        df = pd.DataFrame(step_records)
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    save_json({"episodes": episodes_summary}, json_path)
    print(f"\nTraining complete. Saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except Exception: pass

    run_training_and_simulation(
        num_episodes=50, # <-- Run for 50 episodes
        use_ddpg_aggregator=True
    )