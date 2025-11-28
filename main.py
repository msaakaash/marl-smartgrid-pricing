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
import torch
import re # <-- Used to find the latest checkpoint

# --- AGENT IMPORTS ---
from agents.consumer_agent import ConsumerAgentDQN

try:
    from agents.aggregator_agent import AggregatorAgentDDPG as AggregatorAgentImpl
    AGGREGATOR_IS_DDPG = True
    print("Successfully imported AggregatorAgentDDPG (MADDPG compatible).")
except ImportError as e:
    print(f"CRITICAL: Could not import AggregatorAgentDDPG ({e}). Exiting.")
    exit()


# --- CSV Columns and Helper Functions ---
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
    # (This function is unchanged)
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

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def find_latest_checkpoint(directory):
    """Finds the latest episode number from saved model files."""
    latest_episode = 0
    if not os.path.isdir(directory):
        return 0
    # Regex to find '..._ep[NUMBER].pth'
    pattern = re.compile(r'_ep(\d+)\.pth$')
    for f in os.listdir(directory):
        match = pattern.search(f)
        if match:
            episode_num = int(match.group(1))
            if episode_num > latest_episode:
                latest_episode = episode_num
    return latest_episode
# --- End of helper functions ---


def run_training_and_simulation(
    schema_paths=None,
    num_episodes: int = 50,
    use_ddpg_aggregator: bool = True,
    save_dir: str = ".",
    checkpoint_dir: str = "checkpoints",
    train_every_n_steps: int = 4
):
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
        exit()
    print("Aggregator implementation: {}".format("DDPG (MADDPG)" if use_ddpg_aggregator else "Heuristic/Other"))
    
    consumer_agents, consumer_metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents, use_ddpg=use_ddpg_aggregator)

    # --- DEFINE FILE PATHS ---
    csv_path = os.path.join(save_dir, "sample.csv")
    json_path = os.path.join(save_dir, "sample.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- CHECKPOINTING: Load models AND previous data ---
    latest_episode = find_latest_checkpoint(checkpoint_dir)
    start_episode = 1
    
    # These lists will hold ALL data from all runs
    all_step_records = []
    all_episodes_summary = []

    if latest_episode > 0:
        print(f"--- RESUMING from checkpoint: Episode {latest_episode} ---")
        start_episode = latest_episode + 1
        for agent in consumer_agents:
            agent.load_models(checkpoint_dir, latest_episode)
            agent.epsilon = agent.epsilon_min 
        for agent in aggregator_agents:
            agent.load_models(checkpoint_dir, latest_episode)
        
        # --- ROBUST LOADING: Load old CSV data ---
        try:
            old_df = pd.read_csv(csv_path)
            all_step_records = old_df.to_dict('records')
            print(f"Successfully loaded {len(all_step_records)} records from old CSV.")
        except FileNotFoundError:
            print("No old CSV found. Starting fresh.")
        except pd.errors.EmptyDataError:
            print("Old CSV was empty. Starting fresh.")
        except Exception as e:
            print(f"Error reading old CSV, starting fresh: {e}")

        # --- ROBUST LOADING: Load old JSON summary ---
        try:
            with open(json_path, 'r') as f:
                old_summary = json.load(f)
                all_episodes_summary = old_summary.get('episodes', [])
            print(f"Successfully loaded {len(all_episodes_summary)} summaries from old JSON.")
        except FileNotFoundError:
            print("No old JSON summary found.")
        except Exception as e:
            print(f"Error reading old JSON, starting fresh: {e}")
    else:
        print("--- STARTING new training run ---")
    
    # --- Main Training Loop ---
    for episode in range(start_episode, num_episodes + 1):
        
        # This list will only hold data for the CURRENT episode
        current_episode_step_records = [] 
        
        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}
        # (rest of reset logic is unchanged)
        for c in consumer_agents: c.reset_episode()
        for agg in aggregator_agents: agg.reset_episode()
        episode_demands = []
        last_agg_signals_dict: Dict[int, np.ndarray] = {c.id: np.array([0.0, 0.0]) for c in consumer_agents}
        current_consumer_states_dict: Dict[int, np.ndarray] = {}
        current_consumer_actions: Dict[int, float] = {c.id: 0.0 for c in consumer_agents}
        next_consumer_actions: Dict[int, float] = {c.id: 0.0 for c in consumer_agents}
        consumer_episode_stats = {c.id: {"reward_sum": 0.0, "steps": 0, "energy_sum": 0.0} for c in consumer_agents}
        aggregator_episode_stats = {agg.id: {"reward_sum": 0.0, "steps": 0, "total_demand": 0.0} for agg in aggregator_agents}

        print("Starting episode {} / {}".format(episode, num_episodes))

        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}"):
            
            # --- STEPS 1, 2, 3, 4, 4.5 (All Unchanged) ---
            # 1. GATHER STATES
            current_consumer_states_dict.clear()
            for c in consumer_agents:
                last_sig = last_agg_signals_dict.get(c.id, np.array([0.0, 0.0]))
                obs_18dim = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=last_sig
                )
                current_consumer_states_dict[c.id] = obs_18dim

            # 2. AGGREGATORS ACT
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

            # 3. CONSUMERS ACT
            all_actions = []
            current_consumer_actions.clear()
            for c in consumer_agents:
                new_targeted_signal = current_agg_signals_dict[c.id]
                obs_18dim_new = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=new_targeted_signal
                )
                action_value, action_index = c.select_action(obs_18dim_new)
                all_actions.append(action_value)
                current_consumer_actions[c.id] = float(action_value.item())

            # 4. ENV STEP
            next_observations, raw_rewards, done, info = env.step(all_actions)
            next_obs_dict = {i: next_observations[i] for i in range(len(next_observations))}
            current_total_demand = sum(b.net_electricity_consumption[-1] for b in env.buildings)
            episode_demands.append(current_total_demand)
            
            # 4.5 GET NEXT ACTIONS
            next_consumer_actions.clear()
            for c in consumer_agents:
                signal_for_next_state = current_agg_signals_dict[c.id]
                next_state_obs = c.get_observation(
                    next_obs_dict[c.id], building_obs_names[c.id], agg_signal=signal_for_next_state
                )
                original_epsilon = c.epsilon
                c.epsilon = 0.0
                next_action_val, _ = c.select_action(next_state_obs)
                c.epsilon = original_epsilon
                next_consumer_actions[c.id] = float(next_action_val.item())

            # 5. CONSUMER LEARNING (with speedup)
            action_idx = 0
            for agg in aggregator_agents:
                for c in agg.consumers:
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
                    if t % train_every_n_steps == 0:
                        c.learn()
                    
                    consumer_episode_stats[c.id]["reward_sum"] += float(custom_reward)
                    consumer_episode_stats[c.id]["steps"] += 1
                    consumer_episode_stats[c.id]["energy_sum"] += float(c.building.net_electricity_consumption[-1])

                    row = {
                        "episode": episode, "time_step": t, "consumer_id": c.id, "consumer_type": c_type,
                        "total_demand": float(current_total_demand),
                        "net_electricity_consumption": float(c.building.net_electricity_consumption[-1]),
                        "soc_after_action": soc_before, "agg_signal": agg_signal_val.tolist(),
                        "action": float(np.atleast_1d(action_value).item()), "reward": float(custom_reward)
                    }
                    # --- CHANGE: Save to current episode list ---
                    current_episode_step_records.append(row)
                    action_idx += 1

            # 6. AGGREGATOR LEARNING (with speedup)
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
                    consumer_actions_list = [current_consumer_actions[c.id] for c in agg.consumers]
                    next_consumer_states_list = []
                    for c in agg.consumers:
                        new_sig = current_agg_signals_dict[c.id]
                        next_obs = c.get_observation(
                            next_obs_dict[c.id], building_obs_names[c.id], agg_signal=new_sig
                        )
                        next_consumer_states_list.append(next_obs)
                    next_agg_state_90dim = agg.get_observation(next_consumer_states_list)
                    
                    agg.store_experience(
                        agg.last_state, agg.last_actions, consumer_actions_list,
                        agg_reward, next_agg_state_90dim, done
                    )
                    
                    actor_loss, critic_loss = None, None
                    if t % train_every_n_steps == 0:
                        next_consumer_actions_list = [next_consumer_actions[c.id] for c in agg.consumers]
                        next_consumer_actions_batch = torch.tensor(
                            np.array([next_consumer_actions_list] * agg.batch_size), 
                            dtype=torch.float32, device=agg.device
                        )
                        actor_loss, critic_loss = agg.learn(
                            all_next_consumer_actions_batch=next_consumer_actions_batch
                        )
                    
                    if t % 1000 == 0 and actor_loss is not None:
                        print(f"Step {t} | Agg {agg.id} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

            # 7. TRANSITION
            obs_dict = next_obs_dict
            last_agg_signals_dict = current_agg_signals_dict
        
        # --- END OF EPISODE ---
        
        # --- ROBUST SAVING: All saving logic is now INSIDE the episode loop ---
        
        # 1. Process episode summary
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
        
        # 2. Append new data to the main lists
        all_episodes_summary.append(episode_record)
        all_step_records.extend(current_episode_step_records)

        # 3. Save model checkpoints
        print(f"\n--- Saving checkpoint for Episode {episode} ---")
        for agent in consumer_agents:
            agent.save_models(checkpoint_dir, episode)
        for agent in aggregator_agents:
            agent.save_models(checkpoint_dir, episode)
        
        # 4. Overwrite the CSV and JSON files with the *complete* data
        try:
            # Save all step records
            all_df = pd.DataFrame(all_step_records)
            all_df = all_df.reindex(columns=CSV_COLUMNS)
            all_df.to_csv(csv_path, index=False)
            
            # Save all summary records
            save_json({"episodes": all_episodes_summary}, json_path)
            print(f"Successfully saved {csv_path} and {json_path} for episode {episode}.")
        except Exception as e:
            print(f"ERROR saving CSV/JSON for episode {episode}: {e}")

        # 5. Clean up old checkpoint
        if episode > 1:
            try:
                old_episode = episode - 1
                for f in os.listdir(checkpoint_dir):
                    if f.endswith(f"_ep{old_episode}.pth"):
                        os.remove(os.path.join(checkpoint_dir, f))
                print(f"Cleaned up checkpoint for Episode {old_episode}")
            except OSError as e:
                print(f"Error cleaning up old checkpoint: {e}")
                
        # (Print summary to console - unchanged)
        print("\nEpisode {} summary: total_energy={:.2f}, total_reward={:.2f}".format(
            episode, episode_total_energy, episode_total_reward
        ))
        for a in aggregators_summary:
            print(" - Aggregator {:02d}: avg_reward={:.4f}, avg_demand={:.2f}".format(
                a["aggregator_id"], a["avg_reward"], a["avg_demand"]
            ))
        # --- END OF EPISODE LOOP ---

    print(f"\nTraining complete. Final data saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except Exception: pass

    run_training_and_simulation(
        num_episodes=50, 
        use_ddpg_aggregator=True,
        train_every_n_steps=4 # <-- SPEEDUP parameter
    )