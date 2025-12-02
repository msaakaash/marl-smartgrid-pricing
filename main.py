# ================================================================
#   main.py  (FINAL – Secure MARL with ChaCha20-Poly1305)
# ================================================================

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
import re

# --- IMPORT AGENTS ---
from agents.consumer_agent import ConsumerAgentDQN

try:
    from agents.aggregator_agent import AggregatorAgentDDPG as AggregatorAgentImpl
    AGGREGATOR_IS_DDPG = True
    print("Successfully imported AggregatorAgentDDPG.")
except ImportError as e:
    print(f"CRITICAL: Could not import AggregatorAgentDDPG ({e}). Exiting.")
    exit()


# ================================================================
#                   CONSTANTS AND HELPERS
# ================================================================

CSV_COLUMNS = [
    "episode", "time_step", "consumer_id", "consumer_type", "total_demand",
    "net_electricity_consumption", "soc_after_action", "agg_signal",
    "action", "reward"
]


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def find_latest_checkpoint(directory):
    latest_episode = 0
    if not os.path.isdir(directory):
        return 0

    pattern = re.compile(r'_ep(\d+)\.pth$')
    for f in os.listdir(directory):
        match = pattern.search(f)
        if match:
            episode_num = int(match.group(1))
            latest_episode = max(latest_episode, episode_num)
    return latest_episode


# ================================================================
#                BUILD CONSUMER AGENTS
# ================================================================

def build_consumers_from_env(env, building_obs_names):
    consumer_agents = []
    consumer_metadata_map = {}

    for i, building in enumerate(env.buildings):

        metadata = {
            "building_type": random.choice(
                ["residential", "office", "mall", "hospital", "school"]),
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
        consumer_metadata_map[i] = metadata

    print(f"Built {len(consumer_agents)} consumer agents.")
    return consumer_agents, consumer_metadata_map


# ================================================================
#                BUILD AGGREGATOR AGENTS
# ================================================================

def build_aggregators(consumer_agents, use_ddpg=True):

    aggregators = []
    num_regions = max(1, (len(consumer_agents) + 4) // 5)

    print(f"Building {num_regions} aggregator regions...")

    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5
        slice_group = consumer_agents[start:end]

        if AGGREGATOR_IS_DDPG:
            agg = AggregatorAgentImpl(region_id=region_id, consumers=slice_group)
        else:
            print("CRITICAL: Aggregator init failed.")
            exit()

        aggregators.append(agg)

    return aggregators


# ================================================================
#              TRAINING AND SIMULATION PIPELINE
# ================================================================

def run_training_and_simulation(
    schema_paths=None,
    num_episodes=10,
    use_ddpg_aggregator=True,
    save_dir=".",
    checkpoint_dir="checkpoints",
    train_every_n_steps=4
):

    if schema_paths is None:
        schema_paths = [
            "citylearn_challenge_2022_phase_1",
            "citylearn_challenge_2022_phase_2",
            "citylearn_challenge_2022_phase_3"
        ]

    # ------------------------- LOAD ENVIRONMENT -------------------------
    buildings = []
    building_obs_names = {}

    for schema_path in schema_paths:
        env_schema = CityLearnEnv(schema=schema_path)

        for i, bld in enumerate(env_schema.buildings):
            if len(buildings) >= 15:
                break

            buildings.append(bld)
            obs_names = env_schema.observation_names[i]
            if not isinstance(obs_names, list):
                obs_names = [f"obs_{j}" for j in range(len(bld.observation_space.sample()))]

            building_obs_names[len(buildings) - 1] = obs_names

        if len(buildings) >= 15:
            break

    env = CityLearnEnv(schema=schema_paths[0], buildings=buildings)
    print(f"Environment loaded with {len(env.buildings)} buildings.")

    # ------------------------- AGENTS -------------------------
    consumer_agents, consumer_metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents, use_ddpg_aggregator)

    # ------------------------- SAVE PATHS -------------------------
    csv_path = os.path.join(save_dir, "sample.csv")
    json_path = os.path.join(save_dir, "sample.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------- CHECKPOINTING -------------------------
    latest_episode = find_latest_checkpoint(checkpoint_dir)
    start_episode = latest_episode + 1 if latest_episode > 0 else 1

    all_step_records = []
    all_episodes_summary = []

    if latest_episode > 0:
        print(f"Resuming from episode {latest_episode}")

        # Load models
        for c in consumer_agents:
            c.load_models(checkpoint_dir, latest_episode)
            c.epsilon = c.epsilon_min

        for a in aggregator_agents:
            a.load_models(checkpoint_dir, latest_episode)

        # Load CSV/JSON data
        try:
            df_old = pd.read_csv(csv_path)
            all_step_records = df_old.to_dict("records")
        except:
            pass

        try:
            with open(json_path) as f:
                summary_old = json.load(f)
                all_episodes_summary = summary_old.get("episodes", [])
        except:
            pass

    # ================================================================
    #                       MAIN TRAINING LOOP
    # ================================================================

    for episode in range(start_episode, num_episodes + 1):

        current_episode_rows = []

        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        for c in consumer_agents:
            c.reset_episode()

        for agg in aggregator_agents:
            agg.reset_episode()

        last_agg_signals = {c.id: np.array([0.0, 0.0]) for c in consumer_agents}

        print(f"\n==================  EPISODE {episode}  ==================")

        # -----------------------------------------------------------
        #                TIMESTEP LOOP
        # -----------------------------------------------------------

        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}"):

            current_consumer_states = {}

            # -----------------------------------------------------------
            # 1. Build consumer states
            # -----------------------------------------------------------
            for c in consumer_agents:
                state = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    last_agg_signals[c.id]
                )
                current_consumer_states[c.id] = state

            # -----------------------------------------------------------
            # 2. Aggregators act → produce raw 10-dim vector
            # -----------------------------------------------------------
            encrypted_packets_for_consumers = {}

            for agg in aggregator_agents:

                state_list = [current_consumer_states[c.id] for c in agg.consumers]
                agg_state_90 = agg.get_observation(state_list)

                raw_action_10dim = agg.select_action(agg_state_90)
                agg.last_state = agg_state_90
                agg.last_actions = raw_action_10dim

                # -------------------------------------------------------
                #  SECURE PHASE → encryption for each consumer
                # -------------------------------------------------------
                encrypted_packets = agg.encrypt_signals_for_consumers(raw_action_10dim)

                for cid, packet in encrypted_packets.items():
                    encrypted_packets_for_consumers[cid] = packet

            # -----------------------------------------------------------
            # 3. Consumers decrypt + act
            # -----------------------------------------------------------
            all_actions = []
            consumer_actions_scalar = {}

            for c in consumer_agents:

                # Decrypt the targeted signal
                decrypted_signal = c.decrypt_signal(encrypted_packets_for_consumers[c.id])

                state = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    decrypted_signal
                )

                action_value, _ = c.select_action(state)
                action_scalar = float(action_value.item())

                all_actions.append([action_scalar])
                consumer_actions_scalar[c.id] = action_scalar
                last_agg_signals[c.id] = decrypted_signal

            # -----------------------------------------------------------
            # 4. Environment step
            # -----------------------------------------------------------
            next_obs, raw_rewards, done, info = env.step(all_actions)
            next_obs_dict = {i: next_obs[i] for i in range(len(next_obs))}

            # -----------------------------------------------------------
            # 5. Consumer Learning
            # -----------------------------------------------------------
            action_idx = 0
            for agg in aggregator_agents:
                for c in agg.consumers:

                    reward_custom = c.get_reward(
                        raw_rewards[action_idx],
                        c.building.net_electricity_consumption[-1],
                        consumer_actions_scalar[c.id],
                        float(c.building.electrical_storage.soc[-1])
                        if len(c.building.electrical_storage.soc) else 0.0,
                        float(obs_dict[c.id][10]) if len(obs_dict[c.id]) > 10 else 0.0,
                        last_agg_signals[c.id]
                    )

                    next_state = c.get_observation(
                        next_obs_dict[c.id], building_obs_names[c.id], last_agg_signals[c.id]
                    )

                    c.store_experience(c.last_state, c.last_action_index,
                                       reward_custom, next_state, done)

                    if t % train_every_n_steps == 0:
                        c.learn()

                    row = {
                        "episode": episode, "time_step": t,
                        "consumer_id": c.id,
                        "consumer_type": c.building_type,
                        "total_demand": float(sum(b.net_electricity_consumption[-1] for b in env.buildings)),
                        "net_electricity_consumption": float(c.building.net_electricity_consumption[-1]),
                        "soc_after_action": float(c.building.electrical_storage.soc[-1])
                        if len(c.building.electrical_storage.soc) else 0.0,
                        "agg_signal": last_agg_signals[c.id].tolist(),
                        "action": consumer_actions_scalar[c.id],
                        "reward": float(reward_custom),
                    }
                    current_episode_rows.append(row)

                    action_idx += 1

            # -----------------------------------------------------------
            # 6. Aggregator Learning
            # -----------------------------------------------------------
            regional_demands = []
            for agg in aggregator_agents:
                regional_demands.append(sum(c.building.net_electricity_consumption[-1]
                                            for c in agg.consumers))

            for idx, agg in enumerate(aggregator_agents):

                r = agg.get_reward([regional_demands[idx]])

                # Build next aggregator state
                next_states_list = []
                for c in agg.consumers:
                    next_states_list.append(
                        c.get_observation(
                            next_obs_dict[c.id],
                            building_obs_names[c.id],
                            last_agg_signals[c.id]
                        )
                    )
                next_agg_state = agg.get_observation(next_states_list)

                agg.store_experience(
                    agg.last_state,
                    agg.last_actions,
                    [consumer_actions_scalar[c.id] for c in agg.consumers],
                    r,
                    next_agg_state,
                    done
                )

                if t % train_every_n_steps == 0:
                    next_c_actions_batch = np.array(
                        [[consumer_actions_scalar[c.id] for c in agg.consumers]]
                        * agg.batch_size
                    )
                    next_c_actions_batch = torch.tensor(
                        next_c_actions_batch, dtype=torch.float32, device=agg.device
                    )
                    agg.learn(next_c_actions_batch)

            obs_dict = next_obs_dict

        # ================================================================
        #                   END OF EPISODE
        # ================================================================

        all_step_records.extend(current_episode_rows)

        # ----- SAVE MODELS -----
        print(f"\nSaving models for episode {episode}")
        for c in consumer_agents:
            c.save_models(checkpoint_dir, episode)
        for agg in aggregator_agents:
            agg.save_models(checkpoint_dir, episode)

        # ----- SAVE CSV -----
        df = pd.DataFrame(all_step_records)
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)

        # ----- SAVE JSON -----
        all_episodes_summary.append({"episode": episode})
        save_json({"episodes": all_episodes_summary}, json_path)

        print(f"Episode {episode} saved.\n")

    print("\n***** TRAINING COMPLETE *****")
    print(f"Results saved to {csv_path} and {json_path}")


# ================================================================
#                            MAIN CALL
# ================================================================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_training_and_simulation(
        num_episodes=2,
        use_ddpg_aggregator=True,
        train_every_n_steps=4
    )
