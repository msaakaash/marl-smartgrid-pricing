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
import re

from agents.consumer_agent import ConsumerAgentDQN
from agents.aggregator_agent import AggregatorAgentDDPG

CSV_COLUMNS = [
    "episode", "time_step", "consumer_id", "consumer_type", "total_demand",
    "net_electricity_consumption", "soc_after_action", "agg_signal",
    "action", "reward"
]


# ======================================================================
#                     CONSUMER + AGGREGATOR BUILDERS
# ======================================================================
def build_consumers_from_env(env, building_obs_names, key_path="security/keys/secret.key"):
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
            action_space=building.action_space,
            key_path=key_path,
            debug=True
        )

        agent.building_type = metadata["building_type"]
        consumer_agents.append(agent)
        consumer_metadata_map[i] = metadata

    print(f"Built {len(consumer_agents)} consumer agents.")
    return consumer_agents, consumer_metadata_map


def build_aggregators(consumer_agents, key_path="security/keys/secret.key", debug=True):
    aggregators = []
    num_regions = max(1, (len(consumer_agents) + 4) // 5)
    print(f"Building {num_regions} aggregator regions...")

    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5
        slice_cons = consumer_agents[start:end]

        agg = AggregatorAgentDDPG(
            region_id=region_id,
            consumers=slice_cons,
            key_path=key_path,
            debug=debug
        )
        aggregators.append(agg)

    return aggregators


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
            ep = int(match.group(1))
            if ep > latest_episode:
                latest_episode = ep

    return latest_episode


# ======================================================================
#                     MAIN TRAINING + SIMULATION LOOP
# ======================================================================
def run_training_and_simulation(schema_paths=None, num_episodes: int = 50,
                                use_ddpg_aggregator: bool = True,
                                save_dir: str = ".", checkpoint_dir: str = "checkpoints",
                                train_every_n_steps: int = 4):

    if schema_paths is None:
        schema_paths = ["citylearn_challenge_2022_phase_1"]

    # Load a CityLearn env with max 15 buildings
    buildings = []
    building_obs_names = {}

    for schema_path in schema_paths:
        env_schema = CityLearnEnv(schema=schema_path)

        for i, bld in enumerate(env_schema.buildings):
            if len(buildings) >= 15:
                break

            buildings.append(bld)
            obs_names = env_schema.observation_names[i]

            if not obs_names:
                obs_names = [f"obs_{j}" for j in range(len(bld.observation_space.sample()))]

            building_obs_names[len(buildings) - 1] = obs_names

        if len(buildings) >= 15:
            break

    env = CityLearnEnv(schema=schema_paths[0], buildings=buildings)
    print(f"Environment loaded: {len(env.buildings)} buildings")

    # Build agents
    consumer_agents, metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents)

    # File output setup
    csv_path = os.path.join(save_dir, "sample.csv")
    json_path = os.path.join(save_dir, "sample.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_episode = find_latest_checkpoint(checkpoint_dir)
    start_episode = latest_episode + 1 if latest_episode > 0 else 1

    if latest_episode > 0:
        print(f"Resuming from Episode {latest_episode}")
        for c in consumer_agents:
            c.load_models(checkpoint_dir, latest_episode)
            c.epsilon = c.epsilon_min
        for a in aggregator_agents:
            a.load_models(checkpoint_dir, latest_episode)

    all_step_records = []
    all_episodes_summary = []

    # ======================================================================
    #                       EPISODE LOOP
    # ======================================================================
    for episode in range(start_episode, num_episodes + 1):

        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        for c in consumer_agents: c.reset_episode()
        for a in aggregator_agents: a.reset_episode()

        print(f"\n==================  EPISODE {episode}  ==================")

        current_episode_step_records = []
        last_agg_signals_dict = {c.id: b"" for c in consumer_agents}
        consumer_state_dict = {}

        episode_demands = []
        consumer_stats = {c.id: {"reward_sum": 0, "steps": 0, "energy_sum": 0} for c in consumer_agents}
        agg_stats = {a.id: {"reward_sum": 0, "steps": 0, "total_demand": 0} for a in aggregator_agents}

        # ==================================================================
        #                          TIME STEP LOOP
        # ==================================================================
        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}", ncols=100):

            # --- 1. Consumers build state (using previous agg signal) ---
            consumer_state_dict.clear()

            for c in consumer_agents:
                packet = last_agg_signals_dict[c.id]
                agg_dec = c.decrypt_agg_packet(packet) if packet else np.array([0.0, 0.0], dtype=np.float32)
                consumer_state_dict[c.id] = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=agg_dec
                )

            # --- 2. Aggregators act → produce encrypted signals ---
            current_agg_signals_dict = {}

            for agg in aggregator_agents:
                cons_states = [consumer_state_dict[c.id] for c in agg.consumers]
                agg_state = agg.get_observation(cons_states)
                agg_action = agg.select_action(agg_state, noise_std=0.0)
                encrypted_packets = agg.encrypt_signals_for_consumers(agg_action)

                # -------------------------------------------------------
                # 🔥 ATTACK SIMULATION (you can toggle this ON/OFF)
                # -------------------------------------------------------
                ATTACK_ENABLED = True
                ATTACK_MODE = random.choice(["replay", "delay","flip","noise","replace"])
                ATTACK_INTENSITY = 0.6   # used only for flip/noise

                for cid, pkt in encrypted_packets.items():
                    if ATTACK_ENABLED:
                        print(f"[⚠️ ATTACK] {ATTACK_MODE.upper()} → Consumer {cid}")
                        tampered = agg.secure.attacker_tamper(
                            pkt,
                            mode=ATTACK_MODE,
                            intensity=ATTACK_INTENSITY
                        )
                        current_agg_signals_dict[cid] = tampered
                    else:
                        current_agg_signals_dict[cid] = pkt


            # --- 3. Consumers act (decrypt the attack) ---
            all_actions = []
            for c in consumer_agents:
                dec = c.decrypt_agg_packet(current_agg_signals_dict[c.id])
                obs_new = c.get_observation(obs_dict[c.id], building_obs_names[c.id], agg_signal=dec)
                action_val, action_idx = c.select_action(obs_new)
                action_scalar = float(np.atleast_1d(action_val)[0])
                all_actions.append([action_scalar])

            # --- 4. Env step ---
            next_obs, raw_rewards, done, _ = env.step(all_actions)
            next_obs_dict = {i: next_obs[i] for i in range(len(next_obs))}
            demand_now = sum(b.net_electricity_consumption[-1] for b in env.buildings)
            episode_demands.append(demand_now)

            # --- 5. Consumer Learning ---
            idx = 0
            for agg in aggregator_agents:
                for c in agg.consumers:

                    soc_before = float(c.building.electrical_storage.soc[-1])
                    price = float(obs_dict[c.id][10]) if len(obs_dict[c.id]) > 10 else 0.0

                    decrypted = c.decrypt_agg_packet(current_agg_signals_dict[c.id])
                    next_state = c.get_observation(next_obs_dict[c.id], building_obs_names[c.id], agg_signal=decrypted)

                    act_scalar = float(np.atleast_1d(all_actions[idx])[0])
                    raw_reward = raw_rewards[idx]

                    reward_custom = c.get_reward(
                        raw_reward,
                        c.building.net_electricity_consumption[-1],
                        act_scalar,
                        soc_before,
                        price,
                        decrypted
                    )

                    c.store_experience(c.last_state, c.last_action_index, reward_custom, next_state, float(done))

                    if t % train_every_n_steps == 0:
                        c.learn()

                    # Logging
                    consumer_stats[c.id]["reward_sum"] += float(reward_custom)
                    consumer_stats[c.id]["steps"] += 1
                    consumer_stats[c.id]["energy_sum"] += float(c.building.net_electricity_consumption[-1])

                    record = {
                        "episode": episode,
                        "time_step": t,
                        "consumer_id": c.id,
                        "consumer_type": metadata_map[c.id]["building_type"],
                        "total_demand": float(demand_now),
                        "net_electricity_consumption": float(c.building.net_electricity_consumption[-1]),
                        "soc_after_action": soc_before,
                        "agg_signal": decrypted.tolist(),
                        "action": act_scalar,
                        "reward": float(reward_custom)
                    }
                    current_episode_step_records.append(record)
                    idx += 1

            # --- 6. Aggregator Learning ---
            regional_demands = []
            for agg in aggregator_agents:
                reg_demand = sum(c.building.net_electricity_consumption[-1] for c in agg.consumers)
                regional_demands.append(reg_demand)

            for ai, agg in enumerate(aggregator_agents):
                reward_agg = agg.get_reward([regional_demands[ai]])
                agg_stats[agg.id]["reward_sum"] += float(reward_agg)
                agg_stats[agg.id]["steps"] += 1
                agg_stats[agg.id]["total_demand"] += float(regional_demands[ai])

                if t % train_every_n_steps == 0:
                    next_states = []
                    for c in agg.consumers:
                        dec = c.decrypt_agg_packet(current_agg_signals_dict[c.id])
                        next_states.append(
                            c.get_observation(next_obs_dict[c.id], building_obs_names[c.id], agg_signal=dec)
                        )
                    next_agg_state = agg.get_observation(next_states)

                    last_actions_list = [float(np.atleast_1d(a)[0]) for a in all_actions]
                    agg.store_experience(agg.last_state, agg.last_actions, last_actions_list, reward_agg, next_agg_state, float(done))
                    agg.learn()

            # Update transition
            obs_dict = next_obs_dict
            last_agg_signals_dict = current_agg_signals_dict

        # ==================================================================
        #                       END OF EPISODE
        # ==================================================================
        total_energy = sum(episode_demands)
        total_reward = sum(v["reward_sum"] for v in consumer_stats.values())

        episode_summary = {
            "episode": episode,
            "total_energy": float(total_energy),
            "total_reward": float(total_reward),
            "consumers": [],
            "aggregators": []
        }

        for cid, st in consumer_stats.items():
            avg_r = st["reward_sum"] / st["steps"]
            avg_e = st["energy_sum"] / st["steps"]
            episode_summary["consumers"].append({
                "consumer_id": cid,
                "building_type": metadata_map[cid]["building_type"],
                "avg_reward": float(avg_r),
                "avg_energy": float(avg_e)
            })

        for aid, st in agg_stats.items():
            avg_r = st["reward_sum"] / st["steps"]
            avg_d = st["total_demand"] / st["steps"]
            episode_summary["aggregators"].append({
                "aggregator_id": aid,
                "avg_reward": float(avg_r),
                "avg_demand": float(avg_d)
            })

        all_episodes_summary.append(episode_summary)
        all_step_records.extend(current_episode_step_records)

        # Save models
        print(f"\n--- Saving Checkpoint EP{episode} ---")
        for c in consumer_agents:
            c.save_models(checkpoint_dir, episode)
        for a in aggregator_agents:
            a.save_models(checkpoint_dir, episode)

        # Save CSV/JSON
        df = pd.DataFrame(all_step_records)
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)

        save_json({"episodes": all_episodes_summary}, json_path)
        print(f"Saved {csv_path} & {json_path}")

        # Delete old checkpoint
        if episode > 1:
            old = episode - 1
            for f in os.listdir(checkpoint_dir):
                if f.endswith(f"_ep{old}.pth"):
                    os.remove(os.path.join(checkpoint_dir, f))

        print(f"Episode {episode} Summary → "
              f"Energy={total_energy:.2f}, Reward={total_reward:.2f}")

    print("\nTraining Completed.")


# ======================================================================
#                              MAIN ENTRY
# ======================================================================
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_training_and_simulation(
        num_episodes=1,     # change number of episodes here
        train_every_n_steps=4
    )
