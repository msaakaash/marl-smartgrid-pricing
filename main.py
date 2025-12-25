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

# ==========================================================
#                  ATTACK SCHEDULE (ADDED)
# ==========================================================
ATTACK_SCHEDULE = [
    "replay", "delay", "flip", "noise", "replace",
    "ddos", "mitm", "impersonation", "blackhole", "grayhole",
    "selective_forward", "reorder", "truncate", "pad",
    "adaptive"
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
def run_training_and_simulation(schema_paths=None, num_episodes: int = 15,
                                use_ddpg_aggregator: bool = True,
                                save_dir: str = ".", checkpoint_dir: str = "checkpoints",
                                train_every_n_steps: int = 4):

    if schema_paths is None:
        schema_paths = ["citylearn_challenge_2022_phase_1"]

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

    consumer_agents, metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents)

    csv_path = os.path.join(save_dir, "sample.csv")
    json_path = os.path.join(save_dir, "sample.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_step_records = []
    all_episodes_summary = []

    # ==================================================================
    #                       EPISODE LOOP
    # ==================================================================
    for episode_idx, episode in enumerate(range(1, num_episodes + 1)):

        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        for c in consumer_agents: c.reset_episode()
        for a in aggregator_agents: a.reset_episode()

        print(f"\n==================  EPISODE {episode}  ==================")

        # ----------------------------------------------------------
        # FIXED ATTACK FOR THIS EPISODE (ONE ATTACK ONLY)
        # ----------------------------------------------------------
        ATTACK_ENABLED = episode_idx < len(ATTACK_SCHEDULE)
        ATTACK_MODE = ATTACK_SCHEDULE[episode_idx] if ATTACK_ENABLED else None
        ATTACK_INTENSITY = 0.6

        if ATTACK_ENABLED:
            print(f"[🔬 EXPERIMENT] ATTACK = {ATTACK_MODE.upper()}")

        last_agg_signals_dict = {c.id: b"" for c in consumer_agents}
        consumer_state_dict = {}

        # ==========================================================
        #                      TIME STEP LOOP
        # ==========================================================
        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}", ncols=100):

            consumer_state_dict.clear()
            for c in consumer_agents:
                packet = last_agg_signals_dict[c.id]
                agg_dec = c.decrypt_agg_packet(packet) if packet else np.array([0.0, 0.0], dtype=np.float32)
                consumer_state_dict[c.id] = c.get_observation(
                    obs_dict[c.id], building_obs_names[c.id], agg_signal=agg_dec
                )

            current_agg_signals_dict = {}

            for agg in aggregator_agents:
                cons_states = [consumer_state_dict[c.id] for c in agg.consumers]
                agg_state = agg.get_observation(cons_states)
                agg_action = agg.select_action(agg_state, noise_std=0.0)
                encrypted_packets = agg.encrypt_signals_for_consumers(agg_action)

                for cid, pkt in encrypted_packets.items():
                    if ATTACK_ENABLED:
                        tampered = agg.secure.attacker_tamper(
                            pkt,
                            mode=ATTACK_MODE,
                            intensity=ATTACK_INTENSITY
                        )
                        current_agg_signals_dict[cid] = tampered
                    else:
                        current_agg_signals_dict[cid] = pkt

            all_actions = []
            for c in consumer_agents:
                dec = c.decrypt_agg_packet(current_agg_signals_dict[c.id])
                obs_new = c.get_observation(obs_dict[c.id], building_obs_names[c.id], agg_signal=dec)
                action_val, action_idx = c.select_action(obs_new)
                all_actions.append([float(np.atleast_1d(action_val)[0])])

            next_obs, raw_rewards, done, _ = env.step(all_actions)
            obs_dict = {i: next_obs[i] for i in range(len(next_obs))}
            last_agg_signals_dict = current_agg_signals_dict

        print(f"Episode {episode} completed under attack: {ATTACK_MODE}")

    print("\nTraining Completed.")


# ======================================================================
#                              MAIN ENTRY
# ======================================================================
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_training_and_simulation(
        num_episodes=15,     # EXACTLY one episode per attack
        train_every_n_steps=4
    )
