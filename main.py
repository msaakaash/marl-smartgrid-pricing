import os
import json
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
from citylearn.citylearn import CityLearnEnv
import torch

from agents.consumer_agent import ConsumerAgentDQN
from agents.aggregator_agent import AggregatorAgentDDPG
from security.metrics import NetworkMetrics

# ==========================================================
#                     ATTACK SCHEDULE
# ==========================================================
ATTACK_SCHEDULE = [
    "replay", "delay", "flip", "noise", "replace",
    "ddos", "mitm", "impersonation", "blackhole", "grayhole",
    "selective_forward", "reorder", "truncate", "pad",
    "adaptive"
]

# ==========================================================
#              CONSUMER + AGGREGATOR BUILDERS
# ==========================================================
def build_consumers_from_env(env, building_obs_names, key_path="security/keys/secret.key"):
    consumer_agents = []

    for i, building in enumerate(env.buildings):
        agent = ConsumerAgentDQN(
            building=building,
            building_id=i,
            metadata={
                "building_type": random.choice(
                    ["residential", "office", "mall", "hospital", "school"]
                )
            },
            action_space=building.action_space,
            key_path=key_path,
            debug=False
        )
        consumer_agents.append(agent)

    print(f"Built {len(consumer_agents)} consumer agents.")
    return consumer_agents


def build_aggregators(consumer_agents, key_path="security/keys/secret.key", debug=False):
    aggregators = []
    num_regions = max(1, (len(consumer_agents) + 4) // 5)
    print(f"Building {num_regions} aggregator regions...")

    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5

        agg = AggregatorAgentDDPG(
            region_id=region_id,
            consumers=consumer_agents[start:end],
            key_path=key_path,
            debug=debug
        )
        aggregators.append(agg)

    return aggregators

# ==========================================================
#              MAIN TRAINING + SIMULATION LOOP
# ==========================================================
def run_training_and_simulation(
    schema_path="citylearn_challenge_2022_phase_1"
):
    env = CityLearnEnv(schema=schema_path)
    env.buildings = env.buildings[:15]

    building_obs_names = {
        i: env.observation_names[i] for i in range(len(env.buildings))
    }

    print(f"Environment loaded: {len(env.buildings)} buildings")

    consumer_agents = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents)

    all_results = []

    # ======================================================
    #                    EPISODE LOOP
    # ======================================================
    for episode_idx, ATTACK_MODE in enumerate(ATTACK_SCHEDULE):

        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        metrics = NetworkMetrics()
        ATTACK_INTENSITY = 0.6

        print(f"\n================== EPISODE {episode_idx + 1} ==================")
        print(f"[ATTACK ENABLED] {ATTACK_MODE.upper()}")

        last_agg_packets: Dict[int, bytes] = {
            c.id: b"" for c in consumer_agents
        }

        # ==================================================
        #                  TIME STEP LOOP
        # ==================================================
        for t in tqdm(range(env.time_steps - 1), ncols=100):

            consumer_states = {}

            # ---------- Consumer Observation ----------
            for c in consumer_agents:
                pkt = last_agg_packets[c.id]
                agg_dec = (
                    c.decrypt_agg_packet(pkt)
                    if pkt else np.array([0.0, 0.0], dtype=np.float32)
                )

                consumer_states[c.id] = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    agg_signal=agg_dec
                )

            current_packets = {}

            # ---------- Aggregator Sends Signals ----------
            for agg in aggregator_agents:
                states = [consumer_states[c.id] for c in agg.consumers]
                agg_state = agg.get_observation(states)
                agg_action = agg.select_action(agg_state, noise_std=0.0)

                encrypted = agg.encrypt_signals_for_consumers(agg_action)

                for cid, pkt in encrypted.items():
                    metrics.record_send()
                    send_time = time.time()

                    tampered = agg.secure.attacker_tamper(
                        pkt,
                        mode=ATTACK_MODE,
                        intensity=ATTACK_INTENSITY
                    )

                    if tampered:
                        delay = time.time() - send_time
                        current_packets[cid] = tampered
                        metrics.record_receive(True, delay)
                    else:
                        current_packets[cid] = b""
                        metrics.record_receive(False, 0.0)

            # ---------- Consumer Actions ----------
            actions = []
            for c in consumer_agents:
                dec = c.decrypt_agg_packet(current_packets[c.id])
                obs_new = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    agg_signal=dec
                )
                action, _ = c.select_action(obs_new)
                actions.append([float(np.atleast_1d(action)[0])])

            next_obs, _, _, _ = env.step(actions)
            obs_dict = {i: next_obs[i] for i in range(len(next_obs))}
            last_agg_packets = current_packets

        # ==================================================
        #                 ENERGY METRICS
        # ==================================================
        net_load = np.array(env.net_electricity_consumption)
        peak_load = float(np.max(net_load))
        avg_load = float(np.mean(net_load))
        par = peak_load / (avg_load + 1e-6)

        security_metrics = metrics.results(par)

        result_row = {
            "Episode": episode_idx + 1,
            "Attack": ATTACK_MODE,
            **security_metrics
        }

        all_results.append(result_row)

        print("\n🔐 Security & Network Metrics")
        print(pd.DataFrame([result_row]).to_string(index=False))

    # ======================================================
    #                 SAVE RESULTS
    # ======================================================
    df = pd.DataFrame(all_results)
    df.to_csv("security_attack_metrics.csv", index=False)

    print("\n📊 FINAL RESULTS (ALL ATTACKS)")
    print(df.to_string(index=False))
    print("\n✅ Saved to security_attack_metrics.csv")

# ==========================================================
#                     MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_training_and_simulation()
