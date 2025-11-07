# main.py
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Mapping, Any, Union
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building

# Import consumer agent (unchanged)
from agents.consumer_agent import ConsumerAgentDQN

# Try to import DQN aggregator; fall back to rule-based aggregator if not found
try:
    from agents.aggregator_agent import AggregatorAgentDQN as AggregatorAgentImpl
    AGGREGATOR_IS_DQN = True
except Exception:
    from agents.aggregator_agent import AggregatorAgent as AggregatorAgentImpl
    AGGREGATOR_IS_DQN = False

# Columns required for per-step CSV output
CSV_COLUMNS = [
    "episode",
    "time_step",
    "consumer_id",
    "consumer_type",
    "total_demand",
    "net_electricity_consumption",
    "soc_after_action",
    "agg_signal",
    "action",
    "reward"
]


def build_consumers_from_env(env, building_obs_names):
    """
    Create ConsumerAgentDQN objects for all buildings in env (up to available).
    Attach a metadata mapping separately to avoid modifying consumer code.
    """
    consumer_agents = []
    consumer_metadata_map = {}

    for i, building in enumerate(env.buildings):
        metadata = {
            "building_type": random.choice(["residential", "office", "mall"]),
            "critical_load_fraction": random.uniform(0.1, 0.3),
            "prime_hours": random.sample(range(8, 20), 3),
            "cheating_propensity": random.uniform(0.0, 0.3),
            "emergency_flag": False
        }
        # Create agent with required constructor args (no changes to consumer code)
        agent = ConsumerAgentDQN(
            building=building,
            building_id=i,
            metadata=metadata,
            action_space=building.action_space
        )
        # attach convenience field for logging (doesn't modify class internals)
        agent.building_type = metadata["building_type"]
        consumer_agents.append(agent)
        consumer_metadata_map[i] = metadata

    return consumer_agents, consumer_metadata_map


def build_aggregators(consumer_agents, use_dqn=True):
    """Create aggregator agents and group consumers into regions of 5 (as before)."""
    aggregators = []
    num_regions = max(1, (len(consumer_agents) + 4) // 5)
    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5
        consumers_slice = consumer_agents[start:end]
        if AGGREGATOR_IS_DQN and use_dqn:
            # DQN aggregator expects (region_id, consumers=...)
            agg = AggregatorAgentImpl(region_id=region_id, consumers=consumers_slice)
        else:
            # Fallback to rule-based aggregator (keeps same signature)
            agg = AggregatorAgentImpl(region_id=region_id, consumers=consumers_slice)
        aggregators.append(agg)
    return aggregators


def record_step_csv(step_rows, csv_path):
    df = pd.DataFrame(step_rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def run_training_and_simulation(
    schema_paths=None,
    num_episodes: int = 3,
    target_update_freq: int = 200,
    grid_virtual_capacity: float = 10.0,
    dr_threshold_factor: float = 0.85,
    save_dir: str = ".",
):
    if schema_paths is None:
        schema_paths = [
            "citylearn_challenge_2022_phase_1",
            "citylearn_challenge_2022_phase_2",
            "citylearn_challenge_2022_phase_3",
        ]

    # select up to 15 buildings from schemas
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

    print("Environment loaded: {} buildings".format(len(env.buildings)))
    print("Aggregator implementation: {}".format("DQN" if AGGREGATOR_IS_DQN else "rule-based fallback"))

    # build agents
    consumer_agents, consumer_metadata_map = build_consumers_from_env(env, building_obs_names)
    aggregator_agents = build_aggregators(consumer_agents, use_dqn=AGGREGATOR_IS_DQN)

    # prepare outputs
    step_records = []
    episodes_summary = []

    # training loop
    for episode in range(1, num_episodes + 1):
        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        # reset consumers and aggregators
        for c in consumer_agents:
            if hasattr(c, "reset_episode"):
                c.reset_episode()
        for agg in aggregator_agents:
            if hasattr(agg, "reset_episode"):
                agg.reset_episode()

        episode_demands = []
        episode_reward_sum = 0.0

        # per-episode per-agent accumulators for summary
        consumer_episode_stats = {c.id: {"reward_sum": 0.0, "steps": 0, "energy_sum": 0.0} for c in consumer_agents}
        aggregator_episode_stats = {agg.id: {"reward_sum": 0.0, "steps": 0, "dr_count": 0, "total_demand": 0.0} for agg in aggregator_agents}

        print("Starting episode {} / {}".format(episode, num_episodes))

        for t in tqdm(range(env.time_steps - 1), desc=f"Episode {episode}"):
            # 1) Aggregators compute total_grid_demand from their internal total_demand (kept from previous steps)
            total_grid_demand = sum(agg.total_demand if hasattr(agg, "total_demand") else 0.0 for agg in aggregator_agents)
            is_dr_needed = total_grid_demand > (grid_virtual_capacity * dr_threshold_factor)

            # 2) For each aggregator: build regional obs and produce signal
            agg_signals = {}
            for agg in aggregator_agents:
                # gather regional observations for aggregator consumers
                regional_obs = []
                regional_obs_names = []
                for c in agg.consumers:
                    regional_obs.append(obs_dict[c.id])
                    regional_obs_names.append(building_obs_names[c.id])

                # call get_observation (supports both rule-based and dqn aggregator)
                try:
                    agg_obs = agg.get_observation(regional_obs, regional_obs_names)
                except TypeError:
                    # some aggregator implementations might expect different args; try fallback
                    agg_obs = agg.get_observation(regional_obs, regional_obs_names)

                # select_action may accept is_dr_needed (rule-based) or not (DQN)
                try:
                    agg_signal, agg_action_index = agg.select_action(agg_obs, is_dr_needed)
                except TypeError:
                    agg_signal, agg_action_index = agg.select_action(agg_obs)

                if agg_signal is None:
                    # fallback: if aggregator returns only index, map to default signal
                    agg_signal = getattr(agg, "last_signal", np.array([0.0, 0.0], dtype=np.float32))

                agg_signals[agg.id] = np.array(agg_signal, dtype=np.float32)

                # if DR active, mark aggregator stat
                if is_dr_needed:
                    aggregator_episode_stats[agg.id]["dr_count"] += 1

            # 3) Consumers receive signal and act
            all_actions = []
            for agg in aggregator_agents:
                for c in agg.consumers:
                    # create observation for consumer (consumer.get_observation handles building obs + agg signal)
                    obs = c.get_observation(obs_dict[c.id], building_obs_names[c.id], agg_signal=agg_signals[agg.id])
                    action_value, action_index = c.select_action(obs)
                    all_actions.append(action_value)

            # 4) Step environment with all actions
            next_observations, raw_rewards, done, info = env.step(all_actions)
            next_obs_dict = {i: next_observations[i] for i in range(len(next_observations))}

            # 5) Compute total demand and per-consumer rewards and learning
            current_total_demand = sum(b.net_electricity_consumption[-1] for b in env.buildings)
            episode_demands.append(current_total_demand)

            # iterate consumers again to compute reward, store experience, learn
            # Note: actions in all_actions correspond to consumers in the order of consumers in aggregators
            action_idx = 0
            for agg in aggregator_agents:
                for c in agg.consumers:
                    # safe metadata access
                    c_type = getattr(c, "building_type", consumer_metadata_map.get(c.id, {}).get("building_type", "unknown"))

                    soc_before = float(c.building.electrical_storage.soc[-1]) if len(c.building.electrical_storage.soc) > 0 else 0.0
                    current_price = float(obs_dict[c.id][10]) if len(obs_dict[c.id]) > 10 else 0.0
                    agg_signal_val = agg_signals[agg.id] if agg.id in agg_signals else np.array([0.0, 0.0], dtype=np.float32)

                    action_value = all_actions[action_idx]
                    raw_reward = raw_rewards[action_idx]

                    # compute custom reward using consumer's method
                    custom_reward = c.get_reward(
                        raw_reward=raw_reward,
                        current_power_demand=c.building.net_electricity_consumption[-1],
                        action_value=action_value,
                        soc_before_action=soc_before,
                        electricity_pricing=current_price,
                        agg_signal=agg_signal_val
                    )

                    # next state
                    next_state = c.get_observation(next_obs_dict[c.id], building_obs_names[c.id], agg_signal=agg_signal_val)

                    # store and learn
                    try:
                        c.store_experience(c.last_state, c.last_action_index, custom_reward, next_state, done)
                    except Exception:
                        # consumer may have different store signature; try append to replay if exists
                        pass

                    try:
                        loss = c.learn()
                    except Exception:
                        loss = None

                    # update per-consumer episode stats
                    consumer_episode_stats[c.id]["reward_sum"] += float(custom_reward)
                    consumer_episode_stats[c.id]["steps"] += 1
                    consumer_episode_stats[c.id]["energy_sum"] += float(c.building.net_electricity_consumption[-1])

                    # CSV row
                    row = {
                        "episode": episode,
                        "time_step": t,
                        "consumer_id": c.id,
                        "consumer_type": c_type,
                        "total_demand": float(current_total_demand),
                        "net_electricity_consumption": float(c.building.net_electricity_consumption[-1]),
                        "soc_after_action": soc_before,
                        "agg_signal": agg_signal_val.tolist(),
                        "action": float(np.atleast_1d(action_value).item()),
                        "reward": float(custom_reward)
                    }
                    step_records.append(row)

                    # print per-consumer concise professional line
                    if t % 1000 == 0:
                        print(
                            "Step {:03d} | Consumer {:03d} | Type {:10s} | Action {:6.3f} | Reward {:7.3f} | SOC {:5.3f}".format(
                                t, c.id, str(c_type), float(np.atleast_1d(action_value).item()), float(custom_reward), soc_before
                            )
                        )

                    action_idx += 1

            # 6) Aggregators compute reward from regional demands and learn (if DQN implemented)
            # Build region-wise demand list for aggregator reward calculations
            regional_demands_by_agg = []
            for agg in aggregator_agents:
                # compute sum demand for that aggregator region (sum net_electricity_consumption for its consumers)
                regs = [c.building.net_electricity_consumption[-1] for c in agg.consumers]
                regional_demands_by_agg.append(np.sum(regs))

            # for each aggregator, compute reward and (if DQN) store and learn
            for agg_idx, agg in enumerate(aggregator_agents):
                try:
                    agg_reward = agg.get_reward(regional_demands_by_agg)  # many implementations take list of all regions
                except TypeError:
                    # fallback: pass only region demand
                    agg_reward = agg.get_reward([regional_demands_by_agg[agg_idx]])

                aggregator_episode_stats[agg.id]["reward_sum"] += float(agg_reward)
                aggregator_episode_stats[agg.id]["steps"] += 1
                aggregator_episode_stats[agg.id]["total_demand"] += float(regional_demands_by_agg[agg_idx])

                # store aggregator experience if available
                try:
                    next_agg_obs = agg.get_observation([obs_dict[c.id] for c in agg.consumers],
                                                       [building_obs_names[c.id] for c in agg.consumers])
                    agg.store_experience(agg.last_state, getattr(agg, "last_action_index", 0), float(agg_reward), next_agg_obs, done)
                except Exception:
                    pass

                try:
                    agg_loss = agg.learn()
                except Exception:
                    agg_loss = None

                # print per-aggregator concise professional line
                if t % 1000 == 0:
                    print(
                        "Step {:03d} | Aggregator {:02d} | RegionDemand {:8.2f} | Reward {:7.3f} | LastSignal {}".format(
                            t, agg.id, float(regional_demands_by_agg[agg_idx]), float(agg_reward), getattr(agg, "last_signal", None)
                        )
                    )

            # optional: target network updates for consumers (if required)
            if (t % target_update_freq) == 0:
                for c in consumer_agents:
                    try:
                        c.update_target_network()
                    except Exception:
                        pass

            # shift observations
            obs_dict = next_obs_dict

        # End of episode: compute episode-level summaries
        episode_total_energy = float(np.sum(episode_demands)) if episode_demands else 0.0
        episode_total_reward = sum(v["reward_sum"] for v in consumer_episode_stats.values())

        # Build per-consumer summary entries
        consumers_summary = []
        for cid, stats in consumer_episode_stats.items():
            avg_reward = float(stats["reward_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            avg_energy = float(stats["energy_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            consumers_summary.append({
                "consumer_id": cid,
                "building_type": consumer_metadata_map.get(cid, {}).get("building_type", getattr(consumer_agents[cid], "building_type", "unknown")),
                "avg_reward": avg_reward,
                "avg_energy": avg_energy
            })

        aggregators_summary = []
        for aid, stats in aggregator_episode_stats.items():
            avg_reward = float(stats["reward_sum"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            avg_demand = float(stats["total_demand"] / stats["steps"]) if stats["steps"] > 0 else 0.0
            aggregators_summary.append({
                "aggregator_id": aid,
                "avg_reward": avg_reward,
                "avg_demand": avg_demand,
                "dr_triggers": stats["dr_count"]
            })

        episode_record = {
            "episode": episode,
            "total_energy": episode_total_energy,
            "total_reward": episode_total_reward,
            "consumers": consumers_summary,
            "aggregators": aggregators_summary
        }
        episodes_summary.append(episode_record)

        # print episode summary (professional)
        print("\nEpisode {} summary: total_energy={:.2f}, total_reward={:.2f}".format(
            episode, episode_total_energy, episode_total_reward
        ))
        print("Aggregators summary (per region):")
        for a in aggregators_summary:
            print(" - Aggregator {:02d}: avg_reward={:.4f}, avg_demand={:.2f}, dr_triggers={}".format(
                a["aggregator_id"], a["avg_reward"], a["avg_demand"], a["dr_triggers"]
            ))
        print("Consumers summary (sample):")
        for c in consumers_summary:
            print(" - Consumer {:03d} ({}): avg_reward={:.4f}, avg_energy={:.2f}".format(
                c["consumer_id"], c["building_type"], c["avg_reward"], c["avg_energy"]
            ))

    # After all episodes: save CSV and JSON
    csv_path = os.path.join(save_dir, "training_step_data.csv")
    json_path = os.path.join(save_dir, "training_summary.json")

    # ensure ordering of CSV columns
    if len(step_records) > 0:
        df = pd.DataFrame(step_records)
        # Ensure all columns present (missing ones filled with NaN)
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)
    else:
        # create empty CSV with headers
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    save_json({"episodes": episodes_summary}, json_path)

    print("\nTraining complete. Saved files:")
    print(" - Per-step CSV: {}".format(csv_path))
    print(" - Episode summary JSON: {}".format(json_path))


if __name__ == "__main__":
    # reproducibility seeds
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except Exception:
        pass

    run_training_and_simulation(num_episodes=3)
