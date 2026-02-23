# ============================== main.py ==============================

import os
import random
import warnings
import io
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
import torch

# Suppress third-party startup noise (Gym notice + matplotlib/pyparsing deprecation spam).
try:
    import gym_notices.notices as _gym_notices
    _gym_notices.notices.clear()
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyparsing\..*")
warnings.filterwarnings("ignore", message=r".*setParseAction.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*parseString.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*resetCache.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*enablePackrat.*deprecated.*", category=DeprecationWarning)
try:
    from pyparsing import PyparsingDeprecationWarning
    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
except Exception:
    pass

from citylearn.citylearn import CityLearnEnv
with contextlib.redirect_stderr(io.StringIO()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

from agents.consumer_agent import ConsumerAgentDQN
from agents.aggregator_agent import AggregatorAgentDDPG
from security.metrics import SmartGridSecurityMetrics
from security.attacks import AttackSimulator

# ==========================================================
#                     ATTACK SCHEDULE
# ==========================================================
ATTACK_SCHEDULE = [
    "sybil",
    "false_data_injection",
    "bit_flip",
    "ddos",
    "replay",
]


def save_final_security_plot(df: pd.DataFrame, output_path: str = "results/final_security_metrics.png"):
    """
    Create a production-style summary chart from final attack metrics.
    The chart uses robust clipping (IQR bounds) per metric to suppress visual outliers.
    """
    if df.empty:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = [
        "Packet Delivery Ratio",
        "Auth Failure Rate",
        "Mean Reward",
        "Reward Variance",
    ]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return

    plot_df = df.copy()
    for col in available_metrics:
        values = pd.to_numeric(plot_df[col], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            plot_df[col] = values.clip(lower=lo, upper=hi)
        else:
            plot_df[col] = values

    attack_labels = plot_df["Attack"].astype(str).tolist() if "Attack" in plot_df.columns else [str(i + 1) for i in range(len(plot_df))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=140)
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for i, metric in enumerate(available_metrics[:4]):
        ax = axes.flat[i]
        y = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy()
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", linewidth=2.4, color=colors[i], markersize=6)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(attack_labels, rotation=20, ha="right")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    for j in range(len(available_metrics), 4):
        axes.flat[j].axis("off")

    fig.suptitle("Security Metrics by Attack Scenario", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Source: final_attack_metrics.csv | Outliers clipped with IQR bounds for cleaner comparison",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _format_table(rows, columns):
    if not rows:
        return "(no rows)"

    widths = {}
    for col in columns:
        widths[col] = max(len(col), max(len(str(row.get(col, ""))) for row in rows))

    sep = "+-" + "-+-".join("-" * widths[col] for col in columns) + "-+"
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    lines = [sep, header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) + " |")
    lines.append(sep)
    return "\n".join(lines)


def print_episode_crypto_tables(episode, attack_mode, aggregators, consumers):
    enc_rows = []
    for agg in aggregators:
        info = getattr(getattr(agg, "secure", None), "last_encrypt_info", None)
        if info:
            enc_rows.append({
                "Episode": episode,
                "Attack": attack_mode,
                "Agent": f"Aggregator-{agg.id}",
                "NonceHex": info.get("nonce_hex", ""),
                "CiphertextHex": info.get("ciphertext_hex", ""),
                "TagHex": info.get("tag_hex", ""),
                "PlaintextVals": info.get("plaintext_vals", []),
                "PacketLen": info.get("packet_len", ""),
            })

    dec_rows = []
    for c in consumers:
        info = getattr(getattr(c, "secure", None), "last_decrypt_info", None)
        if info:
            dec_rows.append({
                "Episode": episode,
                "Attack": attack_mode,
                "Agent": f"Consumer-{c.id}",
                "AuthOK": info.get("auth_ok", ""),
                "NonceHex": info.get("nonce_hex", ""),
                "CiphertextHex": info.get("ciphertext_hex", ""),
                "TagHex": info.get("tag_hex", ""),
                "PlaintextVals": info.get("plaintext_vals", []),
                "PacketLen": info.get("packet_len", ""),
            })

    print("\nCrypto Table: Encryption (latest packet per aggregator)")
    print(_format_table(
        enc_rows,
        ["Episode", "Attack", "Agent", "NonceHex", "CiphertextHex", "TagHex", "PlaintextVals", "PacketLen"]
    ))
    print("\nCrypto Table: Decryption (latest packet per consumer)")
    print(_format_table(
        dec_rows,
        ["Episode", "Attack", "Agent", "AuthOK", "NonceHex", "CiphertextHex", "TagHex", "PlaintextVals", "PacketLen"]
    ))


# ==========================================================
#              CONSUMER + AGGREGATOR BUILDERS
# ==========================================================
def build_consumers_from_env(env, building_obs_names, key_path):
    consumers = []
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
            key_path=key_path
        )
        consumers.append(agent)
    return consumers


def build_aggregators(consumers, metrics, key_path):
    aggregators = []
    num_regions = max(1, (len(consumers) + 4) // 5)

    for region_id in range(num_regions):
        start = region_id * 5
        end = start + 5

        agg = AggregatorAgentDDPG(
            region_id=region_id,
            consumers=consumers[start:end],
            key_path=key_path
        )

        # Attach metrics to secure channel
        agg.secure.metrics = metrics
        aggregators.append(agg)

    return aggregators

# ==========================================================
#              MAIN TRAINING + SIMULATION LOOP
# ==========================================================
def run_training_and_simulation(
    schema_path="citylearn_challenge_2022_phase_1",
    key_path="security/keys/secret.key"
):
    env = CityLearnEnv(schema=schema_path)
    env.buildings = env.buildings[:15]

    building_obs_names = {
        i: env.observation_names[i]
        for i in range(len(env.buildings))
    }

    print(f"Environment loaded: {len(env.buildings)} buildings")

    all_results = []

    # ======================================================
    #                    EPISODE LOOP
    # ======================================================
    for ep, ATTACK_MODE in enumerate(ATTACK_SCHEDULE, start=1):

        observations = env.reset()
        obs_dict = {i: observations[i] for i in range(len(observations))}

        metrics = SmartGridSecurityMetrics()
        ATTACK_INTENSITY = 0.6
        attack_sim = AttackSimulator(
            attack_type=ATTACK_MODE,
            intensity=ATTACK_INTENSITY
        )

        consumers = build_consumers_from_env(env, building_obs_names, key_path)
        aggregators = build_aggregators(consumers, metrics, key_path)

        print("\n" + "=" * 60)
        print(f"EPISODE {ep} | ATTACK MODE: {ATTACK_MODE.upper()}")
        print("=" * 60)

        last_packets: Dict[int, bytes] = {c.id: b"" for c in consumers}

        # ==================================================
        #                  TIME STEP LOOP
        # ==================================================
        for t in tqdm(range(env.time_steps - 1), ncols=100, leave=False):

            consumer_states = {}

            # -------- Consumer Observation --------
            for c in consumers:
                pkt = last_packets[c.id]
                dec = (
                    c.decrypt_signal(pkt)
                    if pkt else np.array([0.0, 0.0], dtype=np.float32)
                )

                consumer_states[c.id] = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    agg_signal=dec
                )

            current_packets = {}

            # -------- Aggregator -> Consumer (Encrypted + Attacked) --------
            for agg in aggregators:
                states = [consumer_states[c.id] for c in agg.consumers]
                agg_state = agg.get_observation(states)
                agg_action = agg.select_action(agg_state)

                encrypted = agg.encrypt_signals_for_consumers(agg_action)

                for cid, pkt in encrypted.items():
                    attacked = attack_sim.apply(pkt)
                    current_packets[cid] = attacked

            # -------- Consumer Actions --------
            actions = []
            for c in consumers:
                dec = c.decrypt_signal(current_packets[c.id])
                obs_new = c.get_observation(
                    obs_dict[c.id],
                    building_obs_names[c.id],
                    agg_signal=dec
                )
                action, _ = c.select_action(obs_new)
                actions.append([float(np.atleast_1d(action)[0])])

            next_obs, rewards, _, _ = env.step(actions)
            obs_dict = {i: next_obs[i] for i in range(len(next_obs))}
            last_packets = current_packets

            for r in rewards:
                metrics.record_reward(float(r))

        # ==================================================
        #                 METRIC CALCULATION
        # ==================================================
        grid_metrics = metrics.grid_metrics(env.net_electricity_consumption)
        security_metrics = metrics.security_metrics()
        learning_metrics = metrics.learning_metrics()

        row = {
            "Episode": ep,
            "Attack": ATTACK_MODE,
            **grid_metrics,
            **security_metrics,
            **learning_metrics
        }

        all_results.append(row)

        print("\nEpisode Summary")
        print(pd.DataFrame([row]).to_string(index=False))
        print_episode_crypto_tables(ep, ATTACK_MODE, aggregators, consumers)

    # ======================================================
    #                 SAVE RESULTS
    # ======================================================
    df = pd.DataFrame(all_results)
    df.to_csv("final_attack_metrics.csv", index=False)
    save_final_security_plot(df, output_path="results/final_security_metrics.png")

    print("\nALL ATTACK SIMULATIONS COMPLETED")
    print("Results saved to final_attack_metrics.csv")
    print("Security plot saved to results/final_security_metrics.png")


# ==========================================================
#                     MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_training_and_simulation()
