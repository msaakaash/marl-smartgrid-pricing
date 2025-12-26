# =========================== metrics.py ===========================

import numpy as np
import time


class SmartGridSecurityMetrics:
    """
    Metrics for MARL-based Smart Grid under Cyber Attacks
    """

    def __init__(self):
        # Communication-level
        self.sent_packets = 0
        self.received_packets = 0
        self.dropped_packets = 0
        self.decryption_failures = 0
        self.delays = []

        # Learning-level
        self.rewards = []

    # ===================== COMMUNICATION =====================

    def record_send(self):
        self.sent_packets += 1

    def record_receive(self, success: bool, delay: float = 0.0):
        if success:
            self.received_packets += 1
            if delay > 0:
                self.delays.append(delay)
        else:
            self.dropped_packets += 1
            self.decryption_failures += 1

    # ===================== LEARNING =====================

    def record_reward(self, reward: float):
        self.rewards.append(reward)

    # ===================== GRID METRICS =====================

    @staticmethod
    def grid_metrics(net_load):
        net_load = np.asarray(net_load)

        peak_load = float(np.max(net_load))
        avg_load = float(np.mean(net_load))
        load_variance = float(np.var(net_load))
        par = peak_load / (avg_load + 1e-6)

        return {
            "Peak Load": round(peak_load, 3),
            "Average Load": round(avg_load, 3),
            "Load Variance": round(load_variance, 3),
            "PAR": round(par, 3)
        }

    # ===================== SECURITY SUMMARY =====================

    def security_metrics(self):
        pdr = self.received_packets / max(1, self.sent_packets)
        packet_loss = self.dropped_packets / max(1, self.sent_packets)
        rejection_rate = self.decryption_failures / max(1, self.sent_packets)
        avg_delay = float(np.mean(self.delays)) if self.delays else 0.0

        return {
            "Packet Delivery Ratio": round(pdr, 3),
            "Packet Loss Rate": round(packet_loss, 3),
            "Packet Rejection Rate": round(rejection_rate, 3),
            "Average Delay (s)": round(avg_delay, 4)
        }

    # ===================== LEARNING SUMMARY =====================

    def learning_metrics(self):
        if not self.rewards:
            return {
                "Mean Reward": 0.0,
                "Reward Variance": 0.0
            }

        return {
            "Mean Reward": round(float(np.mean(self.rewards)), 3),
            "Reward Variance": round(float(np.var(self.rewards)), 3)
        }
