import numpy as np

class SmartGridSecurityMetrics:
    def __init__(self):
        self.rewards = []
        self.total_packets = 0
        self.auth_failures = 0

    # ---------- Called from SecureChannel ----------
    def record_packet(self, auth_success: bool):
        self.total_packets += 1
        if not auth_success:
            self.auth_failures += 1

    def record_reward(self, reward):
        self.rewards.append(reward)

    # ---------- METRICS ----------
    def security_metrics(self):
        pdr = (
            100 * (self.total_packets - self.auth_failures) / self.total_packets
            if self.total_packets > 0 else 100.0
        )

        return {
            "Packet Delivery Ratio": round(pdr, 2),
            "Auth Failure Rate": round(
                self.auth_failures / max(1, self.total_packets), 3
            )
        }

    def learning_metrics(self):
        return {
            "Mean Reward": round(np.mean(self.rewards), 3),
            "Reward Variance": round(np.var(self.rewards), 3)
        }

    def grid_metrics(self, net_load):
        return {
            "Peak Load": round(float(np.max(net_load)), 3),
            "Average Load": round(float(np.mean(net_load)), 3),
            "Load Variance": round(float(np.var(net_load)), 3),
            "PAR": round(
                float(np.max(net_load) / (np.mean(net_load) + 1e-6)), 3
            )
        }
