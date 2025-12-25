import numpy as np
import time

class NetworkMetrics:
    def __init__(self):
        self.sent = 0
        self.received = 0
        self.dropped = 0
        self.fail_times = []
        self.delays = []
        self.arrival_times = []
        self.start_time = time.time()

    def record_send(self):
        self.sent += 1
        self.arrival_times.append(time.time())

    def record_receive(self, success, delay):
        if success:
            self.received += 1
            self.delays.append(delay)
        else:
            self.dropped += 1
            self.fail_times.append(time.time())

    def results(self, par, lifetime_threshold=0.5):
        pdr = self.received / max(1, self.sent)
        accuracy = pdr
        packet_loss = self.dropped / max(1, self.sent)

        avg_delay = float(np.mean(self.delays)) if self.delays else 0.0

        mat = float(np.mean(np.diff(self.arrival_times))) if len(self.arrival_times) > 1 else 0.0

        mttf = (
            float(np.mean(np.diff(self.fail_times)))
            if len(self.fail_times) > 1 else float("inf")
        )

        resilience = 1.0 - packet_loss
        network_lifetime = (
            len(self.arrival_times)
            if pdr >= lifetime_threshold else
            next((i for i in range(len(self.arrival_times)) if pdr < lifetime_threshold), 0)
        )

        energy_efficiency = 1.0 / (par + 1e-6)

        return {
            "Accuracy": round(accuracy, 3),
            "Avg Delay (s)": round(avg_delay, 4),
            "Packet Loss": round(packet_loss, 3),
            "PDR": round(pdr, 3),
            "Energy Efficiency": round(energy_efficiency, 3),
            "MAT": round(mat, 4),
            "MTTF": round(mttf, 3) if mttf != float("inf") else "∞",
            "Resilience": round(resilience, 3),
            "Network Lifetime": network_lifetime
        }
