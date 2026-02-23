import random


class AttackSimulator:
    def __init__(self, attack_type="none", intensity=0.0):
        """
        attack_type:
            - sybil
            - false_data_injection
            - bit_flip
            - ddos
            - replay
            - none
        intensity: 0.0 - 1.0
        """
        self.attack_type = str(attack_type).lower()
        self.intensity = max(0.0, min(1.0, float(intensity)))
        self._last_packet = None

    def apply(self, packet: bytes) -> bytes:
        if not isinstance(packet, (bytes, bytearray)):
            return b""

        packet = bytes(packet)

        if self.attack_type == "false_data_injection":
            return self._false_data_injection(packet)
        if self.attack_type == "bit_flip":
            return self._bit_flip(packet)
        if self.attack_type == "sybil":
            return self._sybil(packet)
        if self.attack_type == "ddos":
            return self._ddos(packet)
        if self.attack_type == "replay":
            return self._replay(packet)
        return packet

    # ---------------- False Data Injection ----------------
    def _false_data_injection(self, packet: bytes) -> bytes:
        if random.random() > self.intensity or len(packet) == 0:
            return packet
        tampered = bytearray(packet)
        # Corrupt one byte to break authentication/integrity.
        tampered[-1] ^= 0xFF
        return bytes(tampered)

    # ---------------- Bit Flip ----------------
    def _bit_flip(self, packet: bytes) -> bytes:
        if random.random() > self.intensity or len(packet) == 0:
            return packet
        tampered = bytearray(packet)
        idx = random.randint(0, len(tampered) - 1)
        tampered[idx] ^= 1 << random.randint(0, 7)
        return bytes(tampered)

    # ---------------- Sybil ----------------
    def _sybil(self, packet: bytes) -> bytes:
        if random.random() > self.intensity:
            return packet
        # Duplicate payload to simulate duplicated identities/traffic.
        return packet + packet

    # ---------------- DDoS ----------------
    def _ddos(self, packet: bytes) -> bytes:
        if random.random() > self.intensity:
            return packet
        # Drop packet to simulate denial of service.
        return b""

    # ---------------- Replay ----------------
    def _replay(self, packet: bytes) -> bytes:
        # Replay a previously seen packet with attack probability.
        if self._last_packet is not None and random.random() <= self.intensity:
            return self._last_packet
        self._last_packet = packet
        return packet
