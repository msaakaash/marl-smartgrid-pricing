import numpy as np
import random

class AttackSimulator:
    def __init__(self, attack_type="none", intensity=0.0):
        """
        attack_type: fdi | bit_flip | sybil | none
        intensity: 0.0 – 1.0
        """
        self.attack_type = attack_type
        self.intensity = intensity

    def apply(self, packet: bytes) -> bytes:
        if self.attack_type == "fdi":
            return self._false_data_injection(packet)
        elif self.attack_type == "bit_flip":
            return self._bit_flip(packet)
        elif self.attack_type == "sybil":
            return self._sybil(packet)
        return packet

    # ---------------- FDIA ----------------
    def _false_data_injection(self, packet: bytes) -> bytes:
        if random.random() > self.intensity:
            return packet
        tampered = bytearray(packet)
        tampered[-1] ^= 0xFF   # corrupt last byte
        return bytes(tampered)

    # ---------------- Bit Flip ----------------
    def _bit_flip(self, packet: bytes) -> bytes:
        if random.random() > self.intensity:
            return packet
        tampered = bytearray(packet)
        idx = random.randint(0, len(tampered)-1)
        tampered[idx] ^= 1 << random.randint(0, 7)
        return bytes(tampered)

    # ---------------- Sybil ----------------
    def _sybil(self, packet: bytes) -> bytes:
        if random.random() > self.intensity:
            return packet
        # duplicate packet pretending multiple identities
        return packet + packet
