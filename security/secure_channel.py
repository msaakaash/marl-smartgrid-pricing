# ======================= security/secure_channel.py =======================

import os
import time
import random
import numpy as np
from collections import deque
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    ChaCha20-Poly1305 authenticated encryption for MARL smart grid control signals.

    Packet format:
        [ NONCE (12 bytes) | CIPHERTEXT + AUTH TAG ]

    Supports explicit cyber-attack simulation for robustness evaluation.
    """

    def __init__(
        self,
        key_path="security/keys/secret.key",
        metrics=None,
        debug=False
    ):
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"[SecureChannel] Key file not found: {key_path}")

        with open(key_path, "rb") as f:
            self.key = f.read()

        if len(self.key) != 32:
            raise ValueError("[SecureChannel] Key must be exactly 32 bytes.")

        self.aead = ChaCha20Poly1305(self.key)
        self.debug = debug

        # Optional metrics collector (SmartGridSecurityMetrics)
        self.metrics = metrics

        # Buffers for replay / reorder attacks
        self.replay_buffer = deque(maxlen=100)
        self.reorder_buffer = deque(maxlen=10)

    # ============================================================
    #                           ENCRYPT
    # ============================================================
    def encrypt(self, vec: np.ndarray) -> bytes:
        vec = np.asarray(vec, dtype=np.float32)
        plaintext = vec.tobytes()
        nonce = os.urandom(12)

        ciphertext = self.aead.encrypt(nonce, plaintext, None)
        packet = nonce + ciphertext

        self.replay_buffer.append(packet)

        if self.metrics:
            self.metrics.record_send()

        return packet

    # ============================================================
    #                           DECRYPT
    # ============================================================
    def decrypt(self, packet: bytes) -> np.ndarray:
        start = time.time()

        try:
            nonce = packet[:12]
            ciphertext = packet[12:]
            plaintext = self.aead.decrypt(nonce, ciphertext, None)

            delay = time.time() - start
            if self.metrics:
                self.metrics.record_receive(success=True, delay=delay)

            return np.frombuffer(plaintext, dtype=np.float32)

        except Exception:
            # Authentication failure or malformed packet
            if self.metrics:
                self.metrics.record_receive(success=False, delay=0.0)

            # Safe fallback (prevents agent crash)
            return np.array([0.0, 0.0], dtype=np.float32)

    # ============================================================
    #                   ATTACK SIMULATOR
    # ============================================================
    def attacker_tamper(self, packet: bytes, mode="flip", intensity=0.1):
        """
        Supported attacks:
        - replay, delay, flip, noise, replace
        - ddos, mitm, impersonation, blackhole, grayhole
        - selective_forward, reorder, truncate, pad
        - desync, timing, flood_amplify, adaptive
        """

        pkt = bytearray(packet)

        # ---------------- BASIC ATTACKS ----------------

        if mode == "replay":
            return random.choice(self.replay_buffer)

        if mode == "delay":
            time.sleep(random.uniform(0.005, 0.02))  # realistic delay
            return bytes(pkt)

        if mode == "flip":
            for _ in range(max(1, int(len(pkt) * intensity))):
                pkt[random.randint(0, len(pkt) - 1)] ^= 0xFF
            return bytes(pkt)

        if mode == "noise":
            for _ in range(max(1, int(len(pkt) * intensity))):
                pkt[random.randint(0, len(pkt) - 1)] = random.randint(0, 255)
            return bytes(pkt)

        if mode == "replace":
            return pkt[:12] + os.urandom(len(pkt) - 12)

        # ---------------- NETWORK-LIKE ATTACKS ----------------

        if mode == "ddos":
            return pkt * int(1 + 10 * intensity)

        if mode == "mitm":
            pkt[random.randint(12, len(pkt) - 1)] ^= 0xAA
            return bytes(pkt)

        if mode == "impersonation":
            return bytes(pkt)

        if mode == "blackhole":
            return b""

        if mode == "grayhole":
            return bytes(pkt) if random.random() > intensity else b""

        if mode == "selective_forward":
            return bytes(pkt) if random.random() > intensity else b""

        if mode == "reorder":
            self.reorder_buffer.append(pkt)
            if len(self.reorder_buffer) >= 3:
                random.shuffle(self.reorder_buffer)
                return bytes(self.reorder_buffer.popleft())
            return bytes(pkt)

        if mode == "truncate":
            cut = random.randint(1, len(pkt) // 2)
            return bytes(pkt[:-cut])

        if mode == "pad":
            return bytes(pkt + os.urandom(int(len(pkt) * intensity)))

        if mode == "desync":
            return bytes(pkt) + bytes(pkt)

        if mode == "timing":
            time.sleep(random.uniform(0.01, 0.05))
            return bytes(pkt)

        if mode == "flood_amplify":
            return pkt * random.randint(2, 5)

        if mode == "adaptive":
            return self.attacker_tamper(
                packet,
                mode=random.choice([
                    "ddos", "mitm", "replay", "delay",
                    "flip", "blackhole", "grayhole",
                    "truncate", "pad"
                ]),
                intensity=intensity
            )

        raise ValueError(f"Unknown attack mode: {mode}")
