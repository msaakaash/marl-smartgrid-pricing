# security/secure_channel.py

import os
import time
import random
import binascii
import numpy as np
from collections import deque
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    ChaCha20-Poly1305 authenticated encryption.
    Packet format:
        [ NONCE (12 bytes) | CIPHERTEXT + AUTH TAG ]

    Includes attack simulation:
        - original: replay, delay, flip, noise, replace
        - extended: 15 research-grade network attacks
    """

    def __init__(self, key_path="security/keys/secret.key", debug=True):
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"[SecureChannel] Key file not found: {key_path}")

        with open(key_path, "rb") as f:
            self.key = f.read()

        if len(self.key) != 32:
            raise ValueError("[SecureChannel] Key must be exactly 32 bytes.")

        self.aead = ChaCha20Poly1305(self.key)
        self.debug = debug

        # buffer for replay / reordering attacks
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
        return packet

    # ============================================================
    #                           DECRYPT
    # ============================================================
    def decrypt(self, packet: bytes) -> np.ndarray:
        try:
            nonce = packet[:12]
            ciphertext = packet[12:]
            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            return np.frombuffer(plaintext, dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

    # ============================================================
    #                   ATTACK SIMULATOR (EXTENDED)
    # ============================================================
    def attacker_tamper(self, packet: bytes, mode="flip", intensity=0.10):
        """
        ORIGINAL modes (unchanged):
          replay, delay, flip, noise, replace

        NEW complex modes:
          ddos, mitm, impersonation, blackhole, grayhole,
          selective_forward, reorder, truncate, pad,
          desync, timing, flood_amplify, adaptive
        """

        pkt = bytearray(packet)

        # ---------------- ORIGINAL MODES ----------------
        if mode == "replay":
            return random.choice(self.replay_buffer)

        if mode == "delay":
            time.sleep(random.uniform(0.005, 0.02))
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

        # ---------------- NEW COMPLEX MODES ----------------

        if mode == "ddos":
            return pkt * int(1 + 20 * intensity)

        if mode == "mitm":
            pkt[random.randint(12, len(pkt) - 1)] ^= 0xAA
            return bytes(pkt)

        if mode == "impersonation":
            return bytes(pkt)  # identity mismatch handled by receiver logic

        if mode == "blackhole":
            return b""  # drop

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
            time.sleep(random.uniform(0.01, 0.1))
            return bytes(pkt)

        if mode == "flood_amplify":
            return pkt * random.randint(2, 10)

        if mode == "adaptive":
            return self.attacker_tamper(
                packet,
                mode=random.choice([
                    "ddos", "mitm", "replay", "delay", "flip",
                    "blackhole", "grayhole", "truncate", "pad"
                ]),
                intensity=intensity
            )

        raise ValueError(f"Unknown attack mode: {mode}")
