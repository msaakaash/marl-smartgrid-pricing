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
        - replay
        - delay
        - flip / noise / replace (existing)
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

        # buffer for replay attack
        self.replay_buffer = deque(maxlen=50)

    # ============================================================
    #                           ENCRYPT
    # ============================================================
    def encrypt(self, vec: np.ndarray) -> bytes:
        """Encrypt float32 vector."""
        try:
            vec = np.asarray(vec, dtype=np.float32)
            plaintext = vec.tobytes()
            nonce = os.urandom(12)

            ciphertext = self.aead.encrypt(nonce, plaintext, None)
            packet = nonce + ciphertext

            if self.debug:
                print("\n================= ENCRYPT =================")
                print(f"Plaintext:          {vec.tolist()}")
                print(f"Nonce (hex):        {binascii.hexlify(nonce).decode()}")
                print(f"Ciphertext length:  {len(ciphertext)} bytes")
                print("===========================================\n")

            # store packet for replay attack
            self.replay_buffer.append(packet)

            return packet

        except Exception as e:
            if self.debug:
                print(f"[SecureChannel:ENCRYPT] ERROR: {e}")
            return b""

    # ============================================================
    #                           DECRYPT
    # ============================================================
    def decrypt(self, packet: bytes) -> np.ndarray:
        """Decrypt packet. If auth fails → return safe vector."""
        try:
            if len(packet) < 13:
                raise ValueError("Packet too short")

            nonce = packet[:12]
            ciphertext = packet[12:]

            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.debug:
                print("\n================= DECRYPT =================")
                print(f"Nonce (hex):        {binascii.hexlify(nonce).decode()}")
                print(f"Recovered PT:       {arr.tolist()}")
                print("Auth:               VALID ✓")
                print("===========================================\n")

            return arr

        except Exception:
            if self.debug:
                print("\n================= DECRYPT =================")
                print("Auth:               FAILED ✗ (Tampered or stale)")
                print("Returning SAFE PT:  [0.0, 0.0]")
                print("===========================================\n")

            return np.array([0.0, 0.0], dtype=np.float32)

    # ============================================================
    #                   ATTACK SIMULATOR
    # ============================================================
    def attacker_tamper(self, packet: bytes, mode="flip", intensity=0.10):
        """
        Attack modes:
            - replay  : resend old valid packet
            - delay   : delay packet delivery
            - flip    : bit-flip
            - noise   : random byte corruption
            - replace : replace ciphertext
        """

        if not isinstance(packet, (bytes, bytearray)):
            raise ValueError("attacker_tamper expects bytes")

        packet = bytearray(packet)

        # ===================== REPLAY ATTACK =====================
        if mode == "replay":
            if len(self.replay_buffer) == 0:
                return bytes(packet)

            replayed = random.choice(list(self.replay_buffer))

            if self.debug:
                print("\n================= ATTACK =================")
                print("Mode:               REPLAY")
                print("Replaying old valid encrypted packet")
                print("===========================================\n")

            return replayed

        # ===================== DELAY ATTACK ======================
        elif mode == "delay":
            delay_time = random.uniform(0.2, 1.0)
            time.sleep(delay_time)

            if self.debug:
                print("\n================= ATTACK =================")
                print("Mode:               DELAY")
                print(f"Delay applied:      {delay_time:.2f} seconds")
                print("===========================================\n")

            return bytes(packet)

        # ===================== BIT FLIP ==========================
        elif mode == "flip":
            tamper_bytes = max(1, int(len(packet) * intensity))
            for _ in range(tamper_bytes):
                idx = random.randint(0, len(packet) - 1)
                packet[idx] ^= 0xFF

        # ===================== NOISE =============================
        elif mode == "noise":
            tamper_bytes = max(1, int(len(packet) * intensity))
            for _ in range(tamper_bytes):
                idx = random.randint(0, len(packet) - 1)
                packet[idx] = random.randint(0, 255)

        # ===================== REPLACE ===========================
        elif mode == "replace":
            nonce = packet[:12]
            cipher = os.urandom(len(packet) - 12)
            packet = nonce + cipher

        else:
            raise ValueError(f"Unknown attack mode: {mode}")

        if self.debug:
            print("\n================= ATTACK =================")
            print(f"Mode:               {mode.upper()}")
            print("Packet was tampered")
            print("===========================================\n")

        return bytes(packet)
