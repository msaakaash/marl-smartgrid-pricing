# security/secure_channel.py

import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """Lightweight AEAD secure communication using ChaCha20-Poly1305.
       Prints encryption/decryption logs for educational/training demonstration.
    """

    def __init__(self, key_path: str = "security/keys/secret.key", verbose: bool = True):
        with open(key_path, "rb") as f:
            self.key = f.read()

        self.aead = ChaCha20Poly1305(self.key)
        self.verbose = verbose

    # ----------------------------------------------------------------------
    #                    ORIGINAL DICT-BASED API (USED BY AGGREGATOR)
    # ----------------------------------------------------------------------
    def encrypt(self, signal: np.ndarray) -> dict:
        """Encrypt a numpy float32 vector and return a dict{'nonce','ciphertext'}."""
        nonce = os.urandom(12)
        plaintext = signal.astype(np.float32).tobytes()
        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        if self.verbose:
            print("\n🔐 ENCRYPT (dict) EVENT")
            print("Nonce:", nonce.hex())
            print("Plaintext:", signal)
            print("Ciphertext:", ciphertext.hex()[:80], "...")

        return {"nonce": nonce, "ciphertext": ciphertext}

    def decrypt(self, packet: dict) -> np.ndarray:
        """Decrypt AEAD packet dict. If authentication fails → safe fallback."""
        try:
            plaintext = self.aead.decrypt(
                packet["nonce"],
                packet["ciphertext"],
                None
            )

            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.verbose:
                print("\n🔓 DECRYPT (dict) EVENT — SUCCESS")
                print("Nonce:", packet["nonce"].hex())
                print("Decrypted Signal:", arr)

            return arr

        except Exception:
            safe = np.array([0.0, 0.0], dtype=np.float32)
            print("\n⚠️  DECRYPT (dict) FAILED — ATTACK DETECTED!")
            print("Packet dropped. Using safe fallback:", safe)
            return safe

    # ----------------------------------------------------------------------
    #                BYTES-BASED API (USED BY CONSUMERS)
    #   packet = nonce(12 bytes) || ciphertext(...)  — convenient for RL.
    # ----------------------------------------------------------------------
    def encrypt_vector(self, signal: np.ndarray) -> bytes:
        """Encrypt a numpy float32 vector and return bytes: nonce + ciphertext."""
        nonce = os.urandom(12)
        plaintext = signal.astype(np.float32).tobytes()
        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        if self.verbose:
            print("\n🔐 ENCRYPT_VECTOR (bytes) EVENT")
            print("Nonce:", nonce.hex())
            print("Plaintext:", signal)
            print("Ciphertext:", ciphertext.hex()[:80], "...")

        return nonce + ciphertext

    def decrypt_vector(self, packet: bytes) -> np.ndarray:
        """Decrypt a bytes packet [nonce(12) + ciphertext] → numpy float32 array."""
        nonce = packet[:12]
        ciphertext = packet[12:]

        try:
            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.verbose:
                print("\n🔓 DECRYPT_VECTOR (bytes) EVENT — SUCCESS")
                print("Nonce:", nonce.hex())
                print("Decrypted:", arr)

            return arr

        except Exception:
            safe = np.array([0.0, 0.0], dtype=np.float32)
            print("\n⚠️  DECRYPT_VECTOR FAILED — ATTACK DETECTED!")
            print("Using safe fallback:", safe)
            return safe
