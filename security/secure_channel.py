# secure_channel.py
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """Lightweight AEAD secure communication using ChaCha20-Poly1305.
       Prints encryption/decryption logs for educational/training demonstration.
    """

    def __init__(self, key_path="security/keys/secret.key", verbose=True):
        with open(key_path, "rb") as f:
            self.key = f.read()

        self.aead = ChaCha20Poly1305(self.key)
        self.verbose = verbose

    # ----------------------------------------------------------------------
    #                             ENCRYPTION
    # ----------------------------------------------------------------------
    def encrypt(self, signal: np.ndarray) -> dict:
        """Encrypt a numpy float32 vector (e.g., aggregator signal)."""

        nonce = np.random.bytes(12)
        plaintext = signal.astype(np.float32).tobytes()

        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        if self.verbose:
            print("\n🔐 ENCRYPTION EVENT")
            print("Nonce:", nonce.hex())
            print("Plaintext:", signal)
            print("Ciphertext:", ciphertext.hex()[:80], "...")

        return {"nonce": nonce, "ciphertext": ciphertext}

    # ----------------------------------------------------------------------
    #                             DECRYPTION
    # ----------------------------------------------------------------------
    def decrypt(self, packet: dict) -> np.ndarray:
        """Decrypt AEAD packet. If authentication fails → safe fallback."""

        try:
            plaintext = self.aead.decrypt(
                packet["nonce"],
                packet["ciphertext"],
                None
            )

            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.verbose:
                print("\n🔓 DECRYPTION EVENT — SUCCESS")
                print("Nonce:", packet["nonce"].hex())
                print("Decrypted Signal:", arr)

            return arr

        except Exception:
            safe = np.array([0.0, 0.0], dtype=np.float32)
            print("\n⚠️  DECRYPTION FAILED — ATTACK DETECTED!")
            print("Packet dropped. Using safe fallback:", safe)
            return safe
