# security/secure_channel.py

import os
import json
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    Lightweight AEAD encryption handler using ChaCha20-Poly1305.
    Used to protect communication between Aggregator <-> Consumers.
    """

    def __init__(self, key_path=None):
        # Default key location: security/keys/secret.key
        if key_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))   # security/
            key_dir = os.path.join(base_dir, "keys")
            key_path = os.path.join(key_dir, "secret.key")

        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"[SecureChannel] ERROR: Key file not found at {key_path}\n"
                f"Run `python security/generate_key.py` first."
            )

        with open(key_path, "rb") as f:
            self.key = f.read()

        if len(self.key) != 32:
            raise ValueError("[SecureChannel] Key must be 32 bytes.")

        self.aead = ChaCha20Poly1305(self.key)

    # --------------------------------------------------------
    # ENCRYPT
    # --------------------------------------------------------
    def encrypt(self, message: dict) -> dict:
        """
        Encrypts a Python dict and returns:
        {
            "nonce": ...,
            "ciphertext": ...,
            "tag": ...
        }
        """
        # Serialize dict → bytes
        plaintext = json.dumps(message).encode("utf-8")

        # 96-bit (12-byte) random nonce
        nonce = os.urandom(12)

        # AEAD encryption → ciphertext + tag
        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        # Return Base64-safe (bytes → hex)
        return {
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex()
        }

    # --------------------------------------------------------
    # DECRYPT
    # --------------------------------------------------------
    def decrypt(self, packet: dict) -> dict | None:
        """
        Decrypts incoming packet.
        Returns dict if valid,
        Returns None if authentication fails (attack detected).
        """
        try:
            nonce = bytes.fromhex(packet["nonce"])
            ciphertext = bytes.fromhex(packet["ciphertext"])

            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            return json.loads(plaintext.decode("utf-8"))

        except Exception:
            # Authentication failed (tampering, corruption, replay, etc.)
            return None
