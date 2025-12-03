# security/secure_channel.py
import os
import binascii
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    Lightweight authenticated encryption wrapper around ChaCha20-Poly1305.

    Packet format:
        [ 12-byte NONCE | CIPHERTEXT+TAG ]
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

    # -----------------------------------------------------------
    #                       ENCRYPTION
    # -----------------------------------------------------------
    def encrypt(self, vec: np.ndarray) -> bytes:
        """
        Encrypt a float32 numpy vector → return packet: nonce || ciphertext.
        """
        try:
            vec = np.asarray(vec, dtype=np.float32)
            plaintext = vec.tobytes()
            nonce = os.urandom(12)

            ciphertext = self.aead.encrypt(nonce, plaintext, None)
            packet = nonce + ciphertext

            if self.debug:
                print(
                    f"[SecureChannel:ENCRYPT] nonce={binascii.hexlify(nonce).decode()} "
                    f"cipher={binascii.hexlify(ciphertext[:16]).decode()}...({len(ciphertext)} bytes) "
                    f"vec={vec.tolist()}"
                )

            return packet

        except Exception as e:
            if self.debug:
                print(f"[SecureChannel:ENCRYPT] ERROR: {e}")
            return b""

    # -----------------------------------------------------------
    #                       DECRYPTION
    # -----------------------------------------------------------
    def decrypt(self, packet: bytes) -> np.ndarray:
        """
        Decrypt packet (nonce || ciphertext). On auth failure → return safe vector.
        """
        try:
            if len(packet) < 13:  # too small to contain nonce+ciphertext
                raise ValueError("Packet too short.")

            nonce = packet[:12]
            ciphertext = packet[12:]

            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.debug:
                print(
                    f"[SecureChannel:DECRYPT] OK nonce={binascii.hexlify(nonce).decode()} "
                    f"arr={arr.tolist()}"
                )

            return arr

        except Exception as e:
            if self.debug:
                print(f"[SecureChannel:DECRYPT] FAILED auth: {e} → returning safe [0.0, 0.0]")

            return np.array([0.0, 0.0], dtype=np.float32)

    # -----------------------------------------------------------
    #               ATTACKER TAMPERING (FDI Simulation)
    # -----------------------------------------------------------
    def attacker_tamper(self, packet: bytes, mode="flip", intensity=0.10):
        """
        Modify an encrypted packet to simulate cyberattacks:

        mode:
            "flip"    → flip random bits (FDI attack)
            "noise"   → randomize random bytes
            "replace" → replace ciphertext entirely

        intensity (0.0–1.0):
            Fraction of packet bytes to tamper.
        """

        if not isinstance(packet, (bytes, bytearray)):
            raise ValueError("attacker_tamper: expected bytes.")

        packet = bytearray(packet)
        tamper_bytes = max(1, int(len(packet) * intensity))

        if mode == "flip":
            # Flip all bits of random packet bytes
            for _ in range(tamper_bytes):
                idx = np.random.randint(0, len(packet))
                packet[idx] ^= 0xFF

        elif mode == "noise":
            # Replace selected bytes with random values
            for _ in range(tamper_bytes):
                idx = np.random.randint(0, len(packet))
                packet[idx] = np.random.randint(0, 256)

        elif mode == "replace":
            # Replace entire ciphertext portion — catastrophic FDI attack
            nonce = packet[:12]
            cipher = os.urandom(len(packet) - 12)
            return nonce + cipher

        else:
            raise ValueError(f"Unknown attack mode: {mode}")

        if self.debug:
            print(
                f"[SecureChannel:ATTACK] mode={mode} intensity={intensity} "
                f"tampered_bytes={tamper_bytes}"
            )

        return bytes(packet)
