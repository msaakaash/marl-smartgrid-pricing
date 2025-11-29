# secure_channel.py
import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    def __init__(self, key_path="security/keys/secret.key", debug=True):
        """Secure AEAD encryption channel"""
        with open(key_path, "rb") as f:
            self.key = f.read()

        self.aead = ChaCha20Poly1305(self.key)
        self.debug = debug

    # ---------------------------------------------------------
    # Encrypt numeric vector (aggregator output)
    # ---------------------------------------------------------
    def encrypt_vector(self, vector):
        """
        Input: Python list/np array → floats
        Output: packet (bytes)
        """
        vec = np.array(vector, dtype=np.float32)
        plaintext = vec.tobytes()
        nonce = os.urandom(12)
        ciphertext = self.aead.encrypt(nonce, plaintext, None)

        packet = nonce + ciphertext

        if self.debug:
            print("\n🔐 === ENCRYPTION (Aggregator → Consumer) ===")
            print(f"Input Vector: {vector}")
            print(f"Nonce: {nonce.hex()}")
            print(f"Ciphertext: {ciphertext.hex()}")
            print(f"Sent Packet: {packet.hex()}")

        return packet

    # ---------------------------------------------------------
    # Decrypt packet (consumer input)
    # ---------------------------------------------------------
    def decrypt_vector(self, packet):
        """
        Input: packet = nonce + ciphertext
        Output: np array of float32
        """

        if packet.startswith(b"PLAINTEXT:"):
            # Fallback plaintext
            raw = packet.replace(b"PLAINTEXT:", b"")
            vec = np.array(list(map(float, raw.decode().split(","))), dtype=np.float32)

            if self.debug:
                print("\n🟡 PLAINTEXT FALLBACK (No Encryption)")
                print(f"Vector: {vec}")

            return vec

        try:
            nonce = packet[:12]
            ciphertext = packet[12:]
            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.debug:
                print("\n🔓 === DECRYPTION (Consumer Receives) ===")
                print(f"Nonce: {nonce.hex()}")
                print(f"Ciphertext: {ciphertext.hex()}")
                print(f"Recovered Vector: {arr}")

            return arr

        except Exception as e:
            print("\n🚨 **TAMPERING DETECTED** — Returning SAFE vector")
            print(f"Error: {e}")
            return np.zeros(2, dtype=np.float32)
