# security/secure_channel.py
import os
import binascii
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    ChaCha20-Poly1305 authenticated encryption.
    Packet = NONCE(12 bytes) || CIPHERTEXT+TAG.
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

    # ============================================================
    #                           ENCRYPT
    # ============================================================
    def encrypt(self, vec: np.ndarray) -> bytes:
        """Encrypt float32 vector and SHOW PT/CT clearly."""
        try:
            vec = np.asarray(vec, dtype=np.float32)
            plaintext = vec.tobytes()
            nonce = os.urandom(12)

            ciphertext = self.aead.encrypt(nonce, plaintext, None)
            packet = nonce + ciphertext

            if self.debug:
                hex_nonce = binascii.hexlify(nonce).decode()
                hex_ct = binascii.hexlify(ciphertext).decode()

                print("\n================= ENCRYPT =================")
                print(f"Plaintext (PT):     {vec.tolist()}")
                print(f"Nonce (hex):        {hex_nonce}")
                print(f"Ciphertext length:  {len(ciphertext)} bytes")
                print(f"Ciphertext (CT):    {hex_ct[:80]}...")  # truncated for readability
                print("===========================================\n")

            return packet

        except Exception as e:
            if self.debug:
                print(f"[SecureChannel:ENCRYPT] ERROR: {e}")
            return b""

    # ============================================================
    #                           DECRYPT
    # ============================================================
    def decrypt_vector(self, packet: bytes) -> np.ndarray:
        """Decrypt packet and show PT/CT. If auth fails → safe vector."""
        try:
            if len(packet) < 13:
                raise ValueError("Packet too short.")

            nonce = packet[:12]
            ciphertext = packet[12:]

            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.debug:
                hex_nonce = binascii.hexlify(nonce).decode()
                hex_ct = binascii.hexlify(ciphertext).decode()

                print("\n================= DECRYPT =================")
                print(f"Nonce (hex):        {hex_nonce}")
                print(f"Ciphertext (CT):    {hex_ct[:80]}...")
                print(f"Recovered PT:       {arr.tolist()}")
                print("Auth:               VALID ✓")
                print("===========================================\n")

            return arr

        except Exception as e:
            if self.debug:
                nonce = packet[:12] if len(packet) >= 12 else b""
                hex_nonce = binascii.hexlify(nonce).decode() if nonce else "N/A"
                ct = packet[12:] if len(packet) > 12 else b""
                hex_ct = binascii.hexlify(ct).decode() if ct else "N/A"

                print("\n================= DECRYPT =================")
                print(f"Nonce (hex):        {hex_nonce}")
                print(f"Ciphertext (CT):    {hex_ct[:80]}...")
                print("Auth:               FAILED ✗ (Tampered or corrupted!)")
                print("Returning SAFE PT:  [0.0, 0.0]")
                print("===========================================\n")

            return np.array([0.0, 0.0], dtype=np.float32)

    # ============================================================
    #                   ATTACK SIMULATOR (FDI)
    # ============================================================
    def attacker_tamper(self, packet: bytes, mode="flip", intensity=0.10):
        """
        Simulate cyberattacks:
        - flip: invert bits
        - noise: random byte values
        - replace: overwrite entire ciphertext
        """
        if not isinstance(packet, (bytes, bytearray)):
            raise ValueError("attacker_tamper: expected bytes")

        packet = bytearray(packet)
        tamper_bytes = max(1, int(len(packet) * intensity))

        if mode == "flip":
            for _ in range(tamper_bytes):
                idx = np.random.randint(0, len(packet))
                packet[idx] ^= 0xFF

        elif mode == "noise":
            for _ in range(tamper_bytes):
                idx = np.random.randint(0, len(packet))
                packet[idx] = np.random.randint(0, 256)

        elif mode == "replace":
            nonce = packet[:12]
            cipher = os.urandom(len(packet) - 12)
            packet = nonce + cipher
            tamper_bytes = len(cipher)

        else:
            raise ValueError(f"Unknown attack mode: {mode}")

        if self.debug:
            print("\n================= ATTACK =================")
            print(f"Mode:               {mode}")
            print(f"Intensity:          {intensity}")
            print(f"Tampered bytes:     {tamper_bytes}")
            print("===========================================\n")

        return bytes(packet)