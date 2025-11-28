# secure_channel.py
import os
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class SecureChannel:
    def _init_(self, key_path="security/keys/secret.key", debug=False):
        with open(key_path, "rb") as f:
            self.key = f.read()
        self.aead = ChaCha20Poly1305(self.key)
        self.debug = debug

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        ciphertext = self.aead.encrypt(nonce, plaintext, None)
        packet = nonce + ciphertext

        if self.debug:
            print("\n🔐 [ENCRYPT]")
            print(f"Nonce: {nonce.hex()}")
            print(f"Plaintext: {plaintext.decode(errors='ignore')}")
            print(f"Ciphertext: {ciphertext.hex()}")
            print(f"Packet (nonce+ciphertext): {packet.hex()}")

        return packet

    def decrypt(self, packet: bytes) -> bytes:
        nonce = packet[:12]
        ciphertext = packet[12:]
        plaintext = self.aead.decrypt(nonce, ciphertext, None)

        if self.debug:
            print("\n🔓 [DECRYPT]")
            print(f"Nonce: {nonce.hex()}")
            print(f"Ciphertext: {ciphertext.hex()}")
            print(f"Plaintext: {plaintext.decode(errors='ignore')}")

        return plaintext