# security/generate_key.py

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import os

def generate_key(path=None):
    # Automatically save inside: security/keys/secret.key
    base_dir = os.path.dirname(os.path.abspath(__file__))   # security/
    key_dir = os.path.join(base_dir, "keys")
    os.makedirs(key_dir, exist_ok=True)

    if path is None:
        path = os.path.join(key_dir, "secret.key")

    key = ChaCha20Poly1305.generate_key()  # 32-byte key

    with open(path, "wb") as f:
        f.write(key)

    print(f"Generated shared key at: {path}")

if __name__ == "__main__":
    generate_key()
