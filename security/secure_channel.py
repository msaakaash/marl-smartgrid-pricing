import os
import binascii
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class SecureChannel:
    """
    ChaCha20-Poly1305 authenticated encryption.
    Packet = NONCE(12 bytes) || CIPHERTEXT (8 bytes) || TAG (16 bytes)
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
    #                  FORMATTERS (HEX + BINARY WRAPPED)
    # ============================================================

    def _hex_lines(self, b: bytes, width_pairs=16):
        """
        Return wrapped hex rows.
        width_pairs = number of hex byte pairs per row.
        """
        hex_str = binascii.hexlify(b).decode()
        pairs = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]
        rows = [" ".join(pairs[i:i + width_pairs]) for i in range(0, len(pairs), width_pairs)]
        return rows

    def _binary_lines(self, b: bytes, width_bits=32):
        """
        Return wrapped binary rows.
        width_bits = number of bits per row (32 = 4 bytes)
        """
        bits = "".join(f"{byte:08b}" for byte in b)
        rows = [bits[i:i + width_bits] for i in range(0, len(bits), width_bits)]
        return rows

    def _print_table(self, title, field_name, b: bytes):
        """
        Nicely formatted table:
        Field | Hex (wrapped) | Binary (wrapped)
        """
        hex_rows = self._hex_lines(b)
        bin_rows = self._binary_lines(b)

        max_rows = max(len(hex_rows), len(bin_rows))

        print(f"\n===== {title} =====")
        print(f"{'Field':10} | {'Hex':45} | Binary")
        print("-" * 105)

        for i in range(max_rows):
            field_col = field_name if i == 0 else ""
            hex_col = hex_rows[i] if i < len(hex_rows) else ""
            bin_col = bin_rows[i] if i < len(bin_rows) else ""
            print(f"{field_col:10} | {hex_col:45} | {bin_col}")

    # ============================================================
    #                           ENCRYPT
    # ============================================================

    def encrypt(self, vec: np.ndarray) -> bytes:
        """Encrypt vector and print PT + NONCE + CT + TAG tables."""
        try:
            vec = np.asarray(vec, dtype=np.float32)
            plaintext = vec.tobytes()
            nonce = os.urandom(12)

            ciphertext_full = self.aead.encrypt(nonce, plaintext, None)
            ct_only = ciphertext_full[:-16]   # encrypted bytes
            tag_only = ciphertext_full[-16:]  # auth tag

            packet = nonce + ciphertext_full

            if self.debug:
                print("\n================= ENCRYPT =================\n")

                # PT table
                print("===== PLAINTEXT (PT) =====")
                print(f"Values: {vec.tolist()}")
                self._print_table("PT BYTES", "PT", plaintext)

                # NONCE
                self._print_table("NONCE", "NONCE", nonce)

                # CT
                self._print_table("CIPHERTEXT", "CT", ct_only)

                # TAG
                self._print_table("AUTH TAG", "TAG", tag_only)

                # FULL PACKET
                self._print_table("FULL PACKET", "PACKET", packet)

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
        """Decrypt packet and display CT/PT in structured table format."""
        try:
            if len(packet) < 13:
                raise ValueError("Packet too short.")

            nonce = packet[:12]
            ciphertext_full = packet[12:]

            ct_only = ciphertext_full[:-16]
            tag_only = ciphertext_full[-16:]

            plaintext = self.aead.decrypt(nonce, ciphertext_full, None)
            arr = np.frombuffer(plaintext, dtype=np.float32)

            if self.debug:
                print("\n================= DECRYPT =================")

                # NONCE
                self._print_table("NONCE", "NONCE", nonce)

                # CT
                self._print_table("CIPHERTEXT", "CT", ct_only)

                # TAG
                self._print_table("AUTH TAG", "TAG", tag_only)

                # PT
                self._print_table("PLAINTEXT (PT)", "PT", plaintext)
                print(f"Recovered PT values: {arr.tolist()}")
                print("Auth: VALID ✓")
                print("===========================================\n")

            return arr

        except Exception as e:
            if self.debug:
                print("\n================= DECRYPT FAILED =================")

                nonce = packet[:12] if len(packet) >= 12 else b""
                ciphertext_full = packet[12:] if len(packet) > 12 else b""

                ct_only = ciphertext_full[:-16] if len(ciphertext_full) >= 16 else ciphertext_full
                tag_only = ciphertext_full[-16:] if len(ciphertext_full) >= 16 else b""

                self._print_table("NONCE", "NONCE", nonce)
                self._print_table("CIPHERTEXT", "CT", ct_only)
                self._print_table("AUTH TAG", "TAG", tag_only)

                print("Auth: FAILED ✗ (tampered or corrupted!)")
                print("Returning SAFE PT: [0.0, 0.0]")
                print("====================================================\n")

            return np.array([0.0, 0.0], dtype=np.float32)
