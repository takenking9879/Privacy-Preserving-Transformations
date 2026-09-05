"""Owner secret material and optional trainer-public-key envelope.

Public/private keys do NOT replace the transform. They only wrap the released
table so a thief of the file cannot read Z. The trainer who unwraps still
sees Z, so known-pair attacks still apply to that party.

The encode function itself must stay private: publishing it would let an
attacker create chosen pairs (X, Z) at will.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class TrainerKeyPair:
    private_pem: bytes
    public_pem: bytes


def generate_trainer_keypair() -> TrainerKeyPair:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return TrainerKeyPair(private_pem=private_pem, public_pem=public_pem)


def wrap_release(Z: np.ndarray, y_tilde: np.ndarray, trainer_public_pem: bytes) -> bytes:
    """Encrypt (Z, y_tilde) to a trainer public key (RSA-OAEP + AES-GCM)."""
    public = serialization.load_pem_public_key(trainer_public_pem)
    session = os.urandom(32)
    nonce = os.urandom(12)
    payload = json.dumps(
        {"Z": np.asarray(Z, dtype=float).tolist(), "y": np.asarray(y_tilde, dtype=float).tolist()}
    ).encode("utf-8")
    blob = AESGCM(session).encrypt(nonce, payload, None)
    wrapped_key = public.encrypt(
        session,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    header = np.array([len(wrapped_key), len(nonce), len(blob)], dtype=np.int64).tobytes()
    return header + wrapped_key + nonce + blob


def unwrap_release(package: bytes, trainer_private_pem: bytes) -> tuple[np.ndarray, np.ndarray]:
    private = serialization.load_pem_private_key(trainer_private_pem, password=None)
    n_wrap, n_nonce, n_blob = np.frombuffer(package[:24], dtype=np.int64)
    off = 24
    wrapped_key = package[off : off + int(n_wrap)]
    off += int(n_wrap)
    nonce = package[off : off + int(n_nonce)]
    off += int(n_nonce)
    blob = package[off : off + int(n_blob)]
    session = private.decrypt(
        wrapped_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    payload = json.loads(AESGCM(session).decrypt(nonce, blob, None))
    return np.asarray(payload["Z"], dtype=np.float64), np.asarray(payload["y"], dtype=np.float64)
