"""
encryptor.py
------------
Encrypts the token map and saves it as a .privacyshield file.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Dict

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


def generate_key() -> bytes:
    """Generate a new random Fernet encryption key."""
    return Fernet.generate_key()


def key_to_string(key: bytes) -> str:
    """Convert a Fernet key (bytes) to a human-readable base64 string."""
    return base64.b64encode(key).decode("utf-8")


def string_to_key(key_str: str) -> bytes:
    """Convert a base64 string back to Fernet key bytes."""
    return base64.b64decode(key_str.encode("utf-8"))


def encrypt_token_map(
    token_map: Dict[str, str],
    output_path: str | Path,
    key: bytes | None = None,
) -> bytes:
    """
    Encrypt the token map and save it to a .privacyshield file.

    Returns the Fernet key — show this to the user immediately,
    it is NOT saved to disk.
    """
    if not token_map:
        raise ValueError("token_map is empty — nothing to encrypt.")

    output_path = Path(output_path)

    if key is None:
        key = generate_key()
        logger.info("Generated new Fernet encryption key.")

    json_bytes = json.dumps(token_map, ensure_ascii=False).encode("utf-8")
    fernet = Fernet(key)
    encrypted_bytes = fernet.encrypt(json_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encrypted_bytes)
    logger.info(f"Saved encrypted token map to: {output_path}")

    return key


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """
    Encrypt raw bytes using a Fernet key.
    Used by app.py to embed encrypted original PDF into redacted file.
    """
    fernet = Fernet(key)
    return fernet.encrypt(data)