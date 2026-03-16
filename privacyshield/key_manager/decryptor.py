"""
decryptor.py
------------
Decrypts a .privacyshield file using the user's key to restore
the original token map.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from cryptography.fernet import Fernet, InvalidToken

from .encryptor import string_to_key

logger = logging.getLogger(__name__)


def decrypt_token_map(
    shield_path: str | Path,
    key: bytes | str,
) -> Dict[str, str]:
    """
    Decrypt a .privacyshield file and return the token map.

    Raises FileNotFoundError if file not found.
    Raises ValueError if key is wrong or file is corrupted.
    """
    shield_path = Path(shield_path)

    if not shield_path.exists():
        raise FileNotFoundError(f".privacyshield file not found: {shield_path}")

    if isinstance(key, str):
        key = string_to_key(key)

    encrypted_bytes = shield_path.read_bytes()

    try:
        fernet = Fernet(key)
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
    except InvalidToken:
        raise ValueError(
            "Decryption failed — wrong key or corrupted .privacyshield file."
        )

    try:
        token_map = json.loads(decrypted_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Decrypted data is not valid JSON: {e}")

    logger.info(f"Successfully decrypted {len(token_map)} tokens from {shield_path}")
    return token_map


def decrypt_bytes(encrypted_data: bytes, key: bytes | str) -> bytes:
    """
    Decrypt raw bytes using a Fernet key.
    Used by app.py to restore original PDF from encrypted payload.
    """
    if isinstance(key, str):
        key = string_to_key(key)

    try:
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)
    except InvalidToken:
        raise ValueError(
            "Decryption failed — wrong key or corrupted data."
        )