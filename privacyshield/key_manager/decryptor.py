"""
decryptor.py
------------
Decrypts a .privacyshield file using the user's key to restore
the original token map.

The token map is then used by redactor.restore_text() to replace
all [TOKEN_ID] placeholders back with real PII values.

Flow:
    1. Read .privacyshield file as raw bytes
    2. Decrypt bytes using the user's Fernet key
    3. Parse JSON → return token map dict

CRITICAL SECURITY RULE:
    The key is NEVER stored by the app. The user must provide it.
    If the key is lost, the original values CANNOT be recovered.
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

    Parameters
    ----------
    shield_path : str or Path
        Path to the .privacyshield file.
    key : bytes or str
        The Fernet encryption key.
        Can be raw bytes (from encrypt_token_map return value)
        or a base64 string (from key_to_string / user input).

    Returns
    -------
    Dict[str, str]
        The decrypted token map.
        Example: {"NAME_1": "John Smith", "SSN_1": "123-45-6789"}

    Raises
    ------
    FileNotFoundError
        If the .privacyshield file does not exist.
    ValueError
        If the key is wrong or the file is corrupted.

    Example
    -------
    >>> token_map = decrypt_token_map("invoice.privacyshield", key)
    >>> print(token_map)
    {"NAME_1": "Lara Meier", "IBAN_1": "CH44 3199 9123 0000 5512 8"}
    """
    shield_path = Path(shield_path)

    if not shield_path.exists():
        raise FileNotFoundError(f".privacyshield file not found: {shield_path}")

    # Accept key as either bytes or base64 string
    if isinstance(key, str):
        key = string_to_key(key)

    # Step 1: Read encrypted bytes from file
    encrypted_bytes = shield_path.read_bytes()
    logger.debug(f"Read {len(encrypted_bytes)} encrypted bytes from {shield_path}")

    # Step 2: Decrypt
    try:
        fernet = Fernet(key)
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
    except InvalidToken:
        raise ValueError(
            "Decryption failed — wrong key or corrupted .privacyshield file. "
            "Make sure you're using the exact key that was generated for this document."
        )

    # Step 3: Parse JSON → token map
    try:
        token_map = json.loads(decrypted_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Decrypted data is not valid JSON: {e}")

    logger.info(f"Successfully decrypted {len(token_map)} tokens from {shield_path}")
    return token_map


def decrypt_bytes(encrypted_payload: bytes, key: bytes | str) -> bytes:
    """Decrypt arbitrary encrypted bytes using a bytes or string key."""
    if isinstance(key, str):
        try:
            key = string_to_key(key)
        except Exception:
            raise ValueError("Invalid key format.")

    try:
        return Fernet(key).decrypt(encrypted_payload)
    except InvalidToken:
        raise ValueError(
            "Decryption failed — wrong key or corrupted encrypted payload."
        )
