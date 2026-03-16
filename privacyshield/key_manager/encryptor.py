"""
encryptor.py
------------
Encrypts the token map and saves it as a .privacyshield file.

The token map looks like:
    {"NAME_1": "John Smith", "SSN_1": "123-45-6789", "IBAN_1": "CH44 3199..."}

We encrypt this so the redacted PDF can be safely shared — even if someone
gets the .privacyshield file, they cannot read it without the key.

CRITICAL SECURITY RULE:
    The .privacyshield file and the encryption key must NEVER be stored
    in the same place. The app saves the file; the USER saves the key.

Flow:
    1. Generate a unique Fernet key (random, 32 bytes)
    2. Convert token map → JSON string → encrypt with Fernet
    3. Save encrypted bytes to .privacyshield file
    4. Return the key to the caller (shown to user as base64 string)
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
    """
    Generate a new random Fernet encryption key.

    Returns
    -------
    bytes
        32-byte Fernet key (URL-safe base64 encoded internally).

    Notes
    -----
    Each document should get its own unique key.
    Never reuse keys across documents.
    For a multi-page PDF, call this ONCE and reuse for all pages.
    """
    return Fernet.generate_key()


def key_to_string(key: bytes) -> str:
    """
    Convert a Fernet key (bytes) to a human-readable base64 string.

    This is what you display to the user so they can copy and save it.

    Parameters
    ----------
    key : bytes
        Raw Fernet key bytes.

    Returns
    -------
    str
        Base64 string representation — safe to display/copy.

    Example
    -------
    >>> key = generate_key()
    >>> print(key_to_string(key))
    'dGhpcyBpcyBhIHRlc3Qga2V5IGZvciBkZW1v...'
    """
    return base64.b64encode(key).decode("utf-8")


def string_to_key(key_str: str) -> bytes:
    """
    Convert a base64 string back to Fernet key bytes.

    This is the reverse of key_to_string — used when the user
    pastes their key back into the app to decrypt.

    Parameters
    ----------
    key_str : str
        Base64 string from key_to_string().

    Returns
    -------
    bytes
        Fernet key bytes.
    """
    return base64.b64decode(key_str.encode("utf-8"))


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Encrypt arbitrary bytes using Fernet key bytes."""
    if not data:
        raise ValueError("No data provided to encrypt.")
    return Fernet(key).encrypt(data)


def encrypt_token_map(
    token_map: Dict[str, str],
    output_path: str | Path,
    key: bytes | None = None,
) -> bytes:
    """
    Encrypt the token map and save it to a .privacyshield file.

    Parameters
    ----------
    token_map : Dict[str, str]
        Mapping of token IDs to original PII values.
        Example: {"NAME_1": "John Smith", "SSN_1": "123-45-6789"}
    output_path : str or Path
        Where to save the encrypted file.
        Should end in .privacyshield (e.g., "invoice_redacted.privacyshield")
    key : bytes, optional
        Fernet key to use. If None, a new key is generated automatically.
        Pass an existing key when processing multi-page PDFs so all pages
        share the same key.

    Returns
    -------
    bytes
        The Fernet encryption key.
        IMPORTANT: Show this to the user immediately — it is NOT saved to disk.

    Raises
    ------
    ValueError
        If token_map is empty or output_path is invalid.
    IOError
        If the file cannot be written.

    Example
    -------
    >>> token_map = {"NAME_1": "Lara Meier", "IBAN_1": "CH44 3199 9123 0000 5512 8"}
    >>> key = encrypt_token_map(token_map, "invoice.privacyshield")
    >>> print("Show this to user:", key_to_string(key))
    """
    if not token_map:
        raise ValueError("token_map is empty — nothing to encrypt.")

    output_path = Path(output_path)

    # Generate key if not provided
    if key is None:
        key = generate_key()
        logger.info("Generated new Fernet encryption key.")

    # Step 1: Convert token map to JSON bytes
    json_bytes = json.dumps(token_map, ensure_ascii=False).encode("utf-8")
    logger.debug(f"Token map serialized: {len(json_bytes)} bytes")

    # Step 2: Encrypt with Fernet
    fernet = Fernet(key)
    encrypted_bytes = fernet.encrypt(json_bytes)
    logger.debug(f"Encrypted size: {len(encrypted_bytes)} bytes")

    # Step 3: Save to .privacyshield file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encrypted_bytes)
    logger.info(f"Saved encrypted token map to: {output_path}")

    # Step 4: Return key — caller must show this to the user!
    # We do NOT save the key anywhere.
    return key
