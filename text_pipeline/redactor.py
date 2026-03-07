"""
PURPOSE:
    Replaces detected PII spans in text with anonymized tokens like
    [NAME_1], [SSN_1], [EMAIL_1] etc. Also builds the token→value
    mapping needed for later un-redaction.

INPUT:
    - text (str): Original plain text from PDF page
    - pii_entities (list): Output from ner_engine.detect_pii()
    - existing_token_map (dict): Optional existing token map to extend
      (used when processing multi-page documents — tokens stay consistent)

OUTPUT:
    - redacted_text (str): Text with PII replaced by tokens
      e.g. "Dear [NAME_1], your SSN is [US_SSN_1]"
    - token_map (dict): Mapping of token → original value
      e.g. {"NAME_1": "John Smith", "US_SSN_1": "123-45-6789"}

METHOD:
    1. Sort PII entities by position (right to left) to avoid offset shifts
    2. For each entity, check if same value was seen before (reuse token)
    3. If new value, create new token: {ENTITY_TYPE}_{counter}
    4. Replace text span with [TOKEN]
    5. Record token → original value mapping

    IMPORTANT: Same value always gets same token across the document.
"""

from collections import defaultdict


# Maps entity type to human-readable token prefix
TOKEN_PREFIX_MAP = {
    "PERSON": "NAME",
    "LOCATION": "ADDRESS",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "IBAN",
    "DATE_TIME": "DATE",
    "US_BANK_NUMBER": "BANK",
    "US_DRIVER_LICENSE": "DL",
    "US_ITIN": "ITIN",
    "US_PASSPORT": "PASSPORT",
    "MEDICAL_LICENSE": "MED_LICENSE",
    "FINANCIAL_AMOUNT": "AMOUNT",
    "IN_AADHAAR": "AADHAAR",
    "IN_PAN": "PAN",
    "IP_ADDRESS": "IP",
    "URL": "URL",
    "NRP": "NRP",
}


def redact_text(
    text: str,
    pii_entities: list[dict],
    existing_token_map: dict = None
) -> tuple[str, dict]:
    if not pii_entities:
        return text, existing_token_map or {}
    token_map = dict(existing_token_map) if existing_token_map else {}
    value_to_token = {v: k for k, v in token_map.items()}
    type_counters = defaultdict(int)
    for token_id in token_map.keys():
        parts = token_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
            type_counters[prefix] = max(type_counters[prefix], int(parts[1]))
    sorted_entities = sorted(pii_entities, key=lambda x: x["start"], reverse=True)
    text_chars = list(text)

    for entity in sorted_entities:
        original_value = entity["text"]
        entity_type = entity["entity_type"]
        start = entity["start"]
        end = entity["end"]
        prefix = TOKEN_PREFIX_MAP.get(entity_type, entity_type)

        if original_value in value_to_token:
            token_id = value_to_token[original_value]
        else:
            type_counters[prefix] += 1
            token_id = f"{prefix}_{type_counters[prefix]}"

            token_map[token_id] = original_value
            value_to_token[original_value] = token_id

        replacement = f"[{token_id}]"
        text_chars[start:end] = list(replacement)

    redacted_text = "".join(text_chars)
    return redacted_text, token_map


def count_redactions(token_map: dict) -> dict:
    counts = defaultdict(int)
    for token_id in token_map.keys():
        parts = token_id.rsplit("_", 1)
        if len(parts) == 2:
            counts[parts[0]] += 1
    return dict(counts)
