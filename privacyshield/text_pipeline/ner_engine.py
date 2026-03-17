import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from langdetect import detect as detect_language
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.35

LANGUAGE_MODELS = {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "it": "it_core_news_lg",
}
DEFAULT_LANGUAGE = "en"
ALL_LANGUAGES = list(LANGUAGE_MODELS.keys())

ORG_FALSE_POSITIVES = {
    "SSN","DOB","EIN","NPI","ID","DOJ",
    "IL","MA","CA","NY","TX",
    "AHV-Nummer","AHV-Nr","AHV","AVS","Tel","Tél","Adresse","Diagnose"
}

SWIFT_FALSE_POSITIVES = {
    "Rechnung","Paziente","Paciente","Telefono","Telefon",
    "Patient","Diagnose","Betrag","Krankenhaus"
}

PERSON_FALSE_POSITIVES = {
    "Email","Phone","Name","Address","Date","Salary","Patient",
    "Doctor","Diagnosis","Contact","Subject","From","To","Ref",
    "Policy","Claim","Invoice","Total","Amount","Note","Dear"
}

_analyzer = None

# ─── IBAN VALIDATOR ───────────────────────────────────────────────────────────

def validate_iban(iban):
    """Validate IBAN using mod-97 checksum."""
    iban = iban.replace(" ", "").replace("-", "")
    if len(iban) < 15 or len(iban) > 34:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = ""
    for ch in rearranged:
        numeric += str(int(ch, 36))
    return int(numeric) % 97 == 1

# ─── CONTEXT-BASED NUMBER DETECTOR ───────────────────────────────────────────

GENERIC_NUMBER_LABELS = [
    "number", "no.", "no:", "#", "num.", "num:", "id:", "id no",
    "nummer", "numéro", "numero", "número",
    "policy number", "policy no", "policy #", "policy#",
    "invoice number", "invoice no", "invoice #",
    "claim number", "claim no", "claim #",
    "reference number", "reference no", "ref no", "ref #",
    "account number", "account no", "account #",
    "document number", "document no", "doc no", "doc #",
    "ticket number", "ticket no", "ticket #",
    "order number", "order no", "order #",
    "case number", "case no", "case #",
    "member number", "member no", "member #",
    "patient number", "patient no", "patient id",
    "employee number", "employee no", "employee id",
    "customer number", "customer no", "customer id",
    "contract number", "contract no",
    "registration number", "registration no",
    "license number", "license no",
]

# UUID pattern — matches standard 8-4-4-4-12 hex UUID (case-insensitive)
_UUID_RE = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
)

def _extract_context_numbers(text):
    """
    Extract values that follow label words like 'policy number', 'invoice #' etc.
    Also directly extracts UUIDs anywhere in the line.
    Returns list of (start, end, value) tuples.
    """
    found = []
    text_lower = text.lower()

    for label in GENERIC_NUMBER_LABELS:
        # FIX: pattern now uses [a-z0-9] (lowercase) to match against text_lower,
        # then maps back to original text positions for the actual value.
        # Also extended value pattern to allow lowercase hex chars for UUIDs.
        pattern = re.escape(label) + r"[\s:]*([a-z0-9][a-z0-9\-\/\.]{2,50})"
        for m in re.finditer(pattern, text_lower):
            actual_start = m.start(1)
            actual_end = m.end(1)
            actual_value = text[actual_start:actual_end]
            found.append((actual_start, actual_end, actual_value))

    # FIX: Also scan for bare UUIDs on the line — these won't be caught by
    # the label loop if the label is something like "Policy #:" (handled above)
    # but is also needed as a safety net for unlabelled UUIDs in body text.
    for m in _UUID_RE.finditer(text):
        found.append((m.start(), m.end(), m.group()))

    return found

# ─── ADDRESS EXTRACTOR ────────────────────────────────────────────────────────

# Regex to capture full address lines that follow an address label
ADDRESS_LABEL_RE = re.compile(
    r"(?i)(?:address|addr|mailing\s+address|home\s+address|street\s+address|"
    r"residence|residential\s+address|adresse|anschrift|indirizzo|dirección)"
    r"\s*[:\-]?\s*"
    r"([A-Za-z0-9\s,\.\-\#\/]{10,150}?)(?=\n|$)"
)

# Regex to detect unlabelled street addresses (letterhead / recipient block style).
# Matches lines like: "0749 Stokes Vista Suite 450, North Angela, OH 43648"
# Pattern: starts with house number, has street name, optional suite/apt, city, state, zip.
ADDRESS_STREET_RE = re.compile(
    r"(?m)^\s*"
    r"(\d{1,6}"                                          # house number
    r"(?:\s+[A-Za-z0-9]+){2,6}"                         # street name words
    r"(?:\s+(?:Suite|Ste|Apt|Apt\.|Floor|Fl|Unit|#)\s*\d+[A-Za-z]?)?"  # optional unit
    r",\s*[A-Za-z\s]{2,30}"                              # city
    r",\s*[A-Z]{2}"                                      # state abbreviation
    r"\s+\d{5}(?:-\d{4})?)"                              # ZIP / ZIP+4
    r"\s*$"
)

def _extract_address_entities(text, existing_entities):
    """
    Extract full addresses that spaCy may have missed or only partially captured.

    Two strategies:
    1. Label-based  — 'Address: 123 Main St, ...'  (ADDRESS_LABEL_RE)
    2. Pattern-based — bare street lines like letterhead addresses (ADDRESS_STREET_RE)
    """
    extra = []

    def _add(start, end, value):
        if len(value.strip()) < 8:
            return
        already = any(
            e["start"] <= start and end <= e["end"]
            for e in existing_entities
        )
        if not already:
            extra.append({
                "entity_type": "LOCATION",
                "text": value.strip(),
                "start": start,
                "end": end,
                "score": 0.92
            })

    # Strategy 1: labelled addresses
    for m in ADDRESS_LABEL_RE.finditer(text):
        _add(m.start(1), m.end(1), m.group(1))

    # Strategy 2: unlabelled street addresses (letterhead style)
    for m in ADDRESS_STREET_RE.finditer(text):
        _add(m.start(1), m.end(1), m.group(1))

    return extra

# ─── CUSTOM RECOGNIZERS ───────────────────────────────────────────────────────

def _build_custom_recognizers():
    recognizers = []

    # SSN — English only
    recognizers.append(PatternRecognizer(supported_entity="US_SSN",
        supported_language="en",
        patterns=[Pattern("SSN_dashes", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
                  Pattern("SSN_plain",  r"\b\d{9}\b", 0.5)],
        context=["ssn","social","security"]))

    # Financial amounts — all languages
    # NOTE: FINANCIAL_AMOUNT entities are intentionally NOT redacted.
    # They are detected only to prevent the numbers inside them from being
    # misidentified as dates, IDs, or other PII by other recognizers.
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="FINANCIAL_AMOUNT",
            supported_language=lang,
            patterns=[
                Pattern("usd",   r"\$\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("chf",   r"CHF\s?[\d,]+(?:\.\d{2})?", 0.8),
                Pattern("eur",   r"EUR\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("gbp",   r"£\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("inr",   r"₹\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("usd_m", r"\$\s?[\d,]+/month", 0.8),
            ],
            context=["salary","income","wage","pay","benefit","amount","due",
                     "premium","gross","net","total","betrag","salaire","stipendio"]))

    # Phone numbers — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="PHONE_NUMBER",
            supported_language=lang,
            patterns=[
                Pattern("intl_phone", r"\+\d{1,3}[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}", 0.85),
                Pattern("us_phone",   r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", 0.75),
                # FIX: us_phone2 raised to score 0.95 and no context needed — parenthesised
                # format like (293)796-3030 is unambiguously a phone number regardless of label.
                Pattern("us_phone2",  r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}", 0.95),
                Pattern("us_phone3",  r"\(\d{3}\)\d{3}-\d{4}", 0.95),
                Pattern("fr_phone",   r"\+33[\s.]?\d[\s.]?(?:\d{2}[\s.]?){4}", 0.85),
            ],
            # Added "contact" and "fax" so lines like "Contact: (293)796-3030" match
            context=["tel","tél","phone","telefon","telefono","teléfono","mobile","handy",
                     "contact","fax","call","reach","helpline","hotline","number"]))

    # Email — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="EMAIL_ADDRESS",
            supported_language=lang,
            patterns=[Pattern("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", 1.0)]))

    # IBAN — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="IBAN_CODE",
            supported_language=lang,
            patterns=[Pattern("iban", r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){4,6}(?:\s?[A-Z0-9]{1,4})?\b", 0.85)],
            context=["iban","bank","konto","compte","conto","cuenta"]))

    # Prefixed ID numbers (EMP-, PAT-, TAX-, RF-, etc.)
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="ID_NUMBER",
            supported_language=lang,
            patterns=[Pattern("prefixed_id",
                r"\b(EMP|PAT|POL|MED|INV|CLM|REF|ZUR|PH|MRN|HB|TAX|ID|ACC|CUST|ORD|CASE)-[\w\d][\w\d-]*\b",
                0.85)],
            context=["employee","patient","policy","member","account","id","number","claim","tax","invoice"]))

    # UUID — all languages (case-insensitive hex, e.g. deafac4f-03f6-408e-b7c4-d038e533bff5)
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="ID_NUMBER",
            supported_language=lang,
            patterns=[Pattern("uuid",
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
                0.95)],
            context=[]))

    # RF Creditor Reference
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="ID_NUMBER",
            supported_language=lang,
            patterns=[Pattern("rf_ref", r"\bRF[A-Z0-9]{4,22}\b", 0.9)],
            context=["reference","ref","creditor","payment","remittance"]))

    # Medical conditions — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="MEDICAL_CONDITION",
            supported_language=lang,
            patterns=[
                Pattern("diagnosis_en", r"(?i)(?:diagnosis|condition|disorder):\s*([A-Z][^\n,\.]{3,50})", 0.8),
                Pattern("diagnosis_de", r"(?i)(?:diagnose|erkrankung):\s*([A-ZÄÖÜ][^\n,\.]{3,40})", 0.8),
                Pattern("condition_inline",
                    r"(?i)\b(diabetes|hypertension|bluthochdruck|asthma|cancer|krebs|"
                    r"arthritis|depression|anxiety|cardiac|cardiovascular)\b", 0.65),
            ],
            context=["diagnosis","diagnose","condition","medical","pre-existing","history"]))

    # AHV/AVS Swiss number — DE, FR, IT
    for lang in ["de","fr","it"]:
        recognizers.append(PatternRecognizer(supported_entity="CH_AHV",
            supported_language=lang,
            patterns=[Pattern("ahv", r"756[.\-]?\d{4}[.\-]?\d{4}[.\-]?\d{2}", 0.95)],
            context=["ahv","avs","swiss","sozialversicherung","nummer","assurance"]))

    # SWIFT/BIC — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="SWIFT_BIC",
            supported_language=lang,
            patterns=[Pattern("bic", r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b(?![a-z])", 0.85)],
            context=["bic","swift","bank","payment","zahlung"]))

    # Internal references — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="INTERNAL_REF",
            supported_language=lang,
            patterns=[Pattern("case_ref", r"\b(ZRH|CASE|DOC|REF|TICKET|SUB)-[\w\d-]+\b", 0.8)],
            context=["case","reference","document","submission","ticket"]))

    # PAN card — English only
    recognizers.append(PatternRecognizer(supported_entity="IN_PAN",
        supported_language="en",
        patterns=[Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", 0.85)],
        context=["pan","permanent account"]))

    return recognizers

# ─── ANALYZER ─────────────────────────────────────────────────────────────────

def build_analyzer():
    models = [{"lang_code": lang, "model_name": model} for lang, model in LANGUAGE_MODELS.items()]
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": models,
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    for r in _build_custom_recognizers():
        analyzer.registry.add_recognizer(r)
    return analyzer

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = build_analyzer()
    return _analyzer

# ─── LANGUAGE + DOCUMENT DETECTION ───────────────────────────────────────────

def auto_detect_language(text):
    try:
        lang = detect_language(text)
        return lang if lang in LANGUAGE_MODELS else DEFAULT_LANGUAGE
    except Exception:
        return DEFAULT_LANGUAGE

def auto_detect_document_type(text):
    keywords = {
        "medical": ["diagnosis","diagnose","patient","physician","hospital","krankenhaus",
                    "clinic","medical","doctor","arzt","treatment","condition","diagnosi"],
        "financial": ["salary","lohn","payroll","gross","net pay","employee","corporation",
                      "income","wages","salaire","stipendio"],
        "insurance": ["policy","police","premium","coverage","claim","insured","beneficiary",
                      "insurance","versicherung","assurance","iban","bic","settlement"],
        "tax": ["tax return","steuererklarung","1040","refund","irs","federal tax","taxable income"]
    }
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in keywords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"

# ─── POST PROCESSING ──────────────────────────────────────────────────────────

def _remove_false_positives(entities, text, document_type):
    email_spans = [(e["start"], e["end"]) for e in entities if e["entity_type"] == "EMAIL_ADDRESS"]

    # ── FIX 1: Pre-compute all currency-prefixed number spans ──────────────────
    # Numbers like $50,000 / CHF 1,200 / £300 must never be redacted.
    # We build a list of (start, end) spans that are "protected" from redaction.
    currency_spans = [
        (m.start(), m.end())
        for m in re.finditer(
            r'(?:[\$€£¥₹]|CHF|EUR|USD|GBP|INR)\s?[\d,]+(?:\.\d+)?(?:/\w+)?',
            text
        )
    ]

    filtered = []
    for entity in entities:
        t = entity["text"].strip()
        et = entity["entity_type"]

        # ── FIX 1: Never redact financial amounts ──────────────────────────────
        # FINANCIAL_AMOUNT entities (salaries, income, premiums) should be visible.
        if et == "FINANCIAL_AMOUNT":
            continue

        # ── FIX 1: Skip any entity whose span falls inside a currency expression ─
        # This prevents "$50,000" from having "50,000" redacted as DATE_TIME/NRP.
        if et in ("DATE_TIME", "NRP", "ID_NUMBER", "NUMBER", "LOCATION") and any(
            cs <= entity["start"] and entity["end"] <= ce
            for cs, ce in currency_spans
        ):
            continue

        # ── Never redact age values (any entity type) ─────────────────────────
        # Covers "Age: 34", "age: forty-two", "Age 28" etc.
        # Check up to 20 chars before entity for the standalone word "age".
        _pre = text[max(0, entity["start"] - 20):entity["start"]].lower()
        if re.search(r'\bage\s*[:\-]?\s*$', _pre):
            continue

        # ── Skip pure short numeric strings mis-tagged as DATE_TIME ───────────
        if et == "DATE_TIME" and re.fullmatch(r'\d{1,3}', t.strip()):
            continue

        # ── Existing filters ───────────────────────────────────────────────────
        if et == "SWIFT_BIC" and (t in SWIFT_FALSE_POSITIVES or not t.isupper()):
            continue
        if et == "PERSON" and t in PERSON_FALSE_POSITIVES:
            continue
        if et == "PERSON" and len(t.split()) == 1 and t.istitle() and len(t) < 6:
            continue
        if et == "IBAN_CODE" and not validate_iban(t):
            continue
        # Never redact organizations/company names
        if et == "ORGANIZATION":
            continue
        if et in ("PERSON", "LOCATION") and t in ORG_FALSE_POSITIVES:
            continue
        if et == "URL" and any(s <= entity["start"] and entity["end"] <= e for s, e in email_spans):
            continue
        if et == "LOCATION" and len(t) == 2 and t.isupper():
            continue
        # Filter "30 days", "7 days", "24 hours" etc — durations not dates
        if et == "DATE_TIME" and any(
            t.lower().endswith(unit)
            for unit in [" days", " day", " hours", " hour", " weeks", " week",
                         " months", " month", " years", " year"]
        ):
            continue
        # Filter short pure numbers detected as dates
        if et == "DATE_TIME" and t.strip().isdigit() and len(t.strip()) <= 2:
            continue

        filtered.append(entity)
    return filtered

def _deduplicate_entities(entities):
    span_map = {}
    for entity in entities:
        key = (entity["start"], entity["end"])
        if key not in span_map or entity["score"] > span_map[key]["score"]:
            span_map[key] = entity
    result = list(span_map.values())
    final = []
    for entity in result:
        contained = any(
            other["start"] <= entity["start"] and entity["end"] <= other["end"]
            and other["score"] >= entity["score"]
            and (other["start"], other["end"]) != (entity["start"], entity["end"])
            for other in result
        )
        if not contained:
            final.append(entity)
    return sorted(final, key=lambda x: x["start"])

# ─── GLOBAL CONSISTENCY ───────────────────────────────────────────────────────

def _apply_global_consistency(entities, text):
    """
    If a value (e.g. a person name, UUID, phone) was detected and confirmed as
    PII at one location in the document, find and tag ALL other occurrences of
    that exact string so redaction is uniform throughout the document.

    Only applies to entity types where repetition is meaningful:
    PERSON, PHONE_NUMBER, EMAIL_ADDRESS, ID_NUMBER, IBAN_CODE, CH_AHV,
    SWIFT_BIC, IN_PAN, US_SSN, INTERNAL_REF, COMPANY_NAME, LOCATION.

    Short or generic values (< 4 chars) are skipped to avoid false positives.
    """
    CONSISTENCY_TYPES = {
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "ID_NUMBER",
        "IBAN_CODE", "CH_AHV", "SWIFT_BIC", "IN_PAN", "US_SSN",
        "INTERNAL_REF", "COMPANY_NAME", "LOCATION", "MEDICAL_CONDITION",
    }

    # Build a map of confirmed values → entity_type
    confirmed = {}
    for e in entities:
        if e["entity_type"] in CONSISTENCY_TYPES and len(e["text"].strip()) >= 4:
            val = e["text"].strip()
            # Prefer higher-confidence entity type if same text appears twice
            if val not in confirmed or e["score"] > confirmed[val]["score"]:
                confirmed[val] = {"entity_type": e["entity_type"], "score": e["score"]}

    if not confirmed:
        return entities

    # Existing covered spans so we don't double-tag
    existing_spans = {(e["start"], e["end"]) for e in entities}
    extra = []

    for value, meta in confirmed.items():
        # Escape for regex; match whole-word boundaries where possible
        escaped = re.escape(value)
        pattern = r'(?<![A-Za-z0-9])' + escaped + r'(?![A-Za-z0-9])'
        for m in re.finditer(pattern, text):
            span = (m.start(), m.end())
            if span not in existing_spans:
                existing_spans.add(span)
                extra.append({
                    "entity_type": meta["entity_type"],
                    "text": value,
                    "start": m.start(),
                    "end": m.end(),
                    "score": round(meta["score"] * 0.95, 3),  # slightly lower — inferred
                })

    if not extra:
        return entities

    combined = entities + extra
    return sorted(combined, key=lambda x: x["start"])


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def detect_pii(text, language="auto", document_type="auto"):
    if not text or not text.strip():
        return []
    if language == "auto":
        language = auto_detect_language(text)
    if document_type == "auto":
        document_type = auto_detect_document_type(text)
    analyzer = get_analyzer()

    # Run NER line by line to prevent entities spanning across newlines
    all_entities = []
    offset = 0
    for line in text.split("\n"):
        if line.strip():
            raw = analyzer.analyze(text=line, language=language, score_threshold=CONFIDENCE_THRESHOLD)
            for r in raw:
                all_entities.append({
                    "entity_type": r.entity_type,
                    "text": line[r.start:r.end],
                    "start": r.start + offset,
                    "end": r.end + offset,
                    "score": round(r.score, 3)
                })

            # Also run context-based number detection on this line
            context_numbers = _extract_context_numbers(line)
            existing_spans = {(e["start"] - offset, e["end"] - offset) for e in all_entities if e["start"] >= offset}
            for (start, end, value) in context_numbers:
                if (start, end) not in existing_spans:
                    all_entities.append({
                        "entity_type": "ID_NUMBER",
                        "text": value,
                        "start": start + offset,
                        "end": end + offset,
                        "score": 0.8
                    })

        offset += len(line) + 1  # +1 for newline

    # ── FIX 2: Extract full addresses via label-based regex ───────────────────
    # spaCy often only captures fragments of addresses (e.g., just the city or
    # just the street name). This step finds the full address string after a
    # label like "Address:", "Home Address:", "Adresse:" etc. and tags the
    # entire value as LOCATION so it gets fully redacted.
    all_entities += _extract_address_entities(text, all_entities)

    # ── Extract names from label patterns missed by spaCy ─────────────────────
    # e.g. "Insured Name: Amanda Coleman", "Taxpayer Name: Tyler Moore"
    name_label_re = re.compile(
        r"(?i)(?:insured|taxpayer|signer|patient|employee|member|"
        r"policy.?holder|claimant|beneficiary|account.?holder|"
        r"authorized|contact|primary|secondary|full)\s+name\s*[:\-]\s*"
        r"([A-Z][a-z]+(?:[ ][A-Z][a-z]+)+)"
    )
    for m in name_label_re.finditer(text):
        start = m.start(1)
        end = m.end(1)
        already = any(e["start"] <= start and end <= e["end"] for e in all_entities)
        if not already:
            all_entities.append({
                "entity_type": "PERSON",
                "text": m.group(1),
                "start": start,
                "end": end,
                "score": 0.9
            })

    # ── FIX 2: Extract company/insurer names after a Company/Insurer label ─────
    # "Company: Hughes Group", "Insurer: Acme Ltd", "Employer: Globex Corp" etc.
    # These are tagged as COMPANY_NAME so they get redacted.
    company_label_re = re.compile(
        r"(?i)(?:company|insurer|insured\s+by|employer|firm|organisation|organization|"
        r"carrier|underwriter|provider|issuer|issued\s+by)\s*[:\-]\s*"
        r"([A-Z][A-Za-z0-9&',\.\s\-]{1,80}?)(?=\n|$|,|\|)"
    )
    for m in company_label_re.finditer(text):
        start = m.start(1)
        end = m.end(1)
        value = m.group(1).strip()
        if not value:
            continue
        already = any(e["start"] <= start and end <= e["end"] for e in all_entities)
        if not already:
            all_entities.append({
                "entity_type": "COMPANY_NAME",
                "text": value,
                "start": start,
                "end": end,
                "score": 0.9
            })

    all_entities = _remove_false_positives(all_entities, text, document_type)
    all_entities = _deduplicate_entities(all_entities)

    # ── FIX 3: Global consistency — if a value was redacted anywhere, redact it
    # everywhere in the document. This prevents the same name/ID from being
    # redacted on line 3 but left visible on line 12.
    all_entities = _apply_global_consistency(all_entities, text)

    return all_entities

def get_pii_summary(entities):
    summary = {}
    for e in entities:
        summary[e["entity_type"]] = summary.get(e["entity_type"], 0) + 1
    return summary
