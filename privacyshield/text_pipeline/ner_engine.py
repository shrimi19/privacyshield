from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from langdetect import detect as detect_language

CONFIDENCE_THRESHOLD = 0.4
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

SWIFT_FALSE_POSITIVES = {"Rechnung","Paziente","Paciente","Telefono","Telefon","Patient","Diagnose","Betrag"}

ORG_FALSE_POSITIVES = {
    "SSN","DOB","EIN","NPI","ID","DOJ",
    "IL","MA","CA","NY","TX",
    "AHV-Nummer","AHV-Nr","AHV","AVS","Tel","Tél","Adresse","Diagnose"
}

LANGUAGE_MODELS = {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "it": "it_core_news_lg",
}
DEFAULT_LANGUAGE = "en"
ALL_LANGUAGES = list(LANGUAGE_MODELS.keys())
_analyzer = None

def _build_custom_recognizers():
    recognizers = []

    # SSN — English only
    recognizers.append(PatternRecognizer(supported_entity="US_SSN",
        supported_language="en",
        patterns=[Pattern("SSN_dashes", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
                  Pattern("SSN_plain",  r"\b\d{9}\b", 0.5)],
        context=["ssn","social","security"]))

    # Financial amounts — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="FINANCIAL_AMOUNT",
            supported_language=lang,
            patterns=[
                Pattern("usd",   r"\$\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("chf",   r"CHF\s?[\d,]+(?:\.\d{2})?", 0.8),
                Pattern("eur",   r"EUR\s?[\d,]+(?:\.\d{2})?", 0.7),
                Pattern("usd_m", r"\$\s?[\d,]+/month", 0.8),
            ],
            context=["salary","income","wage","pay","benefit","amount","due",
                     "premium","gross","net","total","betrag","salaire","stipendio"]))

    # Phone numbers — all languages (handles +41, +33, +39, +34 formats)
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="PHONE_NUMBER",
            supported_language=lang,
            patterns=[
                Pattern("intl_phone", r"\+\d{1,3}[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}", 0.85),
                Pattern("us_phone",   r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", 0.75),
                Pattern("fr_phone", r"\+33[\s.]?\d[\s.]?(?:\d{2}[\s.]?){4}", 0.85),
            ],
            context=["tel","tél","phone","telefon","telefono","teléfono","mobile","handy"]))

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

    # ID numbers — all languages
    for lang in ALL_LANGUAGES:
        recognizers.append(PatternRecognizer(supported_entity="ID_NUMBER",
            supported_language=lang,
            patterns=[Pattern("prefixed_id",
                r"\b(EMP|PAT|POL|MED|INV|CLM|REF|ZUR|PH|MRN|HB)-[\w\d][\w\d-]*\b", 0.85)],
            context=["employee","patient","policy","member","account","id","number","claim"]))

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

    # AHV/AVS Swiss number — German + French
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

# Words incorrectly matched as SWIFT/BIC codes
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

SWIFT_FALSE_POSITIVES = {
    "Rechnung","Paziente","Paciente","Telefono","Telefon",
    "Patient","Diagnose","Betrag","Krankenhaus"
}

def _remove_false_positives(entities, text, document_type):
    email_spans = [(e["start"], e["end"]) for e in entities if e["entity_type"] == "EMAIL_ADDRESS"]
    filtered = []
    for entity in entities:
        t = entity["text"].strip()
        et = entity["entity_type"]
        if et == "SWIFT_BIC" and (t in SWIFT_FALSE_POSITIVES or not t.isupper()):
            continue
        if et == "IBAN_CODE" and not validate_iban(t):
            continue
        if et == "ORGANIZATION" and t.upper() in {fp.upper() for fp in ORG_FALSE_POSITIVES}:
            continue
        if et in ("ORGANIZATION","PERSON","LOCATION") and t in ORG_FALSE_POSITIVES:
            continue
        if et == "URL" and any(s <= entity["start"] and entity["end"] <= e for s, e in email_spans):
            continue
        if et == "LOCATION" and len(t) == 2 and t.isupper():
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

def detect_pii(text, language="auto", document_type="auto"):
    if not text or not text.strip():
        return []
    if language == "auto":
        language = auto_detect_language(text)
    if document_type == "auto":
        document_type = auto_detect_document_type(text)
    analyzer = get_analyzer()
    raw = analyzer.analyze(text=text, language=language, score_threshold=CONFIDENCE_THRESHOLD)
    entities = [{"entity_type": r.entity_type, "text": text[r.start:r.end],
                 "start": r.start, "end": r.end, "score": round(r.score, 3)} for r in raw]
    entities = _remove_false_positives(entities, text, document_type)
    entities = _deduplicate_entities(entities)
    return entities

def get_pii_summary(entities):
    summary = {}
    for e in entities:
        summary[e["entity_type"]] = summary.get(e["entity_type"], 0) + 1
    return summary
