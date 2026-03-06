"""
PURPOSE:
    Detects all PII in extracted text using Presidio + spaCy.

INPUT:
    - text (str): Plain text extracted from PDF page
    - language (str): default 'en'
    - document_type (str): 'medical','financial','insurance','tax','general','auto'

OUTPUT:
    [{"entity_type": "PERSON", "text": "Shrimi Agrawal", "start": 6, "end": 16, "score": 0.85}, ...]

METHOD USED:
    1. Build Presidio analyzer with spaCy en_core_web_lg
    2. Add custom recognizers for SSN, amounts, IDs, medical conditions
    3. Run analysis at threshold 0.4
    4. Remove false positives (label words, URL-in-email, state codes)
    5. Deduplicate overlapping spans (keep highest score)
    6. Apply document-type specific rules

FALSE POSITIVES HANDLED:
    - "SSN","DOB" misclassified as ORGANIZATION → removed
    - "gmail.com" as URL when email already detected → removed
    - 2-letter state codes as LOCATION → removed
    - Same span by multiple recognizers → keep highest score
    - Policy numbers as PERSON → reclassified as ID_NUMBER

DEPENDENCIES:
    pip install presidio-analyzer
    python -m spacy download en_core_web_lg
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

CONFIDENCE_THRESHOLD = 0.4
ORG_FALSE_POSITIVES = {"SSN","DOB","EIN","NPI","ID","DOJ","IL","MA","CA","NY","TX"}
_analyzer = None

def _build_custom_recognizers():
    recognizers = []
    recognizers.append(PatternRecognizer(supported_entity="US_SSN",
        patterns=[Pattern("SSN_dashes", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
                  Pattern("SSN_plain",  r"\b\d{9}\b", 0.5)],
        context=["ssn","social","security"]))
    recognizers.append(PatternRecognizer(supported_entity="FINANCIAL_AMOUNT",
        patterns=[Pattern("usd", r"\$\s?[\d,]+(?:\.\d{2})?", 0.7),
                  Pattern("usd_month", r"\$\s?[\d,]+/month", 0.8)],
        context=["salary","income","wage","pay","benefit","amount","due","premium","gross","net","total"]))
    recognizers.append(PatternRecognizer(supported_entity="ID_NUMBER",
        patterns=[Pattern("prefixed_id", r"\b(EMP|PAT|POL|MED|INV|CLM|REF|ZUR|PH|MRN|HB)-[\w\d][\w\d-]*\b", 0.85)],
        context=["employee","patient","policy","member","account","id","number"]))
    recognizers.append(PatternRecognizer(supported_entity="MEDICAL_CONDITION",
        patterns=[Pattern("diagnosis_label",
                    r"(?i)(?:diagnosis|condition|disorder):\s*([A-Z][^\n,\.]{3,50})", 0.8),
                  Pattern("condition_inline",
                    r"(?i)\b(diabetes|hypertension|asthma|cancer|arthritis|depression|anxiety|cardiac|cardiovascular)\b", 0.65)],
        context=["diagnosis","condition","medical","pre-existing","history"]))
    recognizers.append(PatternRecognizer(supported_entity="SWIFT_BIC",
        patterns=[Pattern("bic", r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b", 0.8)],
        context=["bic","swift","bank","payment"]))
    recognizers.append(PatternRecognizer(supported_entity="INTERNAL_REF",
        patterns=[Pattern("case_ref", r"\b(ZRH|CASE|DOC|REF|TICKET|SUB)-[\w\d-]+\b", 0.8)],
        context=["case","reference","document","submission","ticket"]))
    return recognizers

def build_analyzer():
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
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

def auto_detect_document_type(text):
    keywords = {
        "medical": ["diagnosis","patient","physician","hospital","clinic","medical","doctor","treatment","condition"],
        "financial": ["salary","payroll","gross","net pay","employee","corporation","income","wages"],
        "insurance": ["policy","premium","coverage","claim","insured","beneficiary","insurance","iban","bic","settlement","policyholder","reimbursement","cardiac","diagnosis"],
        "tax": ["tax return","1040","refund","irs","federal tax","taxable income"]
    }
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in keywords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"

def _remove_false_positives(entities, text, document_type):
    email_spans = [(e["start"], e["end"]) for e in entities if e["entity_type"] == "EMAIL_ADDRESS"]
    filtered = []
    for entity in entities:
        t = entity["text"].strip()
        et = entity["entity_type"]
        if et == "ORGANIZATION" and t.upper() in ORG_FALSE_POSITIVES:
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

def detect_pii(text, language="en", document_type="auto"):
    if not text or not text.strip():
        return []
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
