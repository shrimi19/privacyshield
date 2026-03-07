"""
pipeline.py
===========
PURPOSE:
    Orchestrates the full redaction pipeline for a PDF file.
    Connects analyzer → extractor → ner_engine → redactor.

    Text pipeline only for now.
    Image pipeline (scanned/mixed pages) plugs in later.

FLOW:
    1. analyze_pdf()         → classify each page (text/scanned/mixed)
    2. extract_text_pages()  → get text + char coordinates for text pages
    3. detect_pii()          → find PII entities in each page's text
    4. redact_text()         → replace PII with [TOKEN_ID] placeholders
    5. Save token map        → passed to key_manager for encryption
    6. Return redacted pages + token map

OUTPUT:
    {
        "pdf_path": str,
        "pages": [
            {
                "page_number": int,
                "page_type": str,
                "original_text": str,
                "redacted_text": str,
                "entities": [...],       ← raw NER output
                "redaction_boxes": [...] ← BoundingBox per PII span
            }
        ],
        "token_map": {"NAME_1": "John Smith", ...},
        "stats": {"NAME": 2, "SSN": 1, "TOTAL": 3}
    }
"""

import logging
from pathlib import Path

from privacyshield.analyzer.pdf_analyzer import analyze_pdf, PageType
from privacyshield.text_pipeline.extractor import (
    extract_text_pages,
    get_merged_bbox_for_span,
)
from privacyshield.text_pipeline.ner_engine import detect_pii
from privacyshield.text_pipeline.redactor import redact_text, get_redaction_stats

logger = logging.getLogger(__name__)


def run_text_pipeline(pdf_path: str) -> dict:
    """
    Run the full text redaction pipeline on a PDF.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        Dict with redacted pages, token map, and stats.
    """
    pdf_path = str(pdf_path)
    logger.info(f"Starting pipeline: {pdf_path}")

    # ── Step 1: Analyze PDF ───────────────────────────────────────────────────
    analysis = analyze_pdf(pdf_path)
    logger.info(analysis.summary())

    # ── Step 2: Extract text from text + mixed pages ──────────────────────────
    pages_to_extract = analysis.text_pages + analysis.mixed_pages
    if not pages_to_extract:
        logger.warning("No text pages found — may be fully scanned PDF")

    extraction = extract_text_pages(pdf_path, page_numbers=pages_to_extract)

    # ── Step 3 + 4: NER + Redaction across all pages ──────────────────────────
    # IMPORTANT: single global token map so same value = same token everywhere
    global_token_map = {}
    redacted_pages = []

    for page_ext in extraction.pages:
        page_num = page_ext.page_number
        text = page_ext.full_text

        if not text.strip():
            logger.debug(f"Page {page_num}: empty text, skipping")
            redacted_pages.append({
                "page_number": page_num,
                "page_type": "text",
                "original_text": "",
                "redacted_text": "",
                "entities": [],
                "redaction_boxes": [],
            })
            continue

        # Detect PII
        try:
            entities = detect_pii(text)
            logger.info(f"Page {page_num}: {len(entities)} PII entities found")
        except Exception as e:
            logger.error(f"Page {page_num}: NER failed — {e}")
            entities = []

        # Get bounding boxes for each PII span
        redaction_boxes = []
        for entity in entities:
            bbox = get_merged_bbox_for_span(
                page_ext,
                entity["start"],
                entity["end"] - 1  # end is exclusive in NER, inclusive here
            )
            if bbox:
                redaction_boxes.append({
                    "entity_type": entity["entity_type"],
                    "text": entity["text"],
                    "bbox": bbox.to_dict(),
                    "page_number": page_num,
                    "page_height": page_ext.height,
                })

        # Redact text — pass global map so tokens stay consistent
        try:
            redacted_text, global_token_map = redact_text(
                text, entities, existing_token_map=global_token_map
            )
        except Exception as e:
            logger.error(f"Page {page_num}: redaction failed — {e}")
            redacted_text = text  # fallback: unredacted

        # Get page type from analysis
        page_info = next(
            (p for p in analysis.pages if p.page_number == page_num), None
        )
        page_type = page_info.page_type.value if page_info else "text"

        redacted_pages.append({
            "page_number": page_num,
            "page_type": page_type,
            "original_text": text,
            "redacted_text": redacted_text,
            "entities": entities,
            "redaction_boxes": redaction_boxes,
        })

    # ── Step 5: Add scanned pages as placeholders (image pipeline later) ──────
    for page_num in analysis.scanned_pages:
        logger.info(f"Page {page_num}: scanned — image pipeline needed (TODO)")
        redacted_pages.append({
            "page_number": page_num,
            "page_type": "scanned",
            "original_text": "",
            "redacted_text": "",
            "entities": [],
            "redaction_boxes": [],
            "note": "scanned page — image pipeline not yet implemented"
        })

    # Sort pages by page number
    redacted_pages.sort(key=lambda x: x["page_number"])

    # ── Step 6: Compile result ────────────────────────────────────────────────
    result = {
        "pdf_path": pdf_path,
        "total_pages": analysis.total_pages,
        "pages": redacted_pages,
        "token_map": global_token_map,
        "stats": get_redaction_stats(global_token_map),
    }

    logger.info(f"Pipeline complete. Stats: {result['stats']}")
    return result


def print_pipeline_report(result: dict):
    """Pretty print pipeline results for debugging."""
    print(f"\n{'='*60}")
    print(f"PIPELINE REPORT: {result['pdf_path']}")
    print(f"{'='*60}")
    print(f"Total pages: {result['total_pages']}")
    print(f"Redaction stats: {result['stats']}")
    print()

    for page in result["pages"]:
        print(f"--- Page {page['page_number']} ({page['page_type']}) ---")
        if page.get("note"):
            print(f"  NOTE: {page['note']}")
            continue
        print(f"  Entities found: {len(page['entities'])}")
        print(f"  Redaction boxes: {len(page['redaction_boxes'])}")
        if page["entities"]:
            for e in page["entities"]:
                print(f"    [{e['entity_type']}] \"{e['text']}\"")
        print(f"  Redacted text preview:")
        preview = page["redacted_text"][:200].replace("\n", " ")
        print(f"    {preview}...")
        print()

    print(f"Token map ({len(result['token_map'])} entries):")
    for token, value in result["token_map"].items():
        print(f"  {token} → {value}")
