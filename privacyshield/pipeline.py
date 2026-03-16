import logging
from pathlib import Path

from privacyshield.analyzer.pdf_analyzer import analyze_pdf, PageType
from privacyshield.text_pipeline.extractor import extract_text_pages, get_merged_bbox_for_span
from privacyshield.text_pipeline.ner_engine import detect_pii, auto_detect_language, auto_detect_document_type
from privacyshield.text_pipeline.redactor import redact_text, get_redaction_stats
from privacyshield.image_pipeline.pdf_to_image import pdf_page_to_image
from privacyshield.image_pipeline.ocr_engine import extract_text_with_coords
from privacyshield.image_pipeline.image_redactor import redact_regions

import fitz

logger = logging.getLogger(__name__)


def _get_redaction_boxes_fitz(pdf_path, page_num, entities, page_height):
    """Use PyMuPDF text search to find exact bbox for each entity."""
    boxes = []
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        for entity in entities:
            search_text = entity["text"]
            if entity["entity_type"] == "MEDICAL_CONDITION" and ":" in search_text:
                search_text = search_text.split(":", 1)[1].strip()
            instances = page.search_for(search_text)
            for rect in instances:
                boxes.append({
                    "entity_type": entity["entity_type"],
                    "text": entity["text"],
                    "bbox": {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
                    "page_number": page_num,
                    "page_height": page_height,
                })
                break
        doc.close()
    except Exception as e:
        logger.error(f"fitz bbox search failed: {e}")
    return boxes


def _run_image_pipeline_on_page(pdf_path, page_num, token_map=None):
    """
    Run OCR + NER + redaction on a single page image.
    Returns (redacted_image, pii_regions) or (None, []) on failure.
    token_map is passed to image_redactor to draw token labels on boxes.
    """
    try:
        image = pdf_page_to_image(pdf_path, page_num=page_num - 1)
        regions = extract_text_with_coords(image)

        if not regions:
            return None, []

        # Build full page text for context-aware detection
        full_text = "\n".join(r["text"] for r in regions)
        language = auto_detect_language(full_text)
        document_type = auto_detect_document_type(full_text)

        # Run NER on each region — also build token map for new PII found
        pii_regions = []
        local_token_map = dict(token_map) if token_map else {}

        for region in regions:
            entities = detect_pii(
                region["text"],
                language=language,
                document_type=document_type,
            )
            if entities:
                # Add any new tokens found in image to token map
                from privacyshield.text_pipeline.redactor import redact_text as _redact
                _, local_token_map = _redact(
                    region["text"],
                    entities,
                    existing_token_map=local_token_map,
                )
                pii_regions.append(region)

        # Pass token map so image_redactor can draw labels
        redacted_image = redact_regions(image, pii_regions, token_map=local_token_map)
        logger.info(
            f"Page {page_num}: image pipeline redacted "
            f"{len(pii_regions)}/{len(regions)} regions"
        )
        return redacted_image, pii_regions

    except Exception as e:
        logger.error(f"Page {page_num}: image pipeline failed — {e}")
        return None, []


def run_text_pipeline(pdf_path: str) -> dict:
    """
    Run the full redaction pipeline on a PDF.
    Handles text, scanned, and mixed pages.
    """
    pdf_path = str(pdf_path)
    logger.info(f"Starting pipeline: {pdf_path}")

    # ── Step 1: Analyze PDF ────────────────────────────────────────────────────
    analysis = analyze_pdf(pdf_path)
    logger.info(analysis.summary())

    # ── Step 2: Extract text from text + mixed pages ───────────────────────────
    pages_to_extract = analysis.text_pages + analysis.mixed_pages
    if not pages_to_extract:
        logger.warning("No text pages found — may be fully scanned PDF")
        extraction_pages = []
    else:
        extraction = extract_text_pages(pdf_path, page_numbers=pages_to_extract)
        extraction_pages = extraction.pages

    # ── Step 3 + 4: NER + Redaction on text/mixed pages ───────────────────────
    global_token_map = {}
    redacted_pages = []

    for page_ext in extraction_pages:
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
                "redacted_image": None,
                "pii_region_count": 0,
            })
            continue

        # Detect PII in text layer
        try:
            entities = detect_pii(text)
            logger.info(f"Page {page_num}: {len(entities)} PII entities found in text layer")
        except Exception as e:
            logger.error(f"Page {page_num}: NER failed — {e}")
            entities = []

        # Get bounding boxes for text layer PII
        redaction_boxes = _get_redaction_boxes_fitz(
            pdf_path, page_num, entities, page_ext.height
        )

        # Redact text — updates global_token_map
        try:
            redacted_text, global_token_map = redact_text(
                text, entities, existing_token_map=global_token_map
            )
        except Exception as e:
            logger.error(f"Page {page_num}: redaction failed — {e}")
            redacted_text = text

        page_info = next(
            (p for p in analysis.pages if p.page_number == page_num), None
        )
        page_type = page_info.page_type.value if page_info else "text"

        page_result = {
            "page_number": page_num,
            "page_type": page_type,
            "original_text": text,
            "redacted_text": redacted_text,
            "entities": entities,
            "redaction_boxes": redaction_boxes,
            "redacted_image": None,
            "pii_region_count": 0,
        }

        # ── Mixed pages: ALSO run image pipeline for embedded scanned content ──
        # Pass global_token_map so image labels stay consistent with text labels.
        # Image overlay goes FIRST in pdf_merger, then text boxes on top.
        if page_type == "mixed":
            logger.info(f"Page {page_num}: mixed — running image pipeline for embedded content")
            redacted_image, pii_regions = _run_image_pipeline_on_page(
                pdf_path, page_num, token_map=global_token_map
            )
            page_result["redacted_image"] = redacted_image
            page_result["pii_region_count"] = len(pii_regions)

        redacted_pages.append(page_result)

    # ── Step 5: Run image pipeline on scanned pages ────────────────────────────
    for page_num in analysis.scanned_pages:
        logger.info(f"Page {page_num}: scanned — running image pipeline")
        redacted_image, pii_regions = _run_image_pipeline_on_page(
            pdf_path, page_num, token_map=global_token_map
        )

        redacted_pages.append({
            "page_number": page_num,
            "page_type": "scanned",
            "original_text": "",
            "redacted_text": "",
            "entities": [],
            "redaction_boxes": [],
            "redacted_image": redacted_image,
            "pii_region_count": len(pii_regions),
        })

    # Sort pages by page number
    redacted_pages.sort(key=lambda x: x["page_number"])

    # ── Step 6: Compile result ─────────────────────────────────────────────────
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
        print(f"  Image PII regions: {page.get('pii_region_count', 0)}")
        if page["entities"]:
            for e in page["entities"]:
                print(f"    [{e['entity_type']}] \"{e['text']}\"")
        if page["redacted_text"]:
            preview = page["redacted_text"][:200].replace("\n", " ")
            print(f"  Redacted text preview: {preview}...")
        print()

    print(f"Token map ({len(result['token_map'])} entries):")
    for token, value in result["token_map"].items():
        print(f"  {token} → {value}")