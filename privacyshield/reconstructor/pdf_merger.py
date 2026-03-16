"""
pdf_merger.py
=============
PURPOSE:
    Takes the pipeline result and produces the final redacted PDF.

    For text pages  → draws black boxes over PII text with token labels
    For scanned pages → overlays the redacted PIL image over full page
    For mixed pages  → image overlay FIRST, then text redaction boxes on top

    Also handles the full redact/unredact flow:
        redact_pdf()   → produces redacted PDF + .privacyshield file + key
        unredact_pdf() → uses key to restore original PDF from redacted PDF
"""

from __future__ import annotations
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install pymupdf")


def _embed_image_on_page(page: "fitz.Page", pil_image) -> None:
    """
    Embed a PIL image as a full-page overlay on a PyMuPDF page.
    Used for scanned/mixed pages where we replace content with
    the redacted image.
    """
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    page_rect = page.rect
    page.insert_image(page_rect, stream=img_bytes.read(), overlay=True)


def merge_redacted_pdf(
    original_pdf_path: str,
    pipeline_result: dict,
    output_path: str,
) -> str:
    """
    Produce the final redacted PDF from pipeline results.

    Order of operations per page:
    1. Image overlay FIRST (covers embedded scanned content)
    2. Text redaction boxes ON TOP (covers text layer PII with token labels)

    This ensures:
    - Scanned image PII is hidden by the redacted image
    - Text layer PII shows [TOKEN_ID] labels on black boxes
    - For mixed pages, both layers are handled correctly
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    original_pdf_path = str(original_pdf_path)
    output_path = str(output_path)

    if not Path(original_pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {original_pdf_path}")

    # Build reverse map: original_value → token_id
    token_map = pipeline_result.get("token_map", {})
    value_to_token = {v: k for k, v in token_map.items()}

    doc = fitz.open(original_pdf_path)
    total_text_boxes = 0
    total_image_pages = 0

    for page_result in pipeline_result["pages"]:
        page_num       = page_result["page_number"]
        page_type      = page_result["page_type"]
        redact_boxes   = page_result.get("redaction_boxes", [])
        redacted_image = page_result.get("redacted_image")

        page = doc[page_num - 1]

        # ── STEP 1: Image overlay FIRST (scanned + mixed pages) ───────────────
        # This covers the embedded scanned image content with redacted version.
        # Must happen before text redactions so text boxes appear on top.
        if redacted_image is not None:
            _embed_image_on_page(page, redacted_image)
            total_image_pages += 1
            logger.debug(f"Page {page_num}: overlaid redacted image")

        # ── STEP 2: Text redactions ON TOP (text + mixed pages) ───────────────
        # These black boxes cover the text layer PII with [TOKEN_ID] labels.
        # Drawn after image overlay so they appear on top.
        if redact_boxes:
            rects_and_labels = []

            for box in redact_boxes:
                bbox          = box["bbox"]
                original_text = box.get("text", "")
                entity_type   = box.get("entity_type", "")
                padding       = 1.5

                rect = fitz.Rect(
                    bbox["x0"] - padding,
                    bbox["y0"] - padding,
                    bbox["x1"] + padding,
                    bbox["y1"] + padding,
                )

                # Find token label — use token map if available
                if original_text in value_to_token:
                    label = f"[{value_to_token[original_text]}]"
                else:
                    label = f"[{entity_type}]"

                rects_and_labels.append((rect, label))
                page.add_redact_annot(rect, fill=(0, 0, 0))
                total_text_boxes += 1

            # Apply all redactions (permanently removes underlying text)
            page.apply_redactions()

            # Draw white token labels on top of black boxes
            for rect, label in rects_and_labels:
                try:
                    page.insert_textbox(
                        rect,
                        label,
                        fontsize=6,
                        color=(1, 1, 1),  # white text
                        align=1,          # center aligned
                    )
                except Exception as e:
                    logger.debug(f"Could not insert label '{label}': {e}")

            logger.debug(f"Page {page_num}: applied {len(redact_boxes)} text redactions")

    logger.info(
        f"Merged: {total_text_boxes} text boxes, "
        f"{total_image_pages} image overlays"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info(f"Saved redacted PDF to: {output_path}")
    return output_path


def redact_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    shield_path: str,
) -> str:
    """
    Full redact flow — single entry point.

    1. Run the full pipeline on the input PDF
    2. Merge results into redacted PDF with token labels
    3. Encrypt token map → save .privacyshield file
    4. Return the encryption key (show to user!)
    """
    from privacyshield.pipeline import run_text_pipeline
    from privacyshield.key_manager.encryptor import encrypt_token_map, key_to_string

    logger.info(f"Running pipeline on: {input_pdf_path}")
    result = run_text_pipeline(input_pdf_path)

    merge_redacted_pdf(input_pdf_path, result, output_pdf_path)

    if result["token_map"]:
        key = encrypt_token_map(result["token_map"], shield_path)
        key_string = key_to_string(key)
        logger.info(f"Saved .privacyshield to: {shield_path}")
    else:
        key_string = None
        logger.warning("No PII found — no .privacyshield file created")

    return key_string


def unredact_pdf(
    redacted_pdf_path: str,
    shield_path: str,
    key_string: str,
    output_pdf_path: str,
) -> str:
    """
    Full unredact flow — restore original PDF from redacted version.

    Note: Image-redacted pages (scanned/mixed) cannot be fully restored
    because pixel content is permanently overwritten.
    Only text-layer redactions are reversible via the token map.
    The app.py restore flow uses encrypted original bytes instead,
    which gives perfect restoration.
    """
    from privacyshield.key_manager.decryptor import decrypt_token_map

    # Step 1: Decrypt token map
    token_map = decrypt_token_map(shield_path, key_string)
    logger.info(f"Decrypted {len(token_map)} tokens")

    # Step 2: Open redacted PDF and restore text layer
    doc = fitz.open(redacted_pdf_path)

    for page in doc:
        text = page.get_text()
        if not text.strip():
            continue

        has_tokens = any(f"[{token}]" in text for token in token_map)
        if not has_tokens:
            continue

        # First pass: white out all token placeholders
        for token_id in token_map:
            placeholder = f"[{token_id}]"
            instances = page.search_for(placeholder)
            for rect in instances:
                page.add_redact_annot(rect, fill=(1, 1, 1))

        page.apply_redactions()

        # Second pass: re-insert original values
        for token_id, original_value in token_map.items():
            placeholder = f"[{token_id}]"
            instances = page.search_for(placeholder)
            for rect in instances:
                page.insert_text(
                    rect.tl,
                    original_value,
                    fontsize=10,
                    color=(0, 0, 0),
                )

    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_pdf_path)
    doc.close()

    logger.info(f"Saved restored PDF to: {output_pdf_path}")
    return output_pdf_path