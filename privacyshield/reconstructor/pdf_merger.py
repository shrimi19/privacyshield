"""
pdf_merger.py
=============
PURPOSE:
    Takes the pipeline result and produces the final redacted PDF.

    For text pages  → draws black boxes over PII text using PyMuPDF
    For scanned pages → replaces the page with the redacted PIL image
    For mixed pages  → draws black boxes on text layer AND overlays
                       the redacted image on top

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
    # Convert PIL image to PNG bytes
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Cover the entire page with the redacted image
    page_rect = page.rect
    page.insert_image(page_rect, stream=img_bytes.read(), overlay=True)


def merge_redacted_pdf(
    original_pdf_path: str,
    pipeline_result: dict,
    output_path: str,
) -> str:
    """
    Produce the final redacted PDF from pipeline results.

    For each page:
    - text pages    → apply black box redactions on text layer
    - scanned pages → overlay the redacted PIL image over the full page
    - mixed pages   → apply text redactions + overlay redacted image

    Args:
        original_pdf_path: Path to original unredacted PDF.
        pipeline_result: Output from pipeline.run_text_pipeline().
        output_path: Where to save the final redacted PDF.

    Returns:
        output_path as string.
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    original_pdf_path = str(original_pdf_path)
    output_path = str(output_path)

    if not Path(original_pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {original_pdf_path}")

    doc = fitz.open(original_pdf_path)
    total_text_boxes = 0
    total_image_pages = 0

    for page_result in pipeline_result["pages"]:
        page_num      = page_result["page_number"]
        page_type     = page_result["page_type"]
        redact_boxes  = page_result.get("redaction_boxes", [])
        redacted_image = page_result.get("redacted_image")

        page = doc[page_num - 1]

        # ── Text redactions (text + mixed pages) ──────────────────────────────
        if redact_boxes:
            for box in redact_boxes:
                bbox = box["bbox"]
                padding = 1.5
                rect = fitz.Rect(
                    bbox["x0"] - padding,
                    bbox["y0"] - padding,
                    bbox["x1"] + padding,
                    bbox["y1"] + padding,
                )
                page.add_redact_annot(rect, fill=(0, 0, 0))
                total_text_boxes += 1

            page.apply_redactions()
            logger.debug(f"Page {page_num}: applied {len(redact_boxes)} text redactions")

        # ── Image overlay (scanned + mixed pages) ─────────────────────────────
        if redacted_image is not None:
            _embed_image_on_page(page, redacted_image)
            total_image_pages += 1
            logger.debug(f"Page {page_num}: overlaid redacted image")

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
    2. Merge results into redacted PDF
    3. Encrypt token map → save .privacyshield file
    4. Return the encryption key (show to user!)

    Args:
        input_pdf_path:  Path to original PDF.
        output_pdf_path: Where to save redacted PDF.
        shield_path:     Where to save .privacyshield file.

    Returns:
        key_string: Base64 encryption key — show this to the user!
    """
    from privacyshield.pipeline import run_text_pipeline
    from privacyshield.key_manager.encryptor import encrypt_token_map, key_to_string

    # Step 1: Run full pipeline
    logger.info(f"Running pipeline on: {input_pdf_path}")
    result = run_text_pipeline(input_pdf_path)

    # Step 2: Merge into redacted PDF
    merge_redacted_pdf(input_pdf_path, result, output_pdf_path)

    # Step 3: Encrypt token map
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

    1. Decrypt .privacyshield → get token map
    2. For each page, restore [TOKEN_ID] placeholders in text layer

    Note: Image-redacted pages (scanned/mixed) cannot be fully restored
    because the original pixel content is permanently overwritten.
    Only text-layer redactions are reversible.

    Args:
        redacted_pdf_path: Path to redacted PDF.
        shield_path:       Path to .privacyshield file.
        key_string:        Base64 key string from user.
        output_pdf_path:   Where to save restored PDF.

    Returns:
        output_pdf_path as string.
    """
    from privacyshield.key_manager.decryptor import decrypt_token_map
    from privacyshield.text_pipeline.redactor import restore_text

    # Step 1: Decrypt token map
    token_map = decrypt_token_map(shield_path, key_string)
    logger.info(f"Decrypted {len(token_map)} tokens")

    # Step 2: Open redacted PDF and restore text layer
    doc = fitz.open(redacted_pdf_path)

    for page in doc:
        # Extract current text
        text = page.get_text()
        if not text.strip():
            continue

        # Check if any tokens exist on this page
        has_tokens = any(f"[{token}]" in text for token in token_map)
        if not has_tokens:
            continue

        # Restore tokens by searching and replacing in PDF
        for token_id, original_value in token_map.items():
            placeholder = f"[{token_id}]"
            instances = page.search_for(placeholder)
            for rect in instances:
                # Black out the placeholder
                page.add_redact_annot(rect, fill=(1, 1, 1))  # white fill
        page.apply_redactions()

        # Re-insert original values
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