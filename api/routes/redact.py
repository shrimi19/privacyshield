"""
redact.py
---------
POST /redact — accepts PDF, returns redacted PDF + encryption key
"""

import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from privacyshield.reconstructor.pdf_merger import redact_pdf
from privacyshield.key_manager.encryptor import (
    generate_key, key_to_string, string_to_key, encrypt_bytes
)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

PAYLOAD_MARKER = b"\n%%PRIVACYSHIELD_ENCRYPTED_ORIGINAL_V1%%\n"


def _embed_encrypted_original(redacted_pdf_path: Path, original_pdf_path: Path, key: bytes) -> None:
    original_bytes = original_pdf_path.read_bytes()
    encrypted_payload = encrypt_bytes(original_bytes, key)
    with redacted_pdf_path.open("ab") as f:
        f.write(PAYLOAD_MARKER)
        f.write(encrypted_payload)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


@router.post("/redact")
async def redact_document(file: UploadFile = File(...)):
    """
    Accept a PDF, redact PII, return redacted PDF + encryption key.

    Returns JSON with job_id and encryption_key.
    Use GET /download/{job_id} to download the redacted PDF.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = uuid.uuid4().hex
    original_name = file.filename
    safe_stem = Path(original_name).stem

    upload_path = UPLOAD_DIR / f"{job_id}.pdf"
    output_path = OUTPUT_DIR / f"{job_id}_redacted.pdf"
    shield_path = OUTPUT_DIR / f"{job_id}.privacyshield"

    # Save uploaded file
    content = await file.read()
    upload_path.write_bytes(content)

    # Run redaction pipeline
    try:
        key_string = redact_pdf(
            input_pdf_path=str(upload_path),
            output_pdf_path=str(output_path),
            shield_path=str(shield_path),
        )
    except Exception as e:
        _safe_unlink(upload_path)
        _safe_unlink(output_path)
        _safe_unlink(shield_path)
        raise HTTPException(status_code=500, detail=f"Redaction pipeline failed: {e}")

    # Generate key if no PII found
    if key_string:
        key = string_to_key(key_string)
    else:
        key = generate_key()
        key_string = key_to_string(key)

    # Embed encrypted original into redacted PDF
    _embed_encrypted_original(output_path, upload_path, key)

    return {
        "job_id": job_id,
        "original_name": original_name,
        "download_name": f"{safe_stem}_redacted.pdf",
        "encryption_key": key_string,
    }


@router.get("/download/{job_id}")
def download_redacted(job_id: str, name: str = "redacted.pdf"):
    """Download the redacted PDF by job_id."""
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job id.")

    output_path = OUTPUT_DIR / f"{job_id}_redacted.pdf"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found. It may have expired.")

    return FileResponse(
        path=str(output_path),
        filename=name,
        media_type="application/pdf",
    )


@router.get("/preview/{job_id}/{variant}")
def preview(job_id: str, variant: str):
    """Preview original or redacted PDF inline."""
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job id.")

    if variant == "original":
        pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    elif variant == "redacted":
        pdf_path = OUTPUT_DIR / f"{job_id}_redacted.pdf"
    else:
        raise HTTPException(status_code=400, detail="Invalid variant.")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Preview file not found.")

    return FileResponse(path=str(pdf_path), media_type="application/pdf")