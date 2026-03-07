"""
pdf_analyzer.py
---------------
Analyzes a PDF and classifies each page as:
  - "text"   → has a real extractable text layer
  - "scanned" → is purely an image (no text layer), needs OCR
  - "mixed"  → has both text AND image elements on the same page

Why this matters:
  Text pages  → go to text_pipeline (fast, accurate coordinates)
  Scanned pages → go to image_pipeline (OCR, then image redaction)
  Mixed pages → both pipelines run; results are merged

Design decision:
  We use pdfplumber (built on pdfminer) as the primary inspector.
  pdfplumber gives us:
    - page.chars  → list of individual characters with coordinates
    - page.images → list of image objects embedded in the page

  A page is "text" if it has chars and no significant images.
  A page is "scanned" if it has images but effectively no chars.
  A page is "mixed" if it has both.

  We define "significant image" as covering > MIN_IMAGE_COVERAGE of
  the page area — this avoids false positives from tiny logos or
  decorative elements that appear even in text PDFs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import pdfplumber

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# A page image must cover at least this fraction of page area to count
# as a "real" scan (vs. a small logo on a text page).
MIN_IMAGE_COVERAGE = 0.30  # 30 %

# A page must have at least this many characters to count as having a text layer.
# This avoids treating a page with one stray character as "text".
MIN_CHAR_COUNT = 10


# ── Data types ─────────────────────────────────────────────────────────────────

class PageType(str, Enum):
    """Classification for a single PDF page."""
    TEXT    = "text"
    SCANNED = "scanned"
    MIXED   = "mixed"


@dataclass
class PageInfo:
    """
    Everything the rest of the pipeline needs to know about one page.

    Attributes
    ----------
    page_number : int
        1-based page index (matches how humans count pages).
    page_type : PageType
        Classification: text / scanned / mixed.
    width : float
        Page width in PDF points (1 pt = 1/72 inch).
    height : float
        Page height in PDF points.
    char_count : int
        Number of extractable characters found on the page.
    image_coverage : float
        Fraction of page area covered by embedded images (0.0 – 1.0).
    """
    page_number:    int
    page_type:      PageType
    width:          float
    height:         float
    char_count:     int       = 0
    image_coverage: float     = 0.0


@dataclass
class PDFAnalysisResult:
    """
    The full analysis of one PDF file.

    Attributes
    ----------
    pdf_path : str
        Path to the file that was analyzed.
    total_pages : int
        How many pages the PDF has.
    pages : List[PageInfo]
        One PageInfo per page.
    text_pages : List[int]
        1-based page numbers classified as TEXT.
    scanned_pages : List[int]
        1-based page numbers classified as SCANNED.
    mixed_pages : List[int]
        1-based page numbers classified as MIXED.
    """
    pdf_path:      str
    total_pages:   int
    pages:         List[PageInfo] = field(default_factory=list)

    @property
    def text_pages(self) -> List[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.TEXT]

    @property
    def scanned_pages(self) -> List[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.SCANNED]

    @property
    def mixed_pages(self) -> List[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.MIXED]

    def summary(self) -> str:
        """Human-readable summary — useful for logging and debugging."""
        return (
            f"PDF: {self.pdf_path}\n"
            f"  Total pages : {self.total_pages}\n"
            f"  Text pages  : {self.text_pages}\n"
            f"  Scanned     : {self.scanned_pages}\n"
            f"  Mixed       : {self.mixed_pages}\n"
        )


# ── Core logic ─────────────────────────────────────────────────────────────────

def _image_coverage(page: pdfplumber.page.Page) -> float:
    """
    Calculate what fraction of the page area is covered by embedded images.

    Parameters
    ----------
    page : pdfplumber.page.Page

    Returns
    -------
    float
        Value between 0.0 (no images) and 1.0 (fully covered).

    Notes
    -----
    pdfplumber exposes page.images as a list of dicts with keys:
      x0, y0, x1, y1  (bounding box in PDF points)
    We sum up unique image areas and cap at page area.
    We do NOT try to merge overlapping rectangles for simplicity;
    the slight over-count doesn't affect the classification threshold.
    """
    page_area = page.width * page.height
    if page_area == 0:
        return 0.0

    total_image_area = 0.0
    for img in page.images:
        img_w = abs(img.get("x1", 0) - img.get("x0", 0))
        img_h = abs(img.get("y1", 0) - img.get("y0", 0))
        total_image_area += img_w * img_h

    # Cap at 1.0 (overlapping images might push it above page area)
    return min(total_image_area / page_area, 1.0)


def _classify_page(char_count: int, image_coverage: float) -> PageType:
    """
    Apply the classification rules:

    | has_text | has_image | → PageType  |
    |----------|-----------|-------------|
    |   True   |   False   | TEXT        |
    |   False  |   True    | SCANNED     |
    |   True   |   True    | MIXED       |
    |   False  |   False   | TEXT (*)    |

    (*) Edge case: blank page or PDF with invisible text — we default
        to TEXT so the text pipeline runs (and finds nothing), which
        is safer than skipping the page entirely.
    """
    has_text  = char_count  >= MIN_CHAR_COUNT
    has_image = image_coverage >= MIN_IMAGE_COVERAGE

    if has_text and has_image:
        return PageType.MIXED
    elif has_image:
        return PageType.SCANNED
    else:
        # includes has_text=True, has_image=False  → TEXT
        # includes has_text=False, has_image=False → TEXT (blank/safe default)
        return PageType.TEXT


def analyze_pdf(pdf_path: str | Path) -> PDFAnalysisResult:
    """
    Main entry point — analyze every page of a PDF.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    PDFAnalysisResult
        Structured result with per-page classifications.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist.
    ValueError
        If the file is not a valid PDF.

    Example
    -------
    >>> result = analyze_pdf("documents/invoice.pdf")
    >>> print(result.summary())
    >>> for page in result.pages:
    ...     print(page.page_number, page.page_type)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Analyzing PDF: {pdf_path}")

    pages_info: List[PageInfo] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  → {total_pages} page(s) found")

            for i, page in enumerate(pdf.pages):
                page_number = i + 1  # convert to 1-based

                # Count characters (text layer check)
                chars = page.chars  # list of dicts, one per character
                char_count = len(chars)

                # Calculate image coverage
                coverage = _image_coverage(page)

                # Classify
                page_type = _classify_page(char_count, coverage)

                page_info = PageInfo(
                    page_number    = page_number,
                    page_type      = page_type,
                    width          = page.width,
                    height         = page.height,
                    char_count     = char_count,
                    image_coverage = coverage,
                )
                pages_info.append(page_info)

                logger.debug(
                    f"  Page {page_number}: {page_type.value} "
                    f"(chars={char_count}, img_coverage={coverage:.2f})"
                )

    except Exception as exc:
        raise ValueError(f"Failed to open/parse PDF '{pdf_path}': {exc}") from exc

    result = PDFAnalysisResult(
        pdf_path    = str(pdf_path),
        total_pages = total_pages,
        pages       = pages_info,
    )

    logger.info(result.summary())
    return result
