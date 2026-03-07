"""
extractor.py
------------
Extracts text from text-based PDF pages with EXACT character coordinates.

Why coordinates matter:
  Later in the pipeline, ner_engine.py will find spans of PII text
  (e.g., "John Smith" at characters 45–54). To draw a redaction box
  over "John Smith" in the final PDF, we need to know the PIXEL/POINT
  position of every character. This extractor captures that.

What we extract per page:
  - Full text string (for NER to run on)
  - List of CharBox objects: each character's position + metadata
  - List of WordBox objects: merged from chars (more useful for NER span mapping)
  - List of LineBox objects: grouped by line (useful for context)

Coordinate system:
  pdfplumber uses PDF coordinate space:
    (0, 0) is bottom-left of the page
    x increases rightward, y increases upward
  We preserve these coordinates as-is since pdf_rebuilder.py uses them
  for placing redaction overlays.

Design decision — why WordBox AND CharBox?
  NER returns spans like (start_char=10, end_char=20) over the full text.
  To find which boxes to redact, we need char-level granularity.
  WordBox is a convenience for debugging and the PII report UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """
    A rectangle in PDF point coordinates.

    PDF coordinate origin (0,0) is at the BOTTOM-LEFT of the page.
    x0, y0 = top-left corner of the box (visually)
    x1, y1 = bottom-right corner

    Note: pdfplumber's y0 < y1 means y0 is visually higher on the page
    because PDF y-axis runs upward. We keep this as-is for consistency
    with pdfplumber's own conventions.
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class CharBox:
    """
    A single character with its position and metadata.

    Attributes
    ----------
    char : str
        The character itself (e.g., 'J').
    char_index : int
        Position of this character in the full page text string.
        Critical for mapping NER span offsets back to positions.
    bbox : BoundingBox
        Exact position on the page.
    font_name : str
        Font used (e.g., 'Helvetica-Bold'). Useful for detecting headers.
    font_size : float
        Font size in points.
    page_number : int
        1-based page number this character belongs to.
    """
    char:        str
    char_index:  int
    bbox:        BoundingBox
    font_name:   str
    font_size:   float
    page_number: int


@dataclass
class WordBox:
    """
    A word (sequence of chars) with a merged bounding box.

    Built by merging consecutive CharBoxes separated by spaces.
    The bbox covers the entire word from first to last character.

    Attributes
    ----------
    text : str
        The word text (e.g., 'John').
    start_char : int
        char_index of the first character in this word.
    end_char : int
        char_index of the last character in this word (inclusive).
    bbox : BoundingBox
        Merged bounding box covering the full word.
    page_number : int
        1-based page number.
    """
    text:        str
    start_char:  int
    end_char:    int
    bbox:        BoundingBox
    page_number: int


@dataclass
class LineBox:
    """
    A line of text with its merged bounding box.

    Lines are determined by grouping characters with the same
    approximate vertical (y) position.

    Attributes
    ----------
    text : str
        Full line text (e.g., 'Name: John Smith').
    start_char : int
        char_index of first character on this line.
    end_char : int
        char_index of last character on this line.
    bbox : BoundingBox
        Merged bounding box for the whole line.
    page_number : int
        1-based page number.
    """
    text:        str
    start_char:  int
    end_char:    int
    bbox:        BoundingBox
    page_number: int


@dataclass
class PageExtraction:
    """
    All extracted data for a single page.

    Attributes
    ----------
    page_number : int
        1-based page index.
    full_text : str
        The complete text of the page as a single string.
        NER runs on this string; char_index values in CharBox
        correspond to positions in this string.
    chars : List[CharBox]
        Every character with position.
    words : List[WordBox]
        Words derived from chars (convenience layer).
    lines : List[LineBox]
        Lines derived from chars (convenience layer).
    width : float
        Page width in PDF points.
    height : float
        Page height in PDF points.
    """
    page_number: int
    full_text:   str
    chars:       List[CharBox]    = field(default_factory=list)
    words:       List[WordBox]    = field(default_factory=list)
    lines:       List[LineBox]    = field(default_factory=list)
    width:       float            = 0.0
    height:      float            = 0.0


@dataclass
class PDFExtraction:
    """
    Complete extraction result for a PDF file.

    Attributes
    ----------
    pdf_path : str
        Path to the source file.
    pages : List[PageExtraction]
        One entry per page that was extracted.
    """
    pdf_path: str
    pages:    List[PageExtraction] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All pages concatenated with page separators."""
        return "\n\n--- PAGE BREAK ---\n\n".join(p.full_text for p in self.pages)

    def get_page(self, page_number: int) -> Optional[PageExtraction]:
        """Retrieve extraction for a specific 1-based page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None


# ── Internal helpers ───────────────────────────────────────────────────────────

# Vertical tolerance: chars within this many points of each other
# are considered on the same line.
LINE_Y_TOLERANCE = 3.0


def _build_charboxes(raw_chars: list, page_number: int) -> tuple[str, List[CharBox]]:
    """
    Convert pdfplumber's raw char dicts into CharBox objects.

    pdfplumber char dict keys we use:
      text  → the character string
      x0    → left edge
      top   → distance from top of page (pdfplumber uses top-origin y)
      x1    → right edge
      bottom→ distance from top of page to bottom of char
      fontname → font name
      size  → font size in points

    We store y coordinates as pdfplumber gives them (top-origin)
    for consistency within this module.

    Returns
    -------
    full_text : str
        All characters concatenated into a string.
    charboxes : List[CharBox]
    """
    charboxes: List[CharBox] = []
    text_parts: List[str] = []

    for i, ch in enumerate(raw_chars):
        char_str  = ch.get("text", "")
        x0        = ch.get("x0", 0.0)
        y0        = ch.get("top", 0.0)      # top of character
        x1        = ch.get("x1", 0.0)
        y1        = ch.get("bottom", 0.0)   # bottom of character
        font_name = ch.get("fontname", "")
        font_size = ch.get("size", 0.0)

        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

        charboxes.append(CharBox(
            char        = char_str,
            char_index  = i,
            bbox        = bbox,
            font_name   = font_name,
            font_size   = font_size,
            page_number = page_number,
        ))
        text_parts.append(char_str)

    full_text = "".join(text_parts)
    return full_text, charboxes


def _merge_bbox(boxes: List[BoundingBox]) -> BoundingBox:
    """Merge multiple bounding boxes into one that contains all of them."""
    return BoundingBox(
        x0 = min(b.x0 for b in boxes),
        y0 = min(b.y0 for b in boxes),
        x1 = max(b.x1 for b in boxes),
        y1 = max(b.y1 for b in boxes),
    )


def _build_wordboxes(charboxes: List[CharBox]) -> List[WordBox]:
    """
    Group CharBoxes into words by splitting on whitespace characters.

    Strategy:
      Walk through charboxes. When we hit a non-space char, start
      accumulating a word. When we hit a space (or end), close the word.
    """
    words: List[WordBox] = []
    current_chars: List[CharBox] = []

    def flush_word():
        if not current_chars:
            return
        text = "".join(c.char for c in current_chars)
        bbox = _merge_bbox([c.bbox for c in current_chars])
        words.append(WordBox(
            text        = text,
            start_char  = current_chars[0].char_index,
            end_char    = current_chars[-1].char_index,
            bbox        = bbox,
            page_number = current_chars[0].page_number,
        ))

    for cb in charboxes:
        if cb.char.strip() == "":
            flush_word()
            current_chars = []
        else:
            current_chars.append(cb)

    flush_word()  # catch any trailing word
    return words


def _build_lineboxes(charboxes: List[CharBox]) -> List[LineBox]:
    """
    Group CharBoxes into lines by their vertical (y0) position.

    Characters with y0 values within LINE_Y_TOLERANCE of each other
    are considered to be on the same line.

    This handles slight rendering differences where characters on the
    same visual line have slightly different y coordinates.
    """
    if not charboxes:
        return []

    lines: List[LineBox] = []
    current_line_chars: List[CharBox] = []
    current_y: float = charboxes[0].bbox.y0

    for cb in charboxes:
        if abs(cb.bbox.y0 - current_y) <= LINE_Y_TOLERANCE:
            current_line_chars.append(cb)
        else:
            # New line — flush current
            if current_line_chars:
                text = "".join(c.char for c in current_line_chars)
                bbox = _merge_bbox([c.bbox for c in current_line_chars])
                lines.append(LineBox(
                    text        = text,
                    start_char  = current_line_chars[0].char_index,
                    end_char    = current_line_chars[-1].char_index,
                    bbox        = bbox,
                    page_number = current_line_chars[0].page_number,
                ))
            current_line_chars = [cb]
            current_y = cb.bbox.y0

    # Flush last line
    if current_line_chars:
        text = "".join(c.char for c in current_line_chars)
        bbox = _merge_bbox([c.bbox for c in current_line_chars])
        lines.append(LineBox(
            text        = text,
            start_char  = current_line_chars[0].char_index,
            end_char    = current_line_chars[-1].char_index,
            bbox        = bbox,
            page_number = current_line_chars[0].page_number,
        ))

    return lines


# ── Core logic ─────────────────────────────────────────────────────────────────

def extract_page(pdf_page: pdfplumber.page.Page, page_number: int) -> PageExtraction:
    """
    Extract text and coordinates from a single pdfplumber page.

    Parameters
    ----------
    pdf_page : pdfplumber.page.Page
        An open pdfplumber page object.
    page_number : int
        1-based page number (for labeling output).

    Returns
    -------
    PageExtraction
        Structured extraction with chars, words, lines.
    """
    raw_chars = pdf_page.chars  # list of dicts from pdfplumber

    if not raw_chars:
        logger.debug(f"  Page {page_number}: no characters found (may be scanned)")
        return PageExtraction(
            page_number = page_number,
            full_text   = "",
            width       = pdf_page.width,
            height      = pdf_page.height,
        )

    full_text, charboxes = _build_charboxes(raw_chars, page_number)
    words = _build_wordboxes(charboxes)
    lines = _build_lineboxes(charboxes)

    logger.debug(
        f"  Page {page_number}: {len(charboxes)} chars, "
        f"{len(words)} words, {len(lines)} lines"
    )

    return PageExtraction(
        page_number = page_number,
        full_text   = full_text,
        chars       = charboxes,
        words       = words,
        lines       = lines,
        width       = pdf_page.width,
        height      = pdf_page.height,
    )


def extract_text_pages(
    pdf_path: str | Path,
    page_numbers: Optional[List[int]] = None,
) -> PDFExtraction:
    """
    Extract text and coordinates from specified pages of a PDF.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.
    page_numbers : List[int], optional
        1-based page numbers to extract. If None, extracts ALL pages.
        Typically you'd pass in the text_pages + mixed_pages from
        PDFAnalysisResult.

    Returns
    -------
    PDFExtraction
        All page extractions collected in one object.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist.
    ValueError
        If the PDF cannot be opened.

    Example
    -------
    >>> from privacyshield.analyzer import analyze_pdf
    >>> from privacyshield.text_pipeline.extractor import extract_text_pages
    >>>
    >>> analysis = analyze_pdf("invoice.pdf")
    >>> # Only extract pages that have a text layer
    >>> extraction = extract_text_pages(
    ...     "invoice.pdf",
    ...     page_numbers=analysis.text_pages + analysis.mixed_pages
    ... )
    >>> for page in extraction.pages:
    ...     print(f"Page {page.page_number}: {len(page.words)} words")
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Extracting text from: {pdf_path}")

    result = PDFExtraction(pdf_path=str(pdf_path))

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)

            # Determine which pages to process
            if page_numbers is None:
                pages_to_process = list(range(1, total + 1))
            else:
                # Filter out invalid page numbers
                pages_to_process = [
                    n for n in page_numbers
                    if 1 <= n <= total
                ]

            logger.info(f"  → Processing pages: {pages_to_process}")

            for page_num in pages_to_process:
                pdf_page = pdf.pages[page_num - 1]  # pdfplumber is 0-indexed
                page_extraction = extract_page(pdf_page, page_number=page_num)
                result.pages.append(page_extraction)

    except Exception as exc:
        raise ValueError(f"Failed to extract text from '{pdf_path}': {exc}") from exc

    logger.info(
        f"  → Extracted {len(result.pages)} page(s), "
        f"{sum(len(p.chars) for p in result.pages)} total chars"
    )
    return result


def get_charboxes_for_span(
    page_extraction: PageExtraction,
    start_char: int,
    end_char: int,
) -> List[CharBox]:
    """
    Return all CharBoxes whose char_index falls within [start_char, end_char].

    This is the KEY function used by the redactor:
    NER says "PII at chars 45–54" → this returns the CharBoxes
    so we know which rectangles to black out.

    Parameters
    ----------
    page_extraction : PageExtraction
    start_char : int
        Inclusive start of the span (from NER output).
    end_char : int
        Inclusive end of the span.

    Returns
    -------
    List[CharBox]
        All characters in the span, in order.
    """
    return [
        cb for cb in page_extraction.chars
        if start_char <= cb.char_index <= end_char
    ]


def get_merged_bbox_for_span(
    page_extraction: PageExtraction,
    start_char: int,
    end_char: int,
) -> Optional[BoundingBox]:
    """
    Get a single merged bounding box for a span of text.

    Useful when you want ONE rectangle to redact an entire PII span.

    Returns None if no characters are found in the span.
    """
    chars_in_span = get_charboxes_for_span(page_extraction, start_char, end_char)
    if not chars_in_span:
        return None
    return _merge_bbox([c.bbox for c in chars_in_span])
