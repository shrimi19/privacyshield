"""
text_pipeline/
--------------
Pipeline for text-based PDF pages.

Public API:
  extract_text_pages()        → PDFExtraction
  get_charboxes_for_span()    → List[CharBox]
  get_merged_bbox_for_span()  → Optional[BoundingBox]
  BoundingBox, CharBox, WordBox, LineBox, PageExtraction, PDFExtraction
"""

from .extractor import (
    BoundingBox,
    CharBox,
    WordBox,
    LineBox,
    PageExtraction,
    PDFExtraction,
    extract_text_pages,
    extract_page,
    get_charboxes_for_span,
    get_merged_bbox_for_span,
)

__all__ = [
    "BoundingBox",
    "CharBox",
    "WordBox",
    "LineBox",
    "PageExtraction",
    "PDFExtraction",
    "extract_text_pages",
    "extract_page",
    "get_charboxes_for_span",
    "get_merged_bbox_for_span",
]
