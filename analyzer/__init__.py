"""
analyzer/
---------
PDF type detection module.

Public API:
  analyze_pdf()      → PDFAnalysisResult
  PageType           → Enum: TEXT | SCANNED | MIXED
  PageInfo           → dataclass: per-page metadata
  PDFAnalysisResult  → dataclass: full document analysis
"""

from .pdf_analyzer import (
    PageType,
    PageInfo,
    PDFAnalysisResult,
    analyze_pdf,
)

__all__ = [
    "PageType",
    "PageInfo",
    "PDFAnalysisResult",
    "analyze_pdf",
]
