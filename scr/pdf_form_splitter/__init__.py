"""Production splitting of combined relay-adjustment PDF forms."""

from .evidence import PageEvidence, build_page_evidence
from .segmenter import DocumentSegment, segment_pages
from .service import PdfFormSplitterService, PdfSplitterConfig, discover_pdf_files

__all__ = [
    "DocumentSegment",
    "PageEvidence",
    "PdfFormSplitterService",
    "PdfSplitterConfig",
    "build_page_evidence",
    "discover_pdf_files",
    "segment_pages",
]
