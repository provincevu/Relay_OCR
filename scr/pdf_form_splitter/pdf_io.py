"""Render, split and validate PDFs while preserving the source file."""

from __future__ import annotations

import re
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from .segmenter import DocumentSegment


def _pypdf():
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("pypdf is required; install src/pdf_form_splitter/requirements.txt") from exc
    return PdfReader, PdfWriter


def pdf_page_count(path: Path) -> int:
    PdfReader, _ = _pypdf()
    return len(PdfReader(str(path)).pages)


def poppler_binary(name: str) -> str:
    found = shutil.which(name)
    candidates: list[Path] = []
    if found:
        found_path = Path(found)
        candidates.append(found_path)
        if found_path.suffix.lower() == ".cmd" and len(found_path.parents) >= 3:
            candidates.insert(0, found_path.parents[2] / "native" / "poppler" / "Library" / "bin" / f"{name}.exe")
    runtime_root = Path.home() / ".cache" / "codex-runtimes"
    if runtime_root.is_dir():
        candidates.extend(runtime_root.glob(f"*/dependencies/native/poppler/Library/bin/{name}.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"{name} was not found. Install Poppler and add its bin directory to PATH.")


def render_pdf(path: Path, output_dir: Path, *, dpi: int = 150) -> list[Path]:
    """Render every page with Poppler for OCR and visual review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("page-*.png"):
        old.unlink()
    prefix = output_dir / "page"
    completed = subprocess.run(
        [poppler_binary("pdftoppm"), "-png", "-r", str(dpi), str(path), str(prefix)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {completed.stderr.strip()}")
    pages = sorted(output_dir.glob("page-*.png"), key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1)))
    expected = pdf_page_count(path)
    if len(pages) != expected:
        raise RuntimeError(f"Rendered {len(pages)} pages but PDF contains {expected}.")
    return pages


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
    return cleaned or "unknown_ticket"


def split_pdf(
    input_pdf: Path,
    segments: Iterable[DocumentSegment],
    output_dir: Path,
    *,
    filename_prefix: str | None = None,
) -> list[dict]:
    PdfReader, PdfWriter = _pypdf()
    reader = PdfReader(str(input_pdf))
    output_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    prefix = f"{safe_name(filename_prefix)}__" if filename_prefix else ""
    for segment in segments:
        ticket = safe_name(segment.ticket_number or f"pages_{segment.start_page:04d}_{segment.end_page:04d}")
        output_path = output_dir / f"{prefix}{segment.segment_index:03d}_{ticket}.pdf"
        writer = PdfWriter()
        for page_index in range(segment.start_page - 1, segment.end_page):
            writer.add_page(reader.pages[page_index])
        writer.add_metadata({"/Title": ticket, "/Subject": f"Split from {input_pdf.name}"})
        with output_path.open("wb") as stream:
            writer.write(stream)
        written_pages = len(PdfReader(str(output_path)).pages)
        if written_pages != segment.page_count:
            raise RuntimeError(f"Output {output_path} has {written_pages} pages; expected {segment.page_count}.")
        documents.append({**segment.as_dict(), "output_pdf": str(output_path), "validated_page_count": written_pages})
    return documents


def render_boundary_reviews(documents: Iterable[dict], output_dir: Path, *, dpi: int = 100) -> list[str]:
    """Render first and last pages of each result for rapid visual QA."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for document in documents:
        pdf = Path(document["output_pdf"])
        page_count = int(document["validated_page_count"])
        boundaries = [("first", 1)]
        if page_count > 1:
            boundaries.append(("last", page_count))
        for label, page_number in boundaries:
            prefix = output_dir / f"{pdf.stem}_{label}"
            completed = subprocess.run(
                [poppler_binary("pdftoppm"), "-png", "-r", str(dpi), "-f", str(page_number), "-l", str(page_number), "-singlefile", str(pdf), str(prefix)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Could not render review for {pdf}: {completed.stderr.strip()}")
            rendered.append(str(prefix.with_suffix(".png")))
    return rendered
