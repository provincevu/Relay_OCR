"""Production service for splitting one PDF or a folder of PDFs into forms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.image_io import read_image

from .evidence import PageEvidence, build_page_evidence
from .ocr_evidence import analyse_rendered_pages
from .pdf_io import pdf_page_count, render_boundary_reviews, render_pdf, safe_name, split_pdf
from .segmenter import STRONG_COVER_SCORE, SUPPORTING_COVER_SCORE, segment_pages


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence(path: Path) -> list[PageEvidence]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages", payload) if isinstance(payload, dict) else payload
    result = []
    for item in pages:
        if isinstance(item, dict) and item.get("ocr_blocks"):
            page_width, page_height = item.get("image_width"), item.get("image_height")
            if (page_width is None or page_height is None) and item.get("image_path"):
                import cv2

                image = read_image(item["image_path"], cv2.IMREAD_COLOR)
                page_height, page_width = image.shape[:2]
            result.append(build_page_evidence(
                int(item["page_index"]),
                item["ocr_blocks"],
                page_width=page_width,
                page_height=page_height,
            ))
        else:
            evidence = item.get("evidence", item) if isinstance(item, dict) else item
            result.append(PageEvidence.from_dict(evidence))
    return result


def discover_pdf_files(folder: Path | str) -> list[Path]:
    """Return direct PDF children in a deterministic order."""

    source = Path(folder).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    pdfs = sorted(
        (path.resolve() for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found directly in: {source}")
    return pdfs


@dataclass(frozen=True)
class PdfSplitterConfig:
    dpi: int = 200
    scan_ratio: float = 0.45
    use_gpu: bool = False
    render_reviews: bool = True

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be greater than zero")
        if not 0.20 <= self.scan_ratio <= 1.0:
            raise ValueError("scan_ratio must be between 0.20 and 1.0")


class PdfFormSplitterService:
    """Split relay-form PDFs using production OCR and multi-signal boundaries."""

    def __init__(
        self,
        config: PdfSplitterConfig | None = None,
        *,
        detector: Any | None = None,
        recognizer: Any | None = None,
    ):
        self.config = config or PdfSplitterConfig()
        self._detector = detector
        self._recognizer = recognizer

    def _analyse_rendered_pages(self, pages: list[Path]) -> tuple[list[PageEvidence], list[dict[str, Any]]]:
        """Lazy-load OCR once and reuse both models throughout a folder batch."""

        if self._detector is None or self._recognizer is None:
            from src.detection import DocumentTextDetectionService
            from src.recognition import VietnameseRecognitionService

            # VietOCR loads PyTorch. On Windows it must precede Paddle to avoid
            # a native torch/lib/shm.dll WinError 127 in mixed OCR processes.
            self._recognizer = VietnameseRecognitionService(use_gpu=self.config.use_gpu)
            self._detector = DocumentTextDetectionService(use_gpu=self.config.use_gpu)
        return analyse_rendered_pages(
            pages,
            scan_ratio=self.config.scan_ratio,
            detector=self._detector,
            recognizer=self._recognizer,
        )

    @staticmethod
    def _cache_valid(payload: dict[str, Any], input_pdf: Path, page_count: int, sha256: str) -> bool:
        source = payload.get("source", {})
        return (
            source.get("input_pdf") == str(input_pdf.resolve())
            and source.get("sha256") == sha256
            and source.get("page_count") == page_count
            and len(payload.get("pages", [])) == page_count
        )

    def split_file(
        self,
        input_pdf: Path | str,
        output_dir: Path | str,
        *,
        evidence_path: Path | str | None = None,
        reuse_ocr: bool = False,
        documents_dir: Path | str | None = None,
        review_dir: Path | str | None = None,
        filename_prefix: str | None = None,
    ) -> dict[str, Any]:
        source = Path(input_pdf).resolve()
        if not source.is_file() or source.suffix.lower() != ".pdf":
            raise FileNotFoundError(source)
        working_output = Path(output_dir).resolve()
        working_output.mkdir(parents=True, exist_ok=True)
        documents_output = Path(documents_dir).resolve() if documents_dir else working_output / "documents"
        review_output = Path(review_dir).resolve() if review_dir else working_output / "review"
        page_count = pdf_page_count(source)
        sha256 = _sha256(source)
        cache_path = working_output / "ocr_cache.json"

        ocr_mode = "supplied_evidence"
        if evidence_path:
            evidence = load_evidence(Path(evidence_path))
        elif reuse_ocr and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not self._cache_valid(cached, source, page_count, sha256):
                raise ValueError(f"OCR cache does not match source identity or page count: {source}")
            evidence = load_evidence(cache_path)
            ocr_mode = "cached"
        else:
            rendered = render_pdf(source, working_output / "rendered", dpi=self.config.dpi)
            evidence, cache_pages = self._analyse_rendered_pages(rendered)
            cache_payload = {
                "schema_version": "1.0",
                "source": {"input_pdf": str(source), "sha256": sha256, "page_count": page_count},
                "render": {"dpi": self.config.dpi, "scan_ratio": self.config.scan_ratio},
                "pages": cache_pages,
            }
            cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            ocr_mode = "live_production_ocr"

        if len(evidence) != page_count:
            raise ValueError(f"Evidence contains {len(evidence)} pages but {source} contains {page_count}.")
        segments = segment_pages(evidence)
        documents = split_pdf(source, segments, documents_output, filename_prefix=filename_prefix)
        reviews = (
            render_boundary_reviews(documents, review_output)
            if self.config.render_reviews else []
        )
        manifest = {
            "schema_version": "1.0",
            "method": "page1_signature_plus_pagination_state_machine",
            "source": {"input_pdf": str(source), "sha256": sha256, "page_count": page_count},
            "ocr_mode": ocr_mode,
            "decision_policy": {
                "pagination_alone_starts_document": False,
                "strong_page1_cover_score": STRONG_COVER_SCORE,
                "supporting_page1_cover_score": SUPPORTING_COVER_SCORE,
                "terminal_condition": "current_page == total_pages",
            },
            "pages": [page.as_dict() for page in evidence],
            "documents": documents,
            "review_images": reviews,
            "summary": {
                "input_pages": page_count,
                "output_documents": len(documents),
                "output_pages": sum(document["validated_page_count"] for document in documents),
                "documents_with_warnings": sum(bool(document["warnings"]) for document in documents),
            },
        }
        (working_output / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def split_folder(
        self,
        folder_dir: Path | str,
        output_dir: Path | str,
        *,
        reuse_ocr: bool = False,
    ) -> dict[str, Any]:
        folder = Path(folder_dir).resolve()
        inputs = discover_pdf_files(folder)
        output = Path(output_dir).resolve()
        documents_output = output / "documents"
        review_output = output / "review"
        manifests = []
        for input_pdf in inputs:
            source_key = safe_name(input_pdf.stem)
            manifests.append(self.split_file(
                input_pdf,
                output / "sources" / source_key,
                reuse_ocr=reuse_ocr,
                documents_dir=documents_output,
                review_dir=review_output,
                filename_prefix=input_pdf.stem,
            ))
        summary = {
            "input_pdfs": len(manifests),
            "input_pages": sum(item["summary"]["input_pages"] for item in manifests),
            "output_documents": sum(item["summary"]["output_documents"] for item in manifests),
            "output_pages": sum(item["summary"]["output_pages"] for item in manifests),
            "documents_with_warnings": sum(item["summary"]["documents_with_warnings"] for item in manifests),
        }
        batch_manifest = {
            "schema_version": "1.0",
            "input_folder": str(folder),
            "documents_folder": str(documents_output),
            "sources": [
                {
                    "input_pdf": item["source"]["input_pdf"],
                    "manifest": str(output / "sources" / safe_name(Path(item["source"]["input_pdf"]).stem) / "split_manifest.json"),
                    "summary": item["summary"],
                }
                for item in manifests
            ],
            "summary": summary,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "batch_manifest.json").write_text(
            json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return batch_manifest
