"""Command-line entry point for the production PDF form splitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import PdfFormSplitterService, PdfSplitterConfig


DEFAULT_OUTPUT = Path("output") / "pdf_form_splitter"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split combined relay-form PDFs using page-1 and pagination evidence")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_pdf", nargs="?", type=Path, help="One combined PDF to split")
    source.add_argument("--folder_dir", type=Path, help="Folder containing combined PDFs (direct children only)")
    parser.add_argument("--output", type=Path, help="Output root; defaults to output/pdf_form_splitter/<source-name>")
    parser.add_argument("--evidence", type=Path, help="Use supplied page evidence JSON for one input PDF")
    parser.add_argument("--reuse-ocr", action="store_true", help="Reuse a matching per-source OCR cache")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--scan-ratio", type=float, default=0.45, help="Top portion recognised for header and page-1 signatures")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-review-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.folder_dir and args.evidence:
        parser.error("--evidence can only be used with one input_pdf, not with --folder_dir")
    service = PdfFormSplitterService(PdfSplitterConfig(
        dpi=args.dpi,
        scan_ratio=args.scan_ratio,
        use_gpu=args.gpu,
        render_reviews=not args.no_review_render,
    ))
    if args.folder_dir:
        folder = args.folder_dir.resolve()
        output = (args.output or (DEFAULT_OUTPUT / folder.name)).resolve()
        result = service.split_folder(folder, output, reuse_ocr=args.reuse_ocr)
    else:
        source = args.input_pdf.resolve()
        output = (args.output or (DEFAULT_OUTPUT / source.stem)).resolve()
        result = service.split_file(
            source,
            output,
            evidence_path=args.evidence,
            reuse_ocr=args.reuse_ocr,
        )
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0
