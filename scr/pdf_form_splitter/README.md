# Production PDF form splitter

This package is the production entry point for splitting combined scanned PDFs
into individual relay-adjustment forms. It depends only on production packages
under `src/`: detection, recognition, and shared pagination analysis. It has no
runtime dependency on `lab/`.

Boundary decisions combine weighted page-1 cover characteristics, dynamic
`x/y` pagination, ticket continuity, and terminal pages where `x == y`.
Pagination equal to one is not sufficient by itself to create a new document.

## Install

```powershell
python -m pip install -r src\detection\requirements.txt
python -m pip install -r src\recognition\requirements.txt
python -m pip install -r src\pdf_form_splitter\requirements.txt
```

Poppler's `pdftoppm` must be available on `PATH`. On Windows, the package also
detects the bundled Codex Poppler runtime under
`%USERPROFILE%\.cache\codex-runtimes\`.

## CLI

Split one PDF:

```powershell
python -m src.pdf_form_splitter data\pdf\combined.pdf
```

Split all direct PDF children of a folder into one documents directory:

```powershell
python -m src.pdf_form_splitter --folder_dir data\pdf --output output\pdf_form_splitter\pdf
```

Folder scanning is non-recursive. Output PDF names include their source stem so
documents from different source PDFs cannot overwrite one another. Each source
keeps an isolated OCR cache and manifest under `sources/<source-name>/`; the
batch root contains `batch_manifest.json`.
Detection and recognition models are loaded lazily and reused across every PDF
in the same `split_folder` call.

PDF and rendered-image paths may contain Vietnamese or other Unicode
characters. The production pipeline uses Unicode-safe image IO on Windows.

## Python API

```python
from src.pdf_form_splitter import PdfFormSplitterService, PdfSplitterConfig

service = PdfFormSplitterService(PdfSplitterConfig(dpi=200, use_gpu=False))
manifest = service.split_folder(
    "data/pdf",
    "output/pdf_form_splitter/pdf",
    reuse_ocr=True,
)
```

The source PDFs are never modified. Splitting a digitally signed PDF creates
new documents, so original document-level signatures are not preserved.
