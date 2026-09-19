# Mục đích: render các page dữ liệu 3+ từ PDF và tạo dataset train/val/test.
# Chức năng: nhận diện page bắt đầu theo logic export_pages_3plus, render PNG,
# rồi chia theo phiếu/PDF để không làm rò rỉ các page cùng một phiếu giữa các tập.
# Luồng xử lý: PDF -> page 2 classification -> render -> split theo phiếu -> manifest.
# Input/output: đọc PDF trong data/pdf; ghi ảnh và manifest vào data/page3plus_dataset.
# Contract: mọi page của cùng một PDF chỉ thuộc đúng một split.

"""Build a balanced image dataset from the data pages of PDF forms.

Pages are split inside each PDF/form because one form can have a different
number of page 3+ pages than another form. The assignment is randomized with a
seed, then follows the requested small-form rules.

The first data page follows ``scripts/export_pages_3plus.py`` exactly:

* physical page 2 is a data page -> render from page 2;
* physical page 2 is a comment page -> skip it and render from page 3.

Example:

    python scripts/build_page3plus_dataset.py

For a form with 2 pages, the split is 1 train and 1 test. For 3-5 pages, each
split gets one page first and remaining pages go in the order train, test, val.
For more pages, the same train/test/val remainder priority is applied to an
otherwise even split. A form with only 1 page is assigned to train because it
cannot appear in two splits at once.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from .export_pages_3plus import find_first_data_page
except ImportError:  # Allows ``python scripts/build_page3plus_dataset.py``.
    from export_pages_3plus import find_first_data_page


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "pdf"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "page3plus_dataset"
DEFAULT_DPI = 400
DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")
PAGE_NAME_PATTERN = re.compile(r"^page_(\d+)\.png$")


@dataclass(frozen=True)
class PdfRecord:
    """Rendered pages belonging to one source PDF/form."""

    pdf_path: Path
    first_data_page: int
    rendered_pages: tuple[Path, ...]
    physical_pages: tuple[int, ...]

    @property
    def page_count(self) -> int:
        return len(self.rendered_pages)


@dataclass(frozen=True)
class PageAssignment:
    """One rendered page assigned to one dataset split."""

    record: PdfRecord
    page_index: int
    split_name: str

    @property
    def rendered_path(self) -> Path:
        return self.record.rendered_pages[self.page_index]

    @property
    def physical_page(self) -> int:
        return self.record.physical_pages[self.page_index]


def _page_number(path: Path) -> int:
    """Extract a physical page number from ``page_003.png``."""
    match = PAGE_NAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Tên ảnh không đúng định dạng page_NNN.png: {path}")
    return int(match.group(1))


def _pdf_files(input_directory: Path) -> list[Path]:
    """Find PDF files case-insensitively in deterministic order."""
    return sorted(
        (
            path
            for path in input_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )


def _find_renderer(renderer: str | None) -> str:
    """Resolve Ghostscript, which is also used by the existing classifier."""
    candidates = [renderer] if renderer else ["gswin64c", "gswin32c", "gs"]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Không tìm thấy Ghostscript. Cài gswin64c (Windows) hoặc truyền "
        "--renderer tới executable Ghostscript."
    )


def _run_ghostscript(command: list[str], description: str) -> None:
    """Run Ghostscript and include a useful error tail on failure."""
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    error = (result.stderr or result.stdout or "").strip()
    if len(error) > 1200:
        error = error[-1200:]
    detail = f"\n{error}" if error else ""
    raise RuntimeError(f"{description} thất bại (exit {result.returncode}).{detail}")


def render_pdf(
    renderer: str,
    pdf_path: Path,
    staging_directory: Path,
    dpi: int,
) -> PdfRecord | None:
    """Classify page 2 and render all data pages from the selected page onward."""
    document_directory = staging_directory / pdf_path.stem
    document_directory.mkdir(parents=True, exist_ok=True)

    # This calls the shared implementation in export_pages_3plus.py, including
    # _is_comment_page(). Therefore page 2 -> page 2 and comment page 2 -> page 3.
    first_data_page = find_first_data_page(
        renderer,
        pdf_path,
        document_directory,
    )
    if first_data_page is None:
        return None

    output_pattern = document_directory / "page_%03d.png"
    command = [
        renderer,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        f"-r{dpi}",
        f"-dFirstPage={first_data_page}",
        f"-sOutputFile={output_pattern}",
        str(pdf_path),
    ]
    _run_ghostscript(command, f"Render {pdf_path.name}")

    rendered_pages = tuple(
        sorted(
            document_directory.glob("page_*.png"),
            key=_page_number,
        )
    )
    if not rendered_pages:
        return None

    # Ghostscript numbers an output sequence from 001 even when -dFirstPage
    # is 2 or 3. Convert that sequence back to physical PDF page numbers.
    physical_pages = tuple(
        first_data_page + offset
        for offset in range(len(rendered_pages))
    )
    return PdfRecord(
        pdf_path=pdf_path,
        first_data_page=first_data_page,
        rendered_pages=rendered_pages,
        physical_pages=physical_pages,
    )


def _split_counts(page_count: int) -> dict[str, int]:
    """Return the requested train/val/test counts for one PDF."""
    if page_count <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    if page_count == 1:
        return {"train": 1, "val": 0, "test": 0}
    if page_count == 2:
        return {"train": 1, "val": 0, "test": 1}

    base_count, remainder = divmod(page_count, 3)
    counts = {name: base_count for name in SPLIT_NAMES}
    # Remainders deliberately go train -> test -> val, as requested.
    for split_name in ("train", "test", "val")[:remainder]:
        counts[split_name] += 1
    return counts


def split_records(
    records: list[PdfRecord],
    seed: int,
) -> dict[str, list[PageAssignment]]:
    """Shuffle and split pages independently inside every PDF/form."""
    randomizer = random.Random(seed)
    assignments_by_split = {name: [] for name in SPLIT_NAMES}

    for record in sorted(records, key=lambda item: item.pdf_path.name.casefold()):
        shuffled_indices = list(range(record.page_count))
        randomizer.shuffle(shuffled_indices)
        counts = _split_counts(record.page_count)
        cursor = 0
        for split_name in ("train", "test", "val"):
            split_count = counts[split_name]
            for page_index in shuffled_indices[cursor:cursor + split_count]:
                assignments_by_split[split_name].append(
                    PageAssignment(record, page_index, split_name)
                )
            cursor += split_count

    # Shuffle final dataset order as well, so manifests/loaders do not always
    # see source PDFs in filename order. The seed keeps it reproducible.
    for assignments in assignments_by_split.values():
        randomizer.shuffle(assignments)
    return assignments_by_split


def _safe_stem(stem: str) -> str:
    """Make a source stem safe and stable as part of an output filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "document"


def _image_name(record: PdfRecord, physical_page: int) -> str:
    return f"{_safe_stem(record.pdf_path.stem)}-page-{physical_page:03d}.png"


def _summary(
    assignments: list[PageAssignment],
    split_name: str,
) -> dict:
    """Build the user-facing counts for one split."""
    records = {
        assignment.record.pdf_path: assignment.record
        for assignment in assignments
    }
    return {
        "split": split_name,
        "pdf_count": len(records),
        "page_count": len(assignments),
        "pdf_names": sorted(
            (record.pdf_path.name for record in records.values()),
            key=str.casefold,
        ),
    }


def _write_dataset(
    output_directory: Path,
    input_directory: Path,
    assignments_by_split: dict[str, list[PageAssignment]],
    seed: int,
    dpi: int,
) -> dict:
    """Copy rendered images into split directories and write manifests."""
    output_directory.mkdir(parents=True, exist_ok=True)
    source_stems: dict[str, Path] = {}
    split_payloads: dict[str, dict] = {}

    for split_name in SPLIT_NAMES:
        split_directory = output_directory / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        assignments = assignments_by_split[split_name]
        pages_payload = []
        assignments_by_pdf: dict[Path, list[dict]] = {}
        records_by_pdf: dict[Path, PdfRecord] = {}
        for assignment in assignments:
            record = assignment.record
            safe_stem = _safe_stem(record.pdf_path.stem)
            previous_source = source_stems.setdefault(
                safe_stem,
                record.pdf_path,
            )
            if previous_source != record.pdf_path:
                raise ValueError(
                    "Hai PDF có cùng tên output sau khi làm sạch: "
                    f"{previous_source.name}, {record.pdf_path.name}"
                )

            image_name = _image_name(record, assignment.physical_page)
            destination = split_directory / image_name
            shutil.copy2(assignment.rendered_path, destination)
            relative_image = (Path(split_name) / image_name).as_posix()
            page_payload = {
                "image": relative_image,
                "pdf_name": record.pdf_path.name,
                "physical_page": assignment.physical_page,
                "first_data_page": record.first_data_page,
            }
            pages_payload.append(page_payload)
            assignments_by_pdf.setdefault(record.pdf_path, []).append(page_payload)
            records_by_pdf[record.pdf_path] = record

        pdf_payload = []
        for pdf_path in sorted(assignments_by_pdf, key=lambda path: path.name.casefold()):
            record = records_by_pdf[pdf_path]
            record_pages = sorted(
                assignments_by_pdf[pdf_path],
                key=lambda page: page["physical_page"],
            )
            pdf_payload.append(
                {
                    "pdf_name": pdf_path.name,
                    "first_data_page": record.first_data_page,
                    "source_page_count": record.page_count,
                    "split_page_count": len(record_pages),
                    "pages": record_pages,
                }
            )

        split_payloads[split_name] = {
            "summary": _summary(
                assignments,
                split_name,
            ),
            "pdfs": pdf_payload,
            "pages": pages_payload,
        }

        (split_directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split": split_name,
                    "summary": split_payloads[split_name]["summary"],
                    "pdfs": pdf_payload,
                    "pages": pages_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "dataset_type": "page3plus_png",
        "source_directory": str(input_directory),
        "dpi": dpi,
        "seed": seed,
        "split_unit": "page_within_pdf_form",
        "split_rule": {
            "one_page": {"train": 1, "val": 0, "test": 0},
            "two_pages": {"train": 1, "val": 0, "test": 1},
            "three_or_more_pages": (
                "floor(page_count / 3) per split; remaining pages go "
                "train, then test, then val"
            ),
            "shuffle": "shuffle pages inside each PDF and each final split",
        },
        "page_rule": {
            "physical_page_checked": 2,
            "if_page_2_is_comment": "start_at_physical_page_3",
            "otherwise": "start_at_physical_page_2",
        },
        "splits": split_payloads,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _configure_console() -> None:
    """Make Vietnamese progress/help text safe in a default Windows console."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render page dữ liệu 3+ từ data/pdf và chia page trong từng "
            "phiếu thành train/val/test theo rule cân bằng."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"Folder PDF (mặc định: {DEFAULT_INPUT_DIRECTORY}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Folder dataset (mặc định: {DEFAULT_OUTPUT_DIRECTORY}).",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--renderer",
        help=(
            "Đường dẫn hoặc tên executable Ghostscript; mặc định tự tìm "
            "trong PATH."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè output directory sau khi render hoàn tất.",
    )
    return parser.parse_args()


def _validate_output_target(input_directory: Path, output_directory: Path) -> None:
    """Prevent --overwrite from targeting the source tree or a filesystem root."""
    project_directory = PROJECT_DIRECTORY.resolve()
    if output_directory == input_directory:
        raise ValueError("Output không được trùng với folder input chứa PDF.")
    if output_directory == project_directory:
        raise ValueError("Không cho phép dùng --overwrite trên thư mục project.")
    if output_directory == Path(output_directory.anchor):
        raise ValueError("Không cho phép dùng --overwrite trên filesystem root.")
    if input_directory.is_relative_to(output_directory):
        raise ValueError(
            "Output không được là thư mục chứa input; điều này có thể xoá PDF.")


def main() -> None:
    _configure_console()
    arguments = _parse_arguments()
    input_directory = arguments.input.resolve()
    output_directory = arguments.output.resolve()

    if arguments.dpi <= 0:
        raise ValueError("DPI phải lớn hơn 0.")
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Không tìm thấy folder input: {input_directory}")
    _validate_output_target(input_directory, output_directory)
    if output_directory.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output đã tồn tại: {output_directory}. Dùng --overwrite nếu muốn tạo lại."
        )

    pdf_files = _pdf_files(input_directory)
    if not pdf_files:
        raise FileNotFoundError(f"Không có file PDF trong: {input_directory}")
    renderer = _find_renderer(arguments.renderer)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=str(output_directory.parent),
        )
    )
    try:
        rendered_root = staging_root / "rendered"
        records: list[PdfRecord] = []
        skipped: list[str] = []
        failures: list[str] = []
        for index, pdf_path in enumerate(pdf_files, start=1):
            print(f"[{index}/{len(pdf_files)}] Đang xử lý {pdf_path.name}...", flush=True)
            try:
                record = render_pdf(renderer, pdf_path, rendered_root, arguments.dpi)
            except Exception as error:  # Report every bad source before aborting.
                failures.append(f"{pdf_path.name}: {error}")
                continue
            if record is None:
                skipped.append(pdf_path.name)
                print(f"  Bỏ qua {pdf_path.name}: không có page dữ liệu.", flush=True)
                continue
            records.append(record)
            print(
                f"  Page vật lý {record.first_data_page}+ -> "
                f"{record.page_count} PNG",
                flush=True,
            )

        if failures:
            details = "\n".join(f"- {failure}" for failure in failures)
            raise RuntimeError(
                "Không thể hoàn tất vì có PDF lỗi:\n" + details
            )
        if not records:
            raise RuntimeError("Không tạo được page dữ liệu nào.")

        assignments_by_split = split_records(records, arguments.seed)

        dataset_staging_directory = staging_root / "dataset"
        manifest = _write_dataset(
            dataset_staging_directory,
            input_directory,
            assignments_by_split,
            arguments.seed,
            arguments.dpi,
        )
        manifest["skipped_pdfs"] = skipped
        (dataset_staging_directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if output_directory.exists():
            if not arguments.overwrite:
                raise FileExistsError(f"Output đã tồn tại: {output_directory}")
            shutil.rmtree(output_directory)
        shutil.move(str(dataset_staging_directory), str(output_directory))

        print(f"\nĐã tạo dataset: {output_directory}")
        for split_name in SPLIT_NAMES:
            summary = manifest["splits"][split_name]["summary"]
            print(
                f"{split_name}: {summary['pdf_count']} PDF, "
                f"{summary['page_count']} page"
            )
        if skipped:
            print(f"PDF bị bỏ qua: {len(skipped)}")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    main()
