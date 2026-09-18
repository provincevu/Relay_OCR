# Mục đích: thử nghiệm Rule Induction tương tác cho key-value trên các page 3+.
# Chức năng: render page 3+, chạy cache detector/recognition và tạo HTML; hỗ trợ đọc folder ảnh đã lọc.
# Luồng xử lý: folder PDF → render page 3+ → recognition cache/OCR → HTML.
# Cấu hình/parameter: nhận folder PDF, --save-images hoặc --image-directory; render page ở 400 DPI.
# Quan hệ: sử dụng OCR từ test_table_key_value_page3_plus.py.
# Input/output: đọc PDF hoặc ảnh; ghi ảnh, các cache và rule_induction.html.
# Contract: page 3+ là các page vật lý có số từ 3 trở đi.

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from html import escape
from pathlib import Path
from time import perf_counter

import cv2

from scripts.export_pages_3plus import DPI
from test_table_key_value_page3_plus import Page3PlusKeyValueExtractor


DETECTOR_CACHE_VERSION = "PP-OCRv6_medium_det-v1"
RECOGNITION_CACHE_VERSION = "page3-plus-recognition-medium-v1"
IMAGE_FILE_PATTERN = re.compile(
    r"^(?P<pdf_stem>.+)_page_(?P<physical_page>[1-9][0-9]*)\.(?:png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)


class CachedTextDetector:
    """Lưu và tái sử dụng polygon do text detector trả về."""

    def __init__(self, detector, cache_directory):
        self.detector = detector
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.last_cache_hit = False

    def predict(self, input, **parameters):
        """Trả kết quả detector từ cache hoặc chạy detector khi chưa có."""
        cache_key = hashlib.sha256()
        cache_key.update(DETECTOR_CACHE_VERSION.encode("utf-8"))
        cache_key.update(str(input.shape).encode("utf-8"))
        cache_key.update(str(input.dtype).encode("utf-8"))
        cache_key.update(input.tobytes())
        cache_key.update(
            json.dumps(
                parameters,
                sort_keys=True,
            ).encode("utf-8")
        )
        cache_path = self.cache_directory / f"{cache_key.hexdigest()}.json"

        if cache_path.is_file():
            self.last_cache_hit = True
            polygons = json.loads(cache_path.read_text(encoding="utf-8"))
            return [{"dt_polys": polygons}]

        self.last_cache_hit = False
        detection_results = self.detector.predict(
            input=input,
            **parameters,
        )
        polygons = [
            [
                [float(x_coordinate), float(y_coordinate)]
                for x_coordinate, y_coordinate in polygon
            ]
            for detection_result in detection_results
            for polygon in detection_result["dt_polys"]
        ]
        cache_path.write_text(
            json.dumps(polygons),
            encoding="utf-8",
        )
        return [{"dt_polys": polygons}]


def _render_pages_3plus(renderer, pdf_file, output_directory):
    """Render toàn bộ page vật lý từ page 3 và trả đường dẫn theo số thật."""
    output_pattern = output_directory / f".{pdf_file.stem}_render_%03d.png"
    rendered_pattern = f".{pdf_file.stem}_render_*.png"
    for rendered_path in output_directory.glob(rendered_pattern):
        rendered_path.unlink()

    command = [
        renderer,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        f"-r{DPI}",
        "-dFirstPage=3",
        f"-sOutputFile={output_pattern}",
        str(pdf_file),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Không thể render page 3+: {pdf_file.name}")

    rendered_pages = sorted(output_directory.glob(rendered_pattern))
    pages = []
    for page_index, rendered_path in enumerate(rendered_pages, start=3):
        output_path = output_directory / (
            f"{pdf_file.stem}_page_{page_index}.png"
        )
        rendered_path.replace(output_path)
        pages.append((page_index, output_path))
    return pages


def parse_arguments():
    """Đọc nguồn dữ liệu và chế độ chạy Rule Induction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_directory", type=Path, nargs="?")
    parser.add_argument("--image-directory", type=Path)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Chỉ render page 3+ ra folder images, chưa chạy OCR.",
    )
    arguments = parser.parse_args()
    if arguments.pdf_directory is None and arguments.image_directory is None:
        parser.error("Cần truyền folder PDF hoặc --image-directory.")
    if arguments.pdf_directory is not None and arguments.image_directory is not None:
        parser.error("Chỉ chọn folder PDF hoặc --image-directory.")
    if arguments.image_directory is not None and arguments.save_images:
        parser.error("--save-images chỉ dùng với folder PDF.")
    return arguments


def build_html(pages):
    """Tạo một HTML Rule Induction cho toàn bộ page đã OCR."""
    pages_json = json.dumps(pages, ensure_ascii=False).replace(
        "</",
        "<\\/",
    )
    first_page = pages[0]
    template = r"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rule Induction key-value</title>
    <style>
        :root {
            --xam: #64748b;
            --xanh-la: #15803d;
            --xanh-duong: #0369a1;
            --tim: #7c3aed;
            --hong: #c026d3;
            --cam: #ea580c;
            --nen: #f1f5f9;
            --chu: #172033;
        }
        * { box-sizing: border-box; }
        body {
            background: var(--nen);
            color: var(--chu);
            font-family: Arial, sans-serif;
            margin: 0;
        }
        header {
            background: linear-gradient(135deg, #0f172a, #1e3a5f);
            color: white;
            padding: 22px 28px;
        }
        header h1 { font-size: 24px; margin: 0 0 8px; }
        header p { color: #dbeafe; margin: 0; }
        .app { margin: 0 auto; max-width: 1800px; padding: 20px; }
        .control-panel {
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
            padding: 18px;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .page-navigation {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            margin-bottom: 14px;
        }
        .page-navigation label { font-weight: 700; }
        .page-navigation select {
            border: 1px solid #94a3b8;
            border-radius: 8px;
            font: inherit;
            min-width: 280px;
            padding: 9px 10px;
        }
        .page-position {
            color: #475569;
            font-size: 14px;
            margin-left: auto;
        }
        .steps { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
        .step {
            background: #e2e8f0;
            border-radius: 999px;
            color: #475569;
            font-size: 14px;
            font-weight: 700;
            padding: 8px 13px;
        }
        .step.active { background: #dbeafe; color: #1d4ed8; }
        .step.done { background: #dcfce7; color: #166534; }
        .selection-status {
            align-items: center;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
            padding: 9px 12px;
        }
        .selection-status span { color: #64748b; font-size: 13px; }
        .selection-status strong { color: #1d4ed8; font-size: 15px; }
        .instruction {
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            font-size: 17px;
            font-weight: 700;
            line-height: 1.5;
            margin-bottom: 12px;
            padding: 13px 15px;
        }
        .instruction.wait-value {
            background: #ecfdf5;
            border-color: var(--xanh-la);
            color: #14532d;
        }
        .instruction.confirmed {
            background: #f0fdf4;
            border-color: #16a34a;
            color: #14532d;
        }
        .hover-info {
            color: #475569;
            font-size: 14px;
            min-height: 22px;
        }
        .actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
        button {
            border: 0;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 700;
            padding: 10px 14px;
        }
        .actions button:disabled { cursor: not-allowed; opacity: 0.45; }
        .secondary { background: #e2e8f0; color: #334155; }
        .danger { background: #fee2e2; color: #991b1b; }
        .confirm { background: #16a34a; color: white; margin-left: auto; }
        .workspace {
            align-items: start;
            display: grid;
            gap: 18px;
            grid-template-columns: minmax(0, 1fr) 360px;
        }
        .card {
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06);
            padding: 15px;
        }
        .card h2 { font-size: 18px; margin: 0 0 12px; }
        .image-toolbar {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .recognition-summary {
            background: #eff6ff;
            border: 1px solid #93c5fd;
            border-radius: 8px;
            color: #1e3a8a;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 10px;
            padding: 9px 11px;
        }
        .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; }
        .legend-item { align-items: center; display: flex; gap: 5px; }
        .swatch { border: 2px solid; height: 13px; width: 22px; }
        .swatch.detected {
            background: rgba(100, 116, 139, 0.14);
            border-color: var(--xam);
        }
        .swatch.manual-key {
            background: rgba(21, 128, 61, 0.18);
            border-color: var(--xanh-la);
        }
        .swatch.manual-value {
            background: rgba(3, 105, 161, 0.18);
            border-color: var(--xanh-duong);
        }
        .swatch.predicted {
            background: rgba(124, 58, 237, 0.16);
            border-color: var(--tim);
        }
        .swatch.inline {
            background: rgba(234, 88, 12, 0.18);
            border-color: var(--cam);
        }
        .image-stage {
            background: #e5e7eb;
            border: 1px solid #94a3b8;
            line-height: 0;
            position: relative;
            user-select: none;
            width: 100%;
        }
        .image-stage img { display: block; height: auto; width: 100%; }
        .box-layer, .link-layer, .logical-layer {
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 100%;
        }
        .link-layer { pointer-events: none; z-index: 5; }
        .box-layer { z-index: 10; }
        .logical-layer { pointer-events: none; z-index: 20; }
        .ocr-box {
            background: rgba(100, 116, 139, 0.10);
            border: 2px solid rgba(71, 85, 105, 0.82);
            cursor: pointer;
            min-height: 5px;
            min-width: 5px;
            position: absolute;
            transition: background 100ms, border-color 100ms, box-shadow 100ms;
        }
        .ocr-box:hover {
            background: rgba(250, 204, 21, 0.22);
            border-color: #ca8a04;
            box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.40);
        }
        .ocr-box.hidden-unused { display: none; }
        .ocr-box.popover-target {
            box-shadow: 0 0 0 4px rgba(250, 204, 21, 0.60);
        }
        .ocr-box.ocr-box.current-key {
            animation: pulse 1.1s infinite;
            background: rgba(21, 128, 61, 0.25);
            border-color: var(--xanh-la);
            border-style: solid;
            border-width: 3px;
            box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.30);
        }
        .ocr-box.manual-key {
            background: rgba(21, 128, 61, 0.22);
            border-color: var(--xanh-la);
            border-width: 3px;
        }
        .ocr-box.manual-value {
            background: rgba(3, 105, 161, 0.22);
            border-color: var(--xanh-duong);
            border-width: 3px;
        }
        .ocr-box.manual-inline {
            background: rgba(234, 88, 12, 0.22);
            border-color: var(--cam);
            border-width: 4px;
        }
        .ocr-box.predicted-key {
            background: rgba(124, 58, 237, 0.17);
            border-color: var(--tim);
            border-style: dashed;
        }
        .ocr-box.predicted-value {
            background: rgba(192, 38, 211, 0.17);
            border-color: var(--hong);
            border-style: dashed;
        }
        .ocr-box.predicted-inline {
            background: rgba(217, 119, 6, 0.17);
            border-color: #d97706;
            border-style: dashed;
        }
        .logical-box {
            border: 4px solid;
            position: absolute;
        }
        .logical-box span {
            color: white;
            font-size: 12px;
            font-weight: 700;
            left: -4px;
            line-height: 1.2;
            padding: 3px 6px;
            position: absolute;
            top: -22px;
            white-space: nowrap;
        }
        .logical-box.manual-key { border-color: var(--xanh-la); }
        .logical-box.manual-key span { background: var(--xanh-la); }
        .logical-box.manual-value { border-color: var(--xanh-duong); }
        .logical-box.manual-value span { background: var(--xanh-duong); }
        .logical-box.predicted-key {
            border-color: var(--tim);
            border-style: dashed;
        }
        .logical-box.predicted-key span { background: var(--tim); }
        .logical-box.predicted-value {
            border-color: var(--hong);
            border-style: dashed;
        }
        .logical-box.predicted-value span { background: var(--hong); }
        .pair-popover {
            background: white;
            border: 1px solid #94a3b8;
            border-radius: 9px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.24);
            display: flex;
            gap: 7px;
            line-height: normal;
            padding: 7px;
            position: absolute;
            transform: translate(-50%, calc(-100% - 8px));
            z-index: 40;
        }
        .pair-popover[hidden] { display: none; }
        @keyframes pulse {
            50% { box-shadow: 0 0 0 6px rgba(21, 128, 61, 0.10); }
        }
        .side-panel { display: grid; gap: 18px; position: sticky; top: 190px; }
        .empty {
            background: #f8fafc;
            border: 1px dashed #94a3b8;
            border-radius: 8px;
            color: #64748b;
            line-height: 1.45;
            padding: 12px;
        }
        .list { display: grid; gap: 8px; max-height: 330px; overflow: auto; }
        .pair-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.45;
            padding: 9px;
        }
        .pair-item strong { color: #0f172a; }
        .pair-arrow { color: var(--cam); font-weight: 700; padding: 0 5px; }
        .pair-status {
            color: #9a3412;
            font-size: 12px;
            font-weight: 700;
            margin-top: 6px;
        }
        .recognition-confidence {
            color: #1d4ed8;
            font-size: 12px;
            margin-top: 6px;
        }
        .remove-pair {
            background: #fee2e2;
            color: #991b1b;
            margin-top: 8px;
            padding: 6px 9px;
        }
        .rule-summary {
            background: #faf5ff;
            border: 1px solid #ddd6fe;
            border-radius: 8px;
            color: #4c1d95;
            font-size: 13px;
            line-height: 1.5;
            margin-top: 10px;
            padding: 10px;
        }
        .result-section { margin-top: 20px; }
        .result-section[hidden] { display: none; }
        table { border-collapse: collapse; width: 100%; }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }
        th { background: #e2e8f0; }
        .source-manual { color: var(--xanh-la); font-weight: 700; }
        .source-rule { color: var(--tim); font-weight: 700; }
        dialog {
            border: 0;
            border-radius: 14px;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.30);
            max-width: 620px;
            padding: 22px;
            width: calc(100% - 32px);
        }
        dialog::backdrop { background: rgba(15, 23, 42, 0.60); }
        dialog h2 { margin: 0 0 8px; }
        dialog p { color: #475569; line-height: 1.5; }
        .inline-original {
            background: #f1f5f9;
            border-radius: 8px;
            line-height: 1.5;
            padding: 10px;
        }
        .inline-form { display: grid; gap: 12px; }
        .inline-form label { display: grid; font-weight: 700; gap: 6px; }
        .inline-form input {
            border: 1px solid #94a3b8;
            border-radius: 7px;
            font: inherit;
            padding: 9px 10px;
        }
        .inline-error { color: #b91c1c; font-weight: 700; min-height: 20px; }
        .inline-dialog-actions {
            display: flex;
            gap: 8px;
            justify-content: flex-end;
        }
        @media (max-width: 1100px) {
            .workspace { grid-template-columns: 1fr; }
            .side-panel { position: static; }
            .confirm { margin-left: 0; }
        }
    </style>
</head>
<body>
<header>
    <h1>Rule Induction key-value</h1>
    <p>
        Chọn ví dụ trực tiếp trên ảnh và xem chương trình suy luận lại
        ngay sau mỗi cặp.
    </p>
</header>
<main class="app">
    <section class="control-panel">
        <div class="page-navigation">
            <label for="pageSelect">Phiếu đang kiểm tra</label>
            <button class="secondary" id="previousPage" type="button">
                Phiếu trước
            </button>
            <select id="pageSelect"></select>
            <button class="secondary" id="nextPage" type="button">
                Phiếu sau
            </button>
            <span class="page-position" id="pagePosition"></span>
        </div>
        <div class="steps">
            <span class="step active" id="stepKey">1. Chọn key</span>
            <span class="step" id="stepValue">2. Chọn value</span>
            <span class="step" id="stepPreview">3. Kiểm tra preview</span>
            <span class="step" id="stepConfirm">4. Xác nhận</span>
        </div>
        <div class="selection-status" role="status" aria-live="polite">
            <span>Trạng thái hiện tại:</span>
            <strong id="selectionStatus">Đang chờ chọn KEY đầu tiên</strong>
        </div>
        <div class="instruction" id="instruction" role="status" aria-live="polite">
            Bước 1: Hãy bấm vào bbox chứa key đầu tiên trên ảnh.
        </div>
        <div class="hover-info" id="hoverInfo">
            Di chuột lên bbox để xem nội dung OCR.
        </div>
        <div class="actions">
            <button class="secondary" id="cancelKey" disabled>
                Hủy key đang chọn
            </button>
            <button class="secondary" id="inlineCurrentKey" disabled>
                Key-value cùng bbox
            </button>
            <button class="secondary" id="undoExample" disabled>
                Xóa mẫu gần nhất
            </button>
            <button class="secondary" id="undoSuppression" disabled>
                Hoàn tác cặp đã xóa
            </button>
            <button class="danger" id="resetAll" disabled>
                Làm lại từ đầu
            </button>
            <button class="confirm" id="confirmResult" disabled>
                Xác nhận kết quả
            </button>
        </div>
    </section>

    <dialog id="inlineDialog">
        <h2>Tách key-value trong cùng bbox</h2>
        <p>
            Chương trình đã thử tách bằng ký tự đặc biệt. Bạn có thể sửa
            separator, key hoặc value trước khi xác nhận.
        </p>
        <p class="inline-original" id="inlineOriginal"></p>
        <div class="inline-form">
            <label>
                Ký tự phân tách
                <input id="inlineSeparator" autocomplete="off">
            </label>
            <label>
                Key
                <input id="inlineKey" autocomplete="off">
            </label>
            <label>
                Value
                <input id="inlineValue" autocomplete="off">
            </label>
            <div class="inline-error" id="inlineError" role="alert"></div>
            <div class="inline-dialog-actions">
                <button class="secondary" id="cancelInline" type="button">
                    Hủy
                </button>
                <button class="confirm" id="confirmInline" type="button">
                    Xác nhận tách
                </button>
            </div>
        </div>
    </dialog>

    <section class="workspace">
        <div class="card">
            <div class="image-toolbar">
                <h2>Ảnh và các bbox OCR</h2>
                <div class="legend">
                    <span class="legend-item">
                        <span class="swatch detected"></span>Đã OCR
                    </span>
                    <span class="legend-item">
                        <span class="swatch manual-key"></span>Key đã chọn
                    </span>
                    <span class="legend-item">
                        <span class="swatch manual-value"></span>Value đã chọn
                    </span>
                    <span class="legend-item">
                        <span class="swatch predicted"></span>Preview suy luận
                    </span>
                    <span class="legend-item">
                        <span class="swatch inline"></span>Inline key=value
                    </span>
                </div>
                <button class="secondary" id="toggleUnusedRegions" disabled>
                    Ẩn bbox xám
                </button>
            </div>
            <div class="image-stage" id="imageStage">
                <img id="pageImage" src="__IMAGE_NAME__" alt="Ảnh page cần phân tích">
                <svg class="link-layer" id="linkLayer"
                     viewBox="0 0 __IMAGE_WIDTH__ __IMAGE_HEIGHT__"
                     preserveAspectRatio="none"></svg>
                <div class="box-layer" id="boxLayer"></div>
                <div class="logical-layer" id="logicalLayer"></div>
                <div
                    class="pair-popover"
                    id="pairPopover"
                    role="toolbar"
                    aria-label="Hành động với cặp preview"
                    hidden
                >
                    <button class="secondary" id="editPrediction" type="button">
                        Sửa
                    </button>
                    <button class="danger" id="deletePrediction" type="button">
                        Xóa
                    </button>
                </div>
            </div>
        </div>

        <aside class="side-panel">
            <section class="card">
                <h2>Mẫu người dùng đã chọn (<span id="exampleCount">0</span>)</h2>
                <div class="empty" id="exampleEmpty">
                    Chưa có mẫu. Bấm một bbox key, sau đó bấm bbox value tương ứng.
                </div>
                <div class="list" id="exampleList"></div>
            </section>

            <section class="card">
                <h2>Preview suy luận (<span id="previewCount">0</span>)</h2>
                <div class="recognition-summary" id="recognitionSummary"></div>
                <div class="empty" id="previewEmpty">
                    Preview sẽ xuất hiện ngay sau khi bạn chọn xong cặp đầu tiên.
                </div>
                <div class="list" id="previewList"></div>
                <div class="rule-summary" id="ruleSummary">
                    Chưa có rule. Số mẫu không bị giới hạn.
                </div>
                <div class="rule-summary" id="suppressionSummary" hidden></div>
            </section>
        </aside>
    </section>

    <section class="card result-section" id="resultSection" hidden>
        <h2>Kết quả key-value đã xác nhận</h2>
        <p id="resultSummary"></p>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Key</th><th>Value</th>
                    <th>Nguồn</th><th>Độ tin cậy rule</th>
                </tr>
            </thead>
            <tbody id="resultBody"></tbody>
        </table>
    </section>
</main>

<script>
    const pages = __PAGES_JSON__;
    let currentPageIndex = 0;
    let PAGE_WIDTH = pages[0].width;
    let PAGE_HEIGHT = pages[0].height;
    let regions = pages[0].regions;
    let regionById = new Map(regions.map(region => [region.id, region]));

    const state = {
        currentKeyId: null,
        examples: [],
        predictions: [],
        rule: null,
        inlineRule: null,
        inlineRegionId: null,
        confirmed: false,
        autoOtherIds: new Set(),
        suppressedPairIds: new Set(),
        suppressionHistory: [],
        hideUnusedRegions: false,
        popoverPair: null,
        editingPrediction: null,
    };

    const boxLayer = document.getElementById("boxLayer");
    const linkLayer = document.getElementById("linkLayer");
    const logicalLayer = document.getElementById("logicalLayer");
    const pageImage = document.getElementById("pageImage");
    const pageSelect = document.getElementById("pageSelect");
    const previousPageButton = document.getElementById("previousPage");
    const nextPageButton = document.getElementById("nextPage");
    const pagePosition = document.getElementById("pagePosition");
    const recognitionSummary = document.getElementById("recognitionSummary");
    const instruction = document.getElementById("instruction");
    const selectionStatus = document.getElementById("selectionStatus");
    const hoverInfo = document.getElementById("hoverInfo");
    const cancelKeyButton = document.getElementById("cancelKey");
    const inlineCurrentKeyButton = document.getElementById("inlineCurrentKey");
    const undoExampleButton = document.getElementById("undoExample");
    const undoSuppressionButton = document.getElementById("undoSuppression");
    const resetAllButton = document.getElementById("resetAll");
    const confirmResultButton = document.getElementById("confirmResult");
    const toggleUnusedRegionsButton = document.getElementById(
        "toggleUnusedRegions",
    );
    const pairPopover = document.getElementById("pairPopover");
    const editPredictionButton = document.getElementById("editPrediction");
    const deletePredictionButton = document.getElementById("deletePrediction");
    const resultSection = document.getElementById("resultSection");
    const inlineDialog = document.getElementById("inlineDialog");
    const inlineOriginal = document.getElementById("inlineOriginal");
    const inlineSeparator = document.getElementById("inlineSeparator");
    const inlineKey = document.getElementById("inlineKey");
    const inlineValue = document.getElementById("inlineValue");
    const inlineError = document.getElementById("inlineError");
    const cancelInlineButton = document.getElementById("cancelInline");
    const confirmInlineButton = document.getElementById("confirmInline");
    const automaticInlineSeparators = ["=>", "→", "=", ":"];

    function centerX(region) {
        return (region.box[0] + region.box[2]) / 2;
    }

    function centerY(region) {
        return (region.box[1] + region.box[3]) / 2;
    }

    function regionHeight(region) {
        return Math.max(region.box[3] - region.box[1], 1);
    }

    function verticalOverlap(keyRegion, valueRegion) {
        const overlap = Math.max(
            0,
            Math.min(keyRegion.box[3], valueRegion.box[3])
                - Math.max(keyRegion.box[1], valueRegion.box[1]),
        );
        return overlap / Math.min(
            regionHeight(keyRegion),
            regionHeight(valueRegion),
        );
    }

    function median(values) {
        const ordered = [...values].sort((left, right) => left - right);
        const middle = Math.floor(ordered.length / 2);
        if (ordered.length % 2 === 1) {
            return ordered[middle];
        }
        return (ordered[middle - 1] + ordered[middle]) / 2;
    }

    function recognitionStatistics(sourceRegions) {
        const scores = sourceRegions.map(region => region.score);
        if (scores.length === 0) {
            return null;
        }
        return {
            count: scores.length,
            average: scores.reduce((sum, score) => sum + score, 0)
                / scores.length,
            median: median(scores),
            minimum: Math.min(...scores),
            belowEighty: scores.filter(score => score < 0.80).length,
        };
    }

    function formatRecognitionStatistics(label, statistics) {
        if (!statistics) {
            return `${label}: không có vùng text`;
        }
        return (
            `${label}: ${statistics.count} vùng · `
            + `trung bình ${(statistics.average * 100).toFixed(1)}% · `
            + `trung vị ${(statistics.median * 100).toFixed(1)}% · `
            + `thấp nhất ${(statistics.minimum * 100).toFixed(1)}% · `
            + `dưới 80%: ${statistics.belowEighty}/${statistics.count}`
        );
    }

    function updateRecognitionSummary() {
        const pageStatistics = recognitionStatistics(regions);
        const allRegions = pages.flatMap(page => page.regions);
        const overallStatistics = recognitionStatistics(allRegions);
        recognitionSummary.textContent = [
            "Confidence PP-OCR trước hiệu chỉnh VietOCR (rec_score)",
            formatRecognitionStatistics("Phiếu hiện tại", pageStatistics),
            formatRecognitionStatistics("Toàn bộ folder", overallStatistics),
        ].join(" | ");
    }

    function maxDeviation(values, expected) {
        return Math.max(...values.map(value => Math.abs(value - expected)));
    }

    function textType(text) {
        const normalized = text.trim();
        if (/^(Y|N|ON|OFF|YES|NO|ENABLED|DISABLED|ENABLE|DISABLE)$/i.test(normalized)) {
            return "lựa chọn";
        }
        if (/^[+-]?[0-9.,]+\s*[A-Za-z%°Ω/'-]+$/u.test(normalized)) {
            return "số kèm đơn vị";
        }
        if (/^[+-]?[0-9.,]+$/u.test(normalized)) {
            return "số";
        }
        if (/^(?=.*[0-9])[A-Z0-9_.<>/+-]{2,16}$/u.test(normalized)) {
            return "mã";
        }
        return "văn bản";
    }

    function combineRegions(parts) {
        const ordered = [...parts].sort((left, right) => (
            centerY(left) - centerY(right) || centerX(left) - centerX(right)
        ));
        const regionIds = ordered.map(region => region.id);
        return {
            id: regionIds.join("+"),
            regionIds,
            text: ordered.map(region => region.text).join(" "),
            box: [
                Math.min(...ordered.map(region => region.box[0])),
                Math.min(...ordered.map(region => region.box[1])),
                Math.max(...ordered.map(region => region.box[2])),
                Math.max(...ordered.map(region => region.box[3])),
            ],
            score: ordered.reduce((sum, region) => sum + region.score, 0)
                / ordered.length,
        };
    }

    function generateBlocks(columnRegions) {
        const ordered = [...columnRegions].sort((left, right) => (
            centerY(left) - centerY(right) || centerX(left) - centerX(right)
        ));
        const centerGaps = [];
        for (let index = 1; index < ordered.length; index += 1) {
            const gap = centerY(ordered[index]) - centerY(ordered[index - 1]);
            if (gap > 0) {
                centerGaps.push(gap);
            }
        }
        const typicalRowPitch = centerGaps.length >= 3
            ? median(centerGaps)
            : Number.POSITIVE_INFINITY;
        const blocks = [];
        for (let start = 0; start < ordered.length; start += 1) {
            const parts = [];
            for (let end = start; end < ordered.length; end += 1) {
                if (end > start) {
                    const previous = ordered[end - 1];
                    const current = ordered[end];
                    const verticalGap = current.box[1] - previous.box[3];
                    const centerGap = centerY(current) - centerY(previous);
                    const allowedGap = Math.max(
                        regionHeight(previous),
                        regionHeight(current),
                    ) * 0.55;
                    if (
                        verticalGap > allowedGap
                        || centerGap >= typicalRowPitch * 0.85
                    ) {
                        break;
                    }
                }
                parts.push(ordered[end]);
                blocks.push(combineRegions(parts));
            }
        }
        return blocks;
    }

    function blockContains(block, regionId) {
        return block.regionIds.includes(regionId);
    }

    function blockUsesAny(block, regionIds) {
        return block.regionIds.some(regionId => regionIds.has(regionId));
    }

    function pairY(pair) {
        return (centerY(pair.key) + centerY(pair.value)) / 2;
    }

    function pairInstanceId(pair) {
        return `${pair.key.regionIds.join("+")}→${pair.value.regionIds.join("+")}`;
    }

    function normalizePairText(text) {
        return text
            .normalize("NFKC")
            .toLocaleLowerCase("vi")
            .replace(/\s+/gu, " ")
            .replace(/\s*([:;=,])\s*/gu, "$1")
            .trim();
    }

    function pairSignature(pair) {
        return `${normalizePairText(pair.key.text)}→${normalizePairText(pair.value.text)}`;
    }

    function hasHeaderStructure(pair) {
        const pairRegionIds = new Set([
            ...pair.key.regionIds,
            ...pair.value.regionIds,
        ]);
        const averageHeight = (
            regionHeight(pair.key) + regionHeight(pair.value)
        ) / 2;
        const currentY = pairY(pair);
        const previousRegions = regions
            .filter(region => (
                !pairRegionIds.has(region.id)
                && centerY(region) < currentY - averageHeight * 0.75
            ))
            .sort((left, right) => centerY(right) - centerY(left));
        if (previousRegions.length === 0) return false;
        const nearestPrevious = previousRegions[0];
        const isWide = (
            nearestPrevious.box[2] - nearestPrevious.box[0]
        ) / PAGE_WIDTH >= 0.30;
        return (
            isWide
            && currentY - centerY(nearestPrevious) <= averageHeight * 3
        );
    }

    function pairFeatures(keyRegion, valueRegion) {
        const averageHeight = (
            regionHeight(keyRegion) + regionHeight(valueRegion)
        ) / 2;
        return {
            keyX: keyRegion.box[0] / PAGE_WIDTH,
            valueX: valueRegion.box[0] / PAGE_WIDTH,
            deltaX: (valueRegion.box[0] - keyRegion.box[0]) / PAGE_WIDTH,
            rowOffset: Math.abs(
                centerY(valueRegion) - centerY(keyRegion),
            ) / Math.max(averageHeight, 1),
            overlap: verticalOverlap(keyRegion, valueRegion),
            keyType: textType(keyRegion.text),
            valueType: textType(valueRegion.text),
        };
    }

    function buildExample(keyId, valueId) {
        const keyAnchor = regionById.get(keyId);
        const valueAnchor = regionById.get(valueId);
        const usedRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        const keyColumn = regions.filter(region => (
            !usedRegionIds.has(region.id)
            && Math.abs(region.box[0] - keyAnchor.box[0]) / PAGE_WIDTH <= 0.055
        ));
        const valueColumn = regions.filter(region => (
            !usedRegionIds.has(region.id)
            && Math.abs(region.box[0] - valueAnchor.box[0]) / PAGE_WIDTH <= 0.055
        ));
        const keyBlocks = generateBlocks(keyColumn).filter(block => (
            blockContains(block, keyId)
        ));
        const valueBlocks = generateBlocks(valueColumn).filter(block => (
            blockContains(block, valueId)
        ));
        const direction = valueAnchor.box[0] >= keyAnchor.box[0] ? 1 : -1;
        let bestExample = null;
        let bestCost = Number.POSITIVE_INFINITY;

        for (const key of keyBlocks) {
            for (const value of valueBlocks) {
                const features = pairFeatures(key, value);
                if (direction * features.deltaX <= 0) {
                    continue;
                }
                const extraLines = (
                    key.regionIds.length + value.regionIds.length - 2
                );
                const cost = (
                    features.rowOffset * 0.55
                    + (1 - features.overlap) * 0.35
                    + extraLines * 0.05
                );
                if (cost < bestCost) {
                    bestCost = cost;
                    bestExample = {key, value, mode: "spatial"};
                }
            }
        }
        return bestExample;
    }

    function virtualInlineBlock(region, text, role) {
        return {
            id: `${region.id}:${role}`,
            regionIds: [region.id],
            text,
            box: [...region.box],
            score: region.score,
        };
    }

    function separatorIsTopLevel(text, separatorIndex) {
        let depth = 0;
        const opening = "([{";
        const closing = ")]}";
        for (let index = 0; index < separatorIndex; index += 1) {
            if (opening.includes(text[index])) {
                depth += 1;
            } else if (closing.includes(text[index])) {
                depth = Math.max(0, depth - 1);
            }
        }
        return depth === 0;
    }

    function isCompactInlineKey(key) {
        const words = key.trim().split(/\s+/u);
        return (
            words.length <= 4
            && !/[()[\]{}]/u.test(key)
            && /[\p{L}\p{N}]/u.test(key)
        );
    }

    function splitInlineText(text, separators, automatic = false) {
        for (const separator of separators) {
            if (!separator) {
                continue;
            }
            const separatorIndex = text.indexOf(separator);
            if (separatorIndex <= 0) {
                continue;
            }
            const key = text.slice(0, separatorIndex).trim();
            const value = text.slice(
                separatorIndex + separator.length,
            ).trim();
            if (
                automatic
                && (
                    !separatorIsTopLevel(text, separatorIndex)
                    || !isCompactInlineKey(key)
                )
            ) {
                continue;
            }
            if (key && value) {
                return {key, value, separator};
            }
        }
        return null;
    }

    function buildInlineExample(regionId, keyText, valueText, separator) {
        const region = regionById.get(regionId);
        return {
            key: virtualInlineBlock(region, keyText, "key"),
            value: virtualInlineBlock(region, valueText, "value"),
            mode: "inline",
            separator,
        };
    }

    function inferInlineRule() {
        const inlineExamples = state.examples.filter(example => (
            example.mode === "inline"
        ));
        if (inlineExamples.length === 0) {
            return null;
        }
        return {
            separators: [...new Set(inlineExamples
                .map(example => example.separator)
                .filter(separator => separator))],
            startY: pairY(inlineExamples[0]),
            sampleCount: inlineExamples.length,
        };
    }

    function predictInlinePairs(rule, occupiedRegionIds) {
        if (!rule || rule.separators.length === 0) {
            return [];
        }
        const manualRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        const predictions = [];
        for (const region of regions) {
            if (
                manualRegionIds.has(region.id)
                || occupiedRegionIds.has(region.id)
                || centerY(region) < rule.startY
            ) {
                continue;
            }
            const split = splitInlineText(
                region.text,
                rule.separators,
                true,
            );
            if (!split) {
                continue;
            }
            const pair = {
                key: virtualInlineBlock(region, split.key, "key"),
                value: virtualInlineBlock(region, split.value, "value"),
                mode: "inline",
                separator: split.separator,
                score: region.score,
            };
            if (!state.suppressedPairIds.has(pairInstanceId(pair))) {
                predictions.push(pair);
                occupiedRegionIds.add(region.id);
            }
        }
        return predictions;
    }

    function inferRule() {
        const spatialExamples = state.examples.filter(example => (
            example.mode !== "inline"
        ));
        if (spatialExamples.length === 0) {
            return null;
        }
        const features = spatialExamples.map(example => pairFeatures(
            example.key,
            example.value,
        ));
        const keyXs = features.map(feature => feature.keyX);
        const valueXs = features.map(feature => feature.valueX);
        const deltaXs = features.map(feature => feature.deltaX);
        const keyX = median(keyXs);
        const valueX = median(valueXs);
        const deltaX = median(deltaXs);
        const firstExample = spatialExamples[0];
        const startY = (
            centerY(firstExample.key) + centerY(firstExample.value)
        ) / 2;
        const sampleCount = features.length;
        const baseColumnTolerance = sampleCount === 1
            ? 0.06
            : sampleCount === 2
                ? 0.04
                : 0.03;
        const keyTolerance = Math.max(
            baseColumnTolerance,
            maxDeviation(keyXs, keyX) * 1.5 + 0.015,
        );
        const valueTolerance = Math.max(
            baseColumnTolerance,
            maxDeviation(valueXs, valueX) * 1.5 + 0.015,
        );
        const deltaTolerance = Math.max(
            baseColumnTolerance * 1.5,
            maxDeviation(deltaXs, deltaX) * 1.5 + 0.025,
        );
        return {
            keyX,
            valueX,
            deltaX,
            keyTolerance,
            valueTolerance,
            deltaTolerance,
            direction: deltaX >= 0 ? 1 : -1,
            rowTolerance: Math.max(
                1.15,
                Math.max(...features.map(feature => feature.rowOffset)) + 0.4,
            ),
            minimumOverlap: Math.max(
                0,
                Math.min(...features.map(feature => feature.overlap)) - 0.25,
            ),
            keyTypes: new Set(features.map(feature => feature.keyType)),
            valueTypes: new Set(features.map(feature => feature.valueType)),
            startY,
            sampleCount,
        };
    }

    function similarity(value, expected, tolerance) {
        return Math.max(0, 1 - Math.abs(value - expected) / tolerance);
    }

    function predictPairs(rule, includeAbove = false) {
        const candidates = [];
        const manualRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        const keyRegions = regions.filter(region => (
            !manualRegionIds.has(region.id)
            && Math.abs(region.box[0] / PAGE_WIDTH - rule.keyX)
                <= rule.keyTolerance
        ));
        const valueRegions = regions.filter(region => (
            !manualRegionIds.has(region.id)
            && Math.abs(region.box[0] / PAGE_WIDTH - rule.valueX)
                <= rule.valueTolerance
        ));
        const keyBlocks = generateBlocks(keyRegions);
        const valueBlocks = generateBlocks(valueRegions);

        for (const keyRegion of keyBlocks) {
            for (const valueRegion of valueBlocks) {
                const keyIds = new Set(keyRegion.regionIds);
                if (valueRegion.regionIds.some(regionId => keyIds.has(regionId))) {
                    continue;
                }
                const valueIds = new Set(valueRegion.regionIds);
                const crossesOtherKey = keyRegions.some(region => (
                    !keyIds.has(region.id)
                    && centerY(region) >= valueRegion.box[1]
                    && centerY(region) <= valueRegion.box[3]
                ));
                const crossesOtherValue = valueRegions.some(region => (
                    !valueIds.has(region.id)
                    && centerY(region) >= keyRegion.box[1]
                    && centerY(region) <= keyRegion.box[3]
                ));
                if (crossesOtherKey || crossesOtherValue) {
                    continue;
                }
                const features = pairFeatures(keyRegion, valueRegion);
                const candidateY = (
                    centerY(keyRegion) + centerY(valueRegion)
                ) / 2;
                if (!includeAbove && candidateY < rule.startY) {
                    continue;
                }
                if (rule.direction * features.deltaX <= 0) {
                    continue;
                }
                if (Math.abs(features.keyX - rule.keyX) > rule.keyTolerance) {
                    continue;
                }
                if (Math.abs(features.valueX - rule.valueX) > rule.valueTolerance) {
                    continue;
                }
                if (Math.abs(features.deltaX - rule.deltaX) > rule.deltaTolerance) {
                    continue;
                }
                if (features.rowOffset > rule.rowTolerance) {
                    continue;
                }
                if (features.overlap < rule.minimumOverlap) {
                    continue;
                }

                const keyColumnScore = similarity(
                    features.keyX,
                    rule.keyX,
                    rule.keyTolerance,
                );
                const valueColumnScore = similarity(
                    features.valueX,
                    rule.valueX,
                    rule.valueTolerance,
                );
                const distanceScore = similarity(
                    features.deltaX,
                    rule.deltaX,
                    rule.deltaTolerance,
                );
                const rowScore = Math.max(
                    0,
                    1 - features.rowOffset / rule.rowTolerance,
                );
                const typeScore = (
                    (rule.keyTypes.has(features.keyType) ? 0.5 : 0.25)
                    + (rule.valueTypes.has(features.valueType) ? 0.5 : 0.25)
                );
                const groupPenalty = (
                    keyRegion.regionIds.length
                    + valueRegion.regionIds.length
                    - 2
                ) * 0.015;
                const score = (
                    keyColumnScore * 0.20
                    + valueColumnScore * 0.20
                    + distanceScore * 0.16
                    + rowScore * 0.19
                    + features.overlap * 0.15
                    + typeScore * 0.10
                    - groupPenalty
                );
                if (score >= 0.58) {
                    candidates.push({
                        key: keyRegion,
                        value: valueRegion,
                        score,
                    });
                }
            }
        }

        candidates.sort((left, right) => right.score - left.score);
        const usedRegionIds = new Set(manualRegionIds);
        const predictions = [];
        for (const candidate of candidates) {
            if (
                blockUsesAny(candidate.key, usedRegionIds)
                || blockUsesAny(candidate.value, usedRegionIds)
            ) {
                continue;
            }
            predictions.push(candidate);
            for (const regionId of candidate.key.regionIds) {
                usedRegionIds.add(regionId);
            }
            for (const regionId of candidate.value.regionIds) {
                usedRegionIds.add(regionId);
            }
        }
        return predictions;
    }

    function classifyPredictions(rule) {
        const allPredictions = predictPairs(rule, true);
        const predictionsAbove = allPredictions
            .filter(pair => pairY(pair) < rule.startY)
            .sort((left, right) => pairY(right) - pairY(left));
        const headerProbe = predictionsAbove[0] || null;
        const predictionsBelow = allPredictions.filter(pair => (
            pairY(pair) >= rule.startY
        ));
        const pairsBySignature = new Map();
        for (const pair of predictionsBelow) {
            const signature = pairSignature(pair);
            if (!pairsBySignature.has(signature)) {
                pairsBySignature.set(signature, []);
            }
            pairsBySignature.get(signature).push(pair);
        }
        const automaticHeaderSignatures = new Set();
        if (headerProbe) automaticHeaderSignatures.add(pairSignature(headerProbe));
        for (const [signature, pairs] of pairsBySignature.entries()) {
            if (pairs.length >= 2 && pairs.some(hasHeaderStructure)) {
                automaticHeaderSignatures.add(signature);
            }
        }
        const autoOtherPairs = [
            ...(headerProbe ? [headerProbe] : []),
            ...predictionsBelow.filter(pair => (
                automaticHeaderSignatures.has(pairSignature(pair))
            )),
        ];
        state.autoOtherIds = new Set(autoOtherPairs.flatMap(allPairIds));
        state.predictions = predictionsBelow.filter(pair => (
            !automaticHeaderSignatures.has(pairSignature(pair))
            && !state.suppressedPairIds.has(pairInstanceId(pair))
        ));
    }

    function renderBoxes() {
        boxLayer.replaceChildren();
        for (const region of regions) {
            const box = document.createElement("button");
            const width = region.box[2] - region.box[0];
            const height = region.box[3] - region.box[1];
            box.type = "button";
            box.className = "ocr-box";
            box.dataset.regionId = region.id;
            box.style.left = `${region.box[0] / PAGE_WIDTH * 100}%`;
            box.style.top = `${region.box[1] / PAGE_HEIGHT * 100}%`;
            box.style.width = `${width / PAGE_WIDTH * 100}%`;
            box.style.height = `${height / PAGE_HEIGHT * 100}%`;
            box.style.zIndex = String(
                Math.round(1000000 / Math.max(width * height, 1))
            );
            box.title = region.text;
            box.addEventListener("click", (event) => {
                event.stopPropagation();
                selectRegion(region.id);
            });
            box.addEventListener("mouseenter", () => {
                hoverInfo.textContent = `OCR: “${region.text}”`;
            });
            box.addEventListener("mouseleave", () => {
                hoverInfo.textContent = "Di chuột lên bbox để xem nội dung OCR.";
            });
            boxLayer.appendChild(box);
        }
        refreshBoxStyles();
    }

    function predictionForRegion(regionId) {
        return state.predictions.find(pair => (
            pair.key.regionIds.includes(regionId)
            || pair.value.regionIds.includes(regionId)
        ));
    }

    function closePredictionPopover() {
        pairPopover.hidden = true;
        state.popoverPair = null;
        refreshBoxStyles();
    }

    function openPredictionPopover(pair) {
        state.popoverPair = pair;
        const left = Math.min(pair.key.box[0], pair.value.box[0]);
        const right = Math.max(pair.key.box[2], pair.value.box[2]);
        const top = Math.min(pair.key.box[1], pair.value.box[1]);
        const center = Math.min(Math.max((left + right) / 2, 0), PAGE_WIDTH);
        pairPopover.style.left = `${center / PAGE_WIDTH * 100}%`;
        pairPopover.style.top = `${top / PAGE_HEIGHT * 100}%`;
        pairPopover.hidden = false;
        refreshBoxStyles();
    }

    function appendLogicalBox(block, cssClass, labelText) {
        if (block.regionIds.length === 1) {
            return;
        }
        const box = document.createElement("div");
        box.className = `logical-box ${cssClass}`;
        box.style.left = `${block.box[0] / PAGE_WIDTH * 100}%`;
        box.style.top = `${block.box[1] / PAGE_HEIGHT * 100}%`;
        box.style.width = `${(block.box[2] - block.box[0]) / PAGE_WIDTH * 100}%`;
        box.style.height = `${(block.box[3] - block.box[1]) / PAGE_HEIGHT * 100}%`;
        const label = document.createElement("span");
        label.textContent = `${labelText} · ${block.regionIds.length} bbox`;
        box.appendChild(label);
        logicalLayer.appendChild(box);
    }

    function renderLogicalBlocks() {
        logicalLayer.replaceChildren();
        for (const example of state.examples) {
            if (example.mode === "inline") {
                continue;
            } else {
                appendLogicalBox(example.key, "manual-key", "KEY gộp");
                appendLogicalBox(example.value, "manual-value", "VALUE gộp");
            }
        }
        for (const prediction of state.predictions) {
            if (prediction.mode === "inline") {
                continue;
            } else {
                appendLogicalBox(prediction.key, "predicted-key", "key? gộp");
                appendLogicalBox(
                    prediction.value,
                    "predicted-value",
                    "value? gộp",
                );
            }
        }
    }

    function refreshBoxStyles() {
        const manualKeys = new Set(state.examples.flatMap(example => (
            example.key.regionIds
        )));
        const manualValues = new Set(state.examples.flatMap(example => (
            example.value.regionIds
        )));
        const predictedKeys = new Set(state.predictions.flatMap(pair => (
            pair.key.regionIds
        )));
        const predictedValues = new Set(state.predictions.flatMap(pair => (
            pair.value.regionIds
        )));
        const manualInline = new Set(state.examples
            .filter(example => example.mode === "inline")
            .flatMap(example => example.key.regionIds));
        const predictedInline = new Set(state.predictions
            .filter(pair => pair.mode === "inline")
            .flatMap(pair => pair.key.regionIds));
        const usedRegionIds = new Set([
            ...manualKeys,
            ...manualValues,
            ...predictedKeys,
            ...predictedValues,
            ...manualInline,
            ...predictedInline,
        ]);
        if (state.currentKeyId) {
            usedRegionIds.add(state.currentKeyId);
        }
        const popoverRegionIds = new Set(state.popoverPair
            ? [
                ...state.popoverPair.key.regionIds,
                ...state.popoverPair.value.regionIds,
            ]
            : []);

        for (const box of boxLayer.querySelectorAll(".ocr-box")) {
            const regionId = box.dataset.regionId;
            box.classList.toggle(
                "hidden-unused",
                state.hideUnusedRegions && !usedRegionIds.has(regionId),
            );
            box.classList.toggle("current-key", state.currentKeyId === regionId);
            box.classList.toggle("manual-key", manualKeys.has(regionId));
            box.classList.toggle("manual-value", manualValues.has(regionId));
            box.classList.toggle("predicted-key", predictedKeys.has(regionId));
            box.classList.toggle("predicted-value", predictedValues.has(regionId));
            box.classList.toggle("manual-inline", manualInline.has(regionId));
            box.classList.toggle("predicted-inline", predictedInline.has(regionId));
            box.classList.toggle("popover-target", popoverRegionIds.has(regionId));
            box.disabled = state.confirmed;
        }
    }

    function svgLine(pair, cssClass, color, dashArray) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", centerX(pair.key));
        line.setAttribute("y1", centerY(pair.key));
        line.setAttribute("x2", centerX(pair.value));
        line.setAttribute("y2", centerY(pair.value));
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", "5");
        line.setAttribute("stroke-linecap", "round");
        line.setAttribute("class", cssClass);
        if (dashArray) {
            line.setAttribute("stroke-dasharray", dashArray);
        }
        linkLayer.appendChild(line);
    }

    function renderLinks() {
        linkLayer.replaceChildren();
        for (const example of state.examples) {
            if (example.mode !== "inline") {
                svgLine(example, "manual-link", "#ea580c", "");
            }
        }
        for (const prediction of state.predictions) {
            if (prediction.mode !== "inline") {
                svgLine(prediction, "predicted-link", "#7c3aed", "16 12");
            }
        }
    }

    function pairHtml(pair, includeScore, allowRemoval = false) {
        const item = document.createElement("div");
        item.className = "pair-item";

        const key = document.createElement("strong");
        key.textContent = pair.key.text;
        const arrow = document.createElement("span");
        arrow.className = "pair-arrow";
        arrow.textContent = "→";
        const value = document.createElement("strong");
        value.textContent = pair.value.text;
        item.append(key, arrow, value);
        if (pair.mode === "inline") {
            const status = document.createElement("div");
            status.className = "pair-status";
            status.textContent = "Inline assignment trong cùng bbox.";
            item.appendChild(status);
        }
        if (includeScore) {
            const score = document.createElement("div");
            score.textContent = (
                `Độ tin cậy rule: ${(pair.score * 100).toFixed(1)}%`
            );
            item.appendChild(score);

            const recognitionConfidence = document.createElement("div");
            recognitionConfidence.className = "recognition-confidence";
            recognitionConfidence.textContent = (
                `PP-OCR trước sửa — key: ${(pair.key.score * 100).toFixed(1)}% · `
                + `value: ${(pair.value.score * 100).toFixed(1)}%`
            );
            item.appendChild(recognitionConfidence);
        }
        if (allowRemoval) {
            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.className = "remove-pair";
            removeButton.textContent = "Xóa cặp này";
            removeButton.disabled = state.confirmed;
            removeButton.addEventListener("click", () => suppressPrediction(pair));
            item.appendChild(removeButton);
        }
        return item;
    }

    function renderSuppressionSummary() {
        const suppressionSummary = document.getElementById("suppressionSummary");
        const manualCount = state.suppressedPairIds.size
            - (state.editingPrediction ? 1 : 0);
        if (state.autoOtherIds.size > 0 || manualCount > 0) {
            suppressionSummary.hidden = false;
            suppressionSummary.textContent = (
                `Đã loại ${state.autoOtherIds.size} region header khỏi suy luận `
                + `và gán OTHER; đã xóa thủ công ${manualCount} cặp.`
            );
        } else {
            suppressionSummary.hidden = true;
            suppressionSummary.textContent = "";
        }
    }

    function renderLists() {
        const exampleList = document.getElementById("exampleList");
        const previewList = document.getElementById("previewList");
        const exampleEmpty = document.getElementById("exampleEmpty");
        const previewEmpty = document.getElementById("previewEmpty");
        exampleList.replaceChildren();
        previewList.replaceChildren();
        for (const example of state.examples) {
            exampleList.appendChild(pairHtml(example, false));
        }
        for (const prediction of state.predictions) {
            previewList.appendChild(pairHtml(prediction, true, true));
        }
        document.getElementById("exampleCount").textContent = state.examples.length;
        document.getElementById("previewCount").textContent = state.predictions.length;
        exampleEmpty.hidden = state.examples.length > 0;
        previewEmpty.hidden = state.predictions.length > 0;
        if (state.examples.length > 0 && state.predictions.length === 0) {
            previewEmpty.textContent = (
                "Chưa tìm thấy cặp tương tự. Hãy chọn thêm một cặp mẫu khác "
                + "hoặc kiểm tra lại mẫu vừa chọn."
            );
        }
        renderSuppressionSummary();
    }

    function renderRuleSummary() {
        const target = document.getElementById("ruleSummary");
        if (!state.rule && !state.inlineRule) {
            target.textContent = "Chưa có rule. Số mẫu không bị giới hạn.";
            return;
        }
        const descriptions = [];
        if (state.rule) {
            descriptions.push(
                `Rule hai cột từ ${state.rule.sampleCount} mẫu; `
                + "vùng header được loại khỏi preview và xem là OTHER",
            );
        }
        if (state.inlineRule) {
            const separators = state.inlineRule.separators.length > 0
                ? state.inlineRule.separators.join(", ")
                : "chưa có separator";
            descriptions.push(
                `Inline rule từ ${state.inlineRule.sampleCount} mẫu, `
                + `separator: ${separators}`,
            );
        }
        target.textContent = (
            `${descriptions.join("; ")}; preview ${state.predictions.length} cặp.`
        );
    }

    function updateSteps() {
        const stepKey = document.getElementById("stepKey");
        const stepValue = document.getElementById("stepValue");
        const stepPreview = document.getElementById("stepPreview");
        const stepConfirm = document.getElementById("stepConfirm");
        for (const step of [stepKey, stepValue, stepPreview, stepConfirm]) {
            step.classList.remove("active", "done");
        }
        if (state.confirmed) {
            selectionStatus.textContent = "Đã xác nhận kết quả";
            for (const step of [stepKey, stepValue, stepPreview, stepConfirm]) {
                step.classList.add("done");
            }
            return;
        }
        if (state.currentKeyId) {
            const keyText = state.editingPrediction
                ? state.editingPrediction.keyText
                : regionById.get(state.currentKeyId).text;
            selectionStatus.textContent = `Đang chờ chọn VALUE cho “${keyText}”`;
            stepKey.classList.add("done");
            stepValue.classList.add("active");
            return;
        }
        selectionStatus.textContent = state.examples.length > 0
            ? "Đang kiểm tra preview — chọn thêm KEY hoặc xác nhận"
            : "Đang chờ chọn KEY đầu tiên";
        stepKey.classList.add("active");
        if (state.examples.length > 0) {
            stepPreview.classList.add("done");
            stepConfirm.classList.add("active");
        }
    }

    function updateButtons() {
        cancelKeyButton.disabled = !state.currentKeyId || state.confirmed;
        inlineCurrentKeyButton.disabled = !state.currentKeyId || state.confirmed;
        undoExampleButton.disabled = state.examples.length === 0 || state.confirmed;
        undoSuppressionButton.disabled = (
            state.suppressionHistory.length === 0 || state.confirmed
        );
        resetAllButton.disabled = (
            state.examples.length === 0 && !state.currentKeyId
        ) || state.confirmed;
        confirmResultButton.disabled = state.examples.length === 0 || state.confirmed;
        toggleUnusedRegionsButton.disabled = state.examples.length === 0;
        toggleUnusedRegionsButton.textContent = state.hideUnusedRegions
            ? "Hiện tất cả bbox OCR"
            : "Ẩn bbox xám";
    }

    function recomputePreview() {
        state.rule = inferRule();
        state.inlineRule = inferInlineRule();
        if (state.rule) {
            classifyPredictions(state.rule);
        } else {
            state.predictions = [];
            state.autoOtherIds = new Set();
        }
        const occupiedRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        for (const pair of state.predictions) {
            for (const regionId of [
                ...pair.key.regionIds,
                ...pair.value.regionIds,
            ]) {
                occupiedRegionIds.add(regionId);
            }
        }
        state.predictions.push(
            ...predictInlinePairs(state.inlineRule, occupiedRegionIds),
        );
        state.predictions.sort((left, right) => pairY(left) - pairY(right));
        refreshBoxStyles();
        renderLogicalBlocks();
        renderLinks();
        renderLists();
        renderRuleSummary();
        updateSteps();
        updateButtons();
    }

    function suppressPrediction(pair) {
        if (state.confirmed) {
            return;
        }
        const pairId = pairInstanceId(pair);
        state.suppressedPairIds.add(pairId);
        state.suppressionHistory.push(pairId);
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = (
            `Đã xóa “${pair.key.text}” → “${pair.value.text}”. `
            + "Thao tác này không thay đổi rule hoặc các cặp khác."
        );
    }

    function undoLastSuppression() {
        const pairId = state.suppressionHistory.pop();
        if (!pairId) {
            return;
        }
        state.suppressedPairIds.delete(pairId);
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = "Đã khôi phục cặp bị xóa gần nhất.";
    }

    function editPrediction(pair) {
        const pairId = pairInstanceId(pair);
        closePredictionPopover();
        state.editingPrediction = {
            pairId,
            keyText: pair.key.text,
        };
        state.suppressedPairIds.add(pairId);
        state.hideUnusedRegions = false;
        state.currentKeyId = pair.key.regionIds[0];
        recomputePreview();
        instruction.className = "instruction wait-value";
        instruction.textContent = (
            `Đang sửa cặp có key “${pair.key.text}”. `
            + "Hãy bấm bbox chứa value đúng; bấm “Hủy key đang chọn” "
            + "để khôi phục cặp cũ."
        );
    }

    function finishEditingPrediction() {
        if (!state.editingPrediction) {
            return;
        }
        state.suppressedPairIds.delete(state.editingPrediction.pairId);
        state.editingPrediction = null;
    }

    function openInlineEditor(regionId) {
        const region = regionById.get(regionId);
        const split = splitInlineText(
            region.text,
            automaticInlineSeparators,
            true,
        );
        state.inlineRegionId = regionId;
        inlineOriginal.textContent = `OCR gốc: “${region.text}”`;
        inlineSeparator.value = split ? split.separator : "";
        inlineKey.value = split ? split.key : region.text;
        inlineValue.value = split ? split.value : "";
        inlineError.textContent = split
            ? `Đã tự tách bằng “${split.separator}”.`
            : "Không tìm thấy separator tự động. Hãy tự nhập key và value.";
        inlineDialog.showModal();
        inlineKey.focus();
    }

    function updateInlineFieldsFromSeparator() {
        const region = regionById.get(state.inlineRegionId);
        const split = splitInlineText(
            region.text,
            [inlineSeparator.value],
        );
        if (!split) {
            inlineError.textContent = (
                "Không tách được OCR gốc bằng separator này. "
                + "Bạn vẫn có thể tự nhập key và value."
            );
            return;
        }
        inlineKey.value = split.key;
        inlineValue.value = split.value;
        inlineError.textContent = `Đã tách lại bằng “${split.separator}”.`;
    }

    function confirmInlineSelection() {
        const keyText = inlineKey.value.trim();
        const valueText = inlineValue.value.trim();
        if (!keyText || !valueText) {
            inlineError.textContent = "Key và value đều phải có nội dung.";
            return;
        }
        const example = buildInlineExample(
            state.inlineRegionId,
            keyText,
            valueText,
            inlineSeparator.value.trim(),
        );
        finishEditingPrediction();
        state.examples.push(example);
        state.currentKeyId = null;
        state.inlineRegionId = null;
        inlineDialog.close();
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = (
            `Đã thêm inline assignment “${keyText}” → “${valueText}”. `
            + "Nếu separator được cung cấp, các bbox tương tự đã được suy luận."
        );
    }

    function cancelInlineSelection() {
        state.inlineRegionId = null;
        inlineDialog.close();
        instruction.textContent = (
            "Đã hủy tách inline. Hãy chọn một bbox value khác hoặc bấm lại "
            + "bbox key để thử lại."
        );
    }

    function selectRegion(regionId) {
        if (state.confirmed) {
            return;
        }
        const region = regionById.get(regionId);
        if (!state.currentKeyId) {
            const prediction = predictionForRegion(regionId);
            if (prediction) {
                openPredictionPopover(prediction);
                return;
            }
        }
        closePredictionPopover();
        const regionWasUsed = state.examples.some(example => (
            example.key.regionIds.includes(regionId)
            || example.value.regionIds.includes(regionId)
        ));
        if (!state.currentKeyId) {
            if (regionWasUsed) {
                instruction.className = "instruction";
                instruction.textContent = (
                    `“${region.text}” đã thuộc một cặp mẫu. `
                    + "Hãy chọn một bbox chưa được dùng làm key mới."
                );
                return;
            }
            state.currentKeyId = regionId;
            instruction.className = "instruction wait-value";
            instruction.textContent = (
                `Đã chọn key: “${region.text}”. `
                + "Hãy bấm bbox chứa value; nếu key-value cùng bbox, "
                + "hãy bấm lại bbox hoặc chọn “Key-value cùng bbox”."
            );
            refreshBoxStyles();
            updateSteps();
            updateButtons();
            return;
        }
        if (state.currentKeyId === regionId) {
            openInlineEditor(regionId);
            return;
        }
        if (regionWasUsed) {
            instruction.textContent = (
                `“${region.text}” đã thuộc một cặp mẫu. `
                + "Hãy bấm một bbox chưa được dùng làm value."
            );
            return;
        }

        const keyId = state.currentKeyId;
        const example = buildExample(keyId, regionId);
        if (!example) {
            instruction.textContent = (
                "Không tạo được cặp logical block từ hai bbox này. "
                + "Hãy bấm một bbox value ở cột đối diện key."
            );
            return;
        }
        finishEditingPrediction();
        state.examples.push(example);
        state.currentKeyId = null;
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = (
            `Đã thêm mẫu “${example.key.text}” → “${example.value.text}” `
            + "và cập nhật preview. "
            + "Chương trình đã tự gộp các bbox liền kề khi chúng thuộc cùng hàng. "
            + "Nếu preview chưa phù hợp, hãy bấm một key khác rồi chọn value của nó. "
            + "Nếu đã đúng, bấm “Xác nhận kết quả”."
        );
    }

    function cancelCurrentKey() {
        const wasEditingPrediction = Boolean(state.editingPrediction);
        finishEditingPrediction();
        state.currentKeyId = null;
        instruction.className = "instruction";
        instruction.textContent = wasEditingPrediction
            ? "Đã hủy sửa và khôi phục cặp preview cũ."
            : "Đã hủy key. Hãy bấm vào bbox chứa key muốn chọn.";
        recomputePreview();
    }

    function undoLastExample() {
        state.examples.pop();
        if (state.examples.length === 0) {
            state.hideUnusedRegions = false;
        }
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = state.examples.length > 0
            ? "Đã xóa mẫu gần nhất và cập nhật preview. Bạn có thể chọn thêm key-value."
            : "Đã xóa hết mẫu. Hãy bấm vào bbox chứa key đầu tiên.";
    }

    function resetAll() {
        state.currentKeyId = null;
        state.examples = [];
        state.predictions = [];
        state.rule = null;
        state.inlineRule = null;
        state.inlineRegionId = null;
        state.confirmed = false;
        state.autoOtherIds = new Set();
        state.suppressedPairIds = new Set();
        state.suppressionHistory = [];
        state.hideUnusedRegions = false;
        state.popoverPair = null;
        state.editingPrediction = null;
        pairPopover.hidden = true;
        resultSection.hidden = true;
        instruction.className = "instruction";
        instruction.textContent = (
            "Bước 1: Hãy bấm vào bbox chứa key đầu tiên trên ảnh."
        );
        recomputePreview();
    }

    function loadPage(pageIndex) {
        currentPageIndex = pageIndex;
        const page = pages[currentPageIndex];
        PAGE_WIDTH = page.width;
        PAGE_HEIGHT = page.height;
        regions = page.regions;
        regionById = new Map(regions.map(region => [region.id, region]));
        pageImage.src = page.imageName;
        pageImage.alt = `Page ${page.physicalPage} của ${page.pdfName}`;
        linkLayer.setAttribute(
            "viewBox",
            `0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}`,
        );
        pageSelect.value = String(currentPageIndex);
        pagePosition.textContent = (
            `${currentPageIndex + 1}/${pages.length} · `
            + `page vật lý ${page.physicalPage}`
        );
        previousPageButton.disabled = currentPageIndex === 0;
        nextPageButton.disabled = currentPageIndex === pages.length - 1;
        if (inlineDialog.open) {
            inlineDialog.close();
        }
        updateRecognitionSummary();
        resetAll();
        renderBoxes();
    }

    function appendResultRow(index, pair, source, score) {
        const row = document.createElement("tr");
        const values = [
            String(index),
            pair.key.text,
            pair.value.text,
            source,
            score,
        ];
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        }
        row.children[3].className = source === "Người dùng chọn"
            ? "source-manual"
            : "source-rule";
        document.getElementById("resultBody").appendChild(row);
    }

    async function confirmResults() {
        const page = pages[currentPageIndex];
        const confirmedResult = {
            pdf_name: page.pdfName,
            physical_page: page.physicalPage,
            image_name: page.imageName,
            confirmed_at: new Date().toISOString(),
            pairs: [
                ...state.examples.map(pair => ({
                    key: pair.key.text,
                    value: pair.value.text,
                    source: "người dùng chọn",
                    rule_score: null,
                    key_ocr_score: pair.key.score,
                    value_ocr_score: pair.value.score,
                    key_box: [...pair.key.box],
                    value_box: [...pair.value.box],
                })),
                ...state.predictions.map(pair => ({
                    key: pair.key.text,
                    value: pair.value.text,
                    source: "suy luận",
                    rule_score: pair.score,
                    key_ocr_score: pair.key.score,
                    value_ocr_score: pair.value.score,
                    key_box: [...pair.key.box],
                    value_box: [...pair.value.box],
                })),
            ],
        };
        const response = await fetch("/confirm", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(confirmedResult),
        });
        if (!response.ok) {
            throw new Error("Không lưu được kết quả đã xác nhận.");
        }

        state.confirmed = true;
        state.currentKeyId = null;
        const resultBody = document.getElementById("resultBody");
        resultBody.replaceChildren();
        let index = 1;
        for (const example of state.examples) {
            appendResultRow(index, example, "Người dùng chọn", "100.0%");
            index += 1;
        }
        for (const prediction of state.predictions) {
            appendResultRow(
                index,
                prediction,
                "Rule suy luận",
                `${(prediction.score * 100).toFixed(1)}%`,
            );
            index += 1;
        }
        document.getElementById("resultSummary").textContent = (
            `Đã xác nhận ${index - 1} cặp: ${state.examples.length} cặp mẫu và `
            + `${state.predictions.length} cặp do Rule Induction suy luận.`
        );
        resultSection.hidden = false;
        instruction.className = "instruction confirmed";
        instruction.textContent = (
            "Đã xác nhận kết quả. Bảng key-value đã được hiển thị phía dưới ảnh; "
            + "dữ liệu đã được lưu ra file."
        );
        refreshBoxStyles();
        renderLists();
        updateSteps();
        updateButtons();
        resultSection.scrollIntoView({behavior: "smooth", block: "start"});
    }

    function toggleUnusedRegions() {
        state.hideUnusedRegions = !state.hideUnusedRegions;
        refreshBoxStyles();
        updateButtons();
    }

    cancelKeyButton.addEventListener("click", cancelCurrentKey);
    pageSelect.addEventListener("change", () => {
        loadPage(Number(pageSelect.value));
    });
    previousPageButton.addEventListener("click", () => {
        loadPage(currentPageIndex - 1);
    });
    nextPageButton.addEventListener("click", () => {
        loadPage(currentPageIndex + 1);
    });
    inlineCurrentKeyButton.addEventListener("click", () => {
        openInlineEditor(state.currentKeyId);
    });
    undoExampleButton.addEventListener("click", undoLastExample);
    undoSuppressionButton.addEventListener("click", undoLastSuppression);
    resetAllButton.addEventListener("click", resetAll);
    confirmResultButton.addEventListener("click", confirmResults);
    toggleUnusedRegionsButton.addEventListener("click", toggleUnusedRegions);
    pairPopover.addEventListener("click", event => event.stopPropagation());
    editPredictionButton.addEventListener("click", () => {
        if (state.popoverPair) {
            editPrediction(state.popoverPair);
        }
    });
    deletePredictionButton.addEventListener("click", () => {
        if (!state.popoverPair) {
            return;
        }
        const pair = state.popoverPair;
        closePredictionPopover();
        suppressPrediction(pair);
    });
    document.addEventListener("click", closePredictionPopover);
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && !pairPopover.hidden) {
            closePredictionPopover();
        }
    });
    inlineSeparator.addEventListener("change", updateInlineFieldsFromSeparator);
    cancelInlineButton.addEventListener("click", cancelInlineSelection);
    confirmInlineButton.addEventListener("click", confirmInlineSelection);
    inlineDialog.addEventListener("cancel", event => {
        event.preventDefault();
        cancelInlineSelection();
    });

    pages.forEach((page, pageIndex) => {
        const option = document.createElement("option");
        option.value = String(pageIndex);
        option.textContent = `${page.pdfName} · page ${page.physicalPage}`;
        pageSelect.appendChild(option);
    });
    loadPage(0);
</script>
</body>
</html>
"""
    html = template.replace(
        "__IMAGE_NAME__",
        escape(first_page["imageName"]),
    )
    html = html.replace("__IMAGE_WIDTH__", str(first_page["width"]))
    html = html.replace("__IMAGE_HEIGHT__", str(first_page["height"]))
    return html.replace("__PAGES_JSON__", pages_json)


def _recognize_regions(extractor, image):
    """OCR ảnh và chuẩn hóa các region dùng trong HTML."""
    text_regions, timings = extractor._recognize_text(image)
    for stage_name, seconds in timings.items():
        stage_label = {
            "Text detection": "Detector/cache",
            "Text recognition": "Tổng recognition (PP-OCR + VietOCR + hiệu chỉnh)",
        }.get(stage_name, stage_name)
        print(f"    {stage_label}: {seconds:.3f}s", flush=True)
    normalization_started = perf_counter()
    text_regions.sort(
        key=lambda region: (
            region["center"][1],
            region["center"][0],
        )
    )
    regions = [
        {
            "id": f"O{index}",
            "text": region["text"],
            "box": [int(coordinate) for coordinate in region["box"]],
            "score": float(region["score"]),
        }
        for index, region in enumerate(text_regions, start=1)
    ]
    print(
        f"    Chuẩn hóa {len(regions)} vùng OCR cho HTML: "
        f"{perf_counter() - normalization_started:.3f}s",
        flush=True,
    )
    return regions


def _recognize_regions_with_cache(extractor, image, cache_directory):
    """Trả region recognition từ cache hoặc chạy OCR khi chưa có."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256()
    cache_key.update(RECOGNITION_CACHE_VERSION.encode("utf-8"))
    cache_key.update(str(image.shape).encode("utf-8"))
    cache_key.update(str(image.dtype).encode("utf-8"))
    cache_key.update(image.tobytes())
    cache_path = cache_directory / f"{cache_key.hexdigest()}.json"

    if cache_path.is_file():
        regions = json.loads(cache_path.read_text(encoding="utf-8"))
        print(
            f"    Cache recognition: hit ({len(regions)} vùng OCR)",
            flush=True,
        )
        return regions, True

    regions = _recognize_regions(extractor, image)
    cache_path.write_text(
        json.dumps(
            regions,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return regions, False


def _create_versioned_output_directory(pdf_directory):
    """Tạo thư mục output mới dạng tên_folder_x cho mỗi lần chạy PDF."""
    output_root = Path("./output/rule_induction")
    output_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        output_directory = output_root / f"{pdf_directory.name}_{index}"
        if not output_directory.exists():
            output_directory.mkdir()
            return output_directory
        index += 1


def _create_extractor(output_directory):
    """Khởi tạo extractor và gắn cache detector theo output hiện tại."""
    print("Đang khởi tạo các model OCR...", flush=True)
    initialization_started = perf_counter()
    extractor = Page3PlusKeyValueExtractor()
    extractor.text_detector = CachedTextDetector(
        extractor.text_detector,
        output_directory / "detector_cache",
    )
    print(
        f"Khởi tạo model/cache: {perf_counter() - initialization_started:.3f}s",
        flush=True,
    )
    return extractor


def _process_image(
    extractor,
    image_path,
    output_directory,
    pdf_name,
    physical_page,
):
    """Chạy recognition trên một ảnh và tạo record tương thích với HTML."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    image_height, image_width = image.shape[:2]
    recognized_regions, recognition_cache_hit = _recognize_regions_with_cache(
        extractor,
        image,
        output_directory / "recognition_cache",
    )
    image_name = image_path.relative_to(output_directory).as_posix()
    page = {
        "pdfName": pdf_name,
        "physicalPage": physical_page,
        "imageName": image_name,
        "width": image_width,
        "height": image_height,
        "regions": recognized_regions,
    }
    return page, recognition_cache_hit


def _write_html(pages, output_directory, batch_started, processed_count, total_count):
    """Ghi HTML và in thống kê cho các page đã xử lý."""
    html_path = output_directory / "rule_induction.html"
    html_started = perf_counter()
    html_path.write_text(
        build_html(pages),
        encoding="utf-8",
    )
    print(
        f"Tạo và lưu HTML: {perf_counter() - html_started:.3f}s",
        flush=True,
    )
    print(
        f"Hoàn tất {len(pages)} page từ "
        f"{processed_count}/{total_count} nguồn trong "
        f"{perf_counter() - batch_started:.1f}s.",
        flush=True,
    )
    print(f"HTML Rule Induction: {html_path}")


def _run_pdf_directory(pdf_directory, save_images):
    """Render và tùy chọn OCR page 3+ từ folder PDF."""
    if not pdf_directory.is_dir():
        raise FileNotFoundError(f"Không tìm thấy folder PDF: {pdf_directory}")

    renderer = shutil.which("gswin64c")
    if not renderer:
        raise FileNotFoundError("Không tìm thấy gswin64c trong PATH.")

    pdf_files = sorted(
        pdf_directory.glob("*.pdf"),
        key=lambda path: path.name,
    )
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy PDF trong: {pdf_directory}")

    output_directory = _create_versioned_output_directory(pdf_directory)
    image_directory = output_directory / "images" if save_images else output_directory
    image_directory.mkdir(parents=True, exist_ok=True)
    batch_started = perf_counter()
    total_files = len(pdf_files)

    if save_images:
        rendered_count = 0
        for file_index, pdf_file in enumerate(pdf_files, start=1):
            progress = f"[{file_index}/{total_files}] {pdf_file.name}"
            print(f"{progress} | Đang render các page 3+...", flush=True)
            rendered_pages = _render_pages_3plus(
                renderer,
                pdf_file,
                image_directory,
            )
            rendered_count += len(rendered_pages)
            print(
                f"{progress} | Đã render {len(rendered_pages)} page",
                flush=True,
            )
        print(
            f"Đã lưu {rendered_count} ảnh page 3+ tại: {image_directory}",
            flush=True,
        )
        print(
            "Hãy xóa ảnh không cần thiết, sau đó chạy lại với "
            f'--image-directory "{image_directory}".',
        )
        return

    extractor = _create_extractor(output_directory)
    pages = []
    processed_pdf_count = 0

    for file_index, pdf_file in enumerate(pdf_files, start=1):
        progress = f"[{file_index}/{total_files}] {pdf_file.name}"
        print(f"{progress} | Đang render các page 3+...", flush=True)
        render_started = perf_counter()
        rendered_pages = _render_pages_3plus(
            renderer,
            pdf_file,
            output_directory,
        )
        if not rendered_pages:
            print(f"{progress} | Bỏ qua: không có page 3+.")
            continue
        processed_pdf_count += 1

        print(
            f"{progress} | Render {len(rendered_pages)} page: "
            f"{perf_counter() - render_started:.3f}s",
            flush=True,
        )
        for physical_page, output_image_path in rendered_pages:
            page_started = perf_counter()
            print(
                f"{progress} | Page vật lý {physical_page}: "
                "đang đọc cache/chạy recognition...",
                flush=True,
            )
            page, recognition_cache_hit = _process_image(
                extractor,
                output_image_path,
                output_directory,
                pdf_file.name,
                physical_page,
            )
            pages.append(page)
            if recognition_cache_hit:
                recognition_source = "cache recognition"
            else:
                detector_source = (
                    "cache detector"
                    if extractor.text_detector.last_cache_hit
                    else "detector mới"
                )
                recognition_source = f"recognition mới, {detector_source}"
            print(
                f"{progress} | Page {physical_page} hoàn tất trong "
                f"{perf_counter() - page_started:.1f}s: "
                f"{recognition_source}",
                flush=True,
            )

    if not pages:
        raise FileNotFoundError("Không có PDF nào chứa page 3+.")
    _write_html(
        pages,
        output_directory,
        batch_started,
        processed_pdf_count,
        total_files,
    )


def _parse_image_metadata(image_path):
    """Lấy tên PDF và số page vật lý từ tên ảnh do chế độ render tạo ra."""
    match = IMAGE_FILE_PATTERN.fullmatch(image_path.name)
    if not match:
        raise ValueError(
            "Tên ảnh phải có dạng <pdf_stem>_page_<số>.png/jpg: "
            f"{image_path.name}"
        )
    return (
        f"{match.group('pdf_stem')}.pdf",
        int(match.group("physical_page")),
    )


def _run_image_directory(image_directory):
    """Chạy detector/recognition trên các ảnh còn lại trong folder đã lọc."""
    image_directory = image_directory.resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Không tìm thấy folder ảnh: {image_directory}")
    output_directory = image_directory.parent
    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }
    )
    if not image_paths:
        raise FileNotFoundError(f"Không có ảnh trong: {image_directory}")

    extractor = _create_extractor(output_directory)
    pages = []
    batch_started = perf_counter()
    for image_index, image_path in enumerate(image_paths, start=1):
        pdf_name, physical_page = _parse_image_metadata(image_path)
        page_started = perf_counter()
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name} | "
            "đang đọc cache/chạy recognition...",
            flush=True,
        )
        page, recognition_cache_hit = _process_image(
            extractor,
            image_path,
            output_directory,
            pdf_name,
            physical_page,
        )
        pages.append(page)
        if recognition_cache_hit:
            recognition_source = "cache recognition"
        else:
            detector_source = (
                "cache detector"
                if extractor.text_detector.last_cache_hit
                else "detector mới"
            )
            recognition_source = f"recognition mới, {detector_source}"
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name} | "
            f"Hoàn tất trong {perf_counter() - page_started:.1f}s: "
            f"{recognition_source}",
            flush=True,
        )

    _write_html(
        pages,
        output_directory,
        batch_started,
        len(image_paths),
        len(image_paths),
    )


def main():
    """Điều phối chế độ PDF, render-only hoặc folder ảnh đã lọc."""
    arguments = parse_arguments()
    if arguments.image_directory is not None:
        _run_image_directory(arguments.image_directory)
        return
    _run_pdf_directory(arguments.pdf_directory, arguments.save_images)


if __name__ == "__main__":
    main()
