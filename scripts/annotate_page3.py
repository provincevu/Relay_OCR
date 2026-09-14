# Mục đích: tạo giao diện annotation CODE/KEY/VALUE từ Rule Induction page 3.
# Chức năng: tái sử dụng template và thuật toán gốc, bổ sung thao tác annotation.
# Luồng xử lý: HTML OCR hiện có → snapshot region → giao diện mở rộng.
# Cấu hình/parameter: thư mục chứa rule_induction.html và các ảnh page 3.
# Quan hệ: đọc template của test_rule_induction_page3_plus.py và JS mở rộng.
# Input/output: đọc HTML/OCR; ghi annotate_page3.html và annotation_manifest.json.
# Contract: giữ nguyên ID, text, box, score của OCR; không chạy lại model OCR.

import argparse
import ast
import hashlib
import json
from html import escape
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def build_annotation_html(pages):
    """Đọc literal template gốc mà không import hoặc khởi tạo thư viện OCR."""
    module = ast.parse(
        (PROJECT_DIRECTORY / "test_rule_induction_page3_plus.py").read_text(
            encoding="utf-8",
        ),
    )
    build_function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_html"
    )
    template_assignment = next(
        node for node in build_function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "template"
                for target in node.targets)
    )
    template = ast.literal_eval(template_assignment.value)
    extension = (Path(__file__).parent / "page3_annotation_ui.js").read_text(
        encoding="utf-8",
    )
    listener_marker = '    cancelKeyButton.addEventListener("click", cancelCurrentKey);'
    template = template.replace("    loadPage(0);", "    initializeAnnotations();")
    template = template.replace(listener_marker, extension + "\n" + listener_marker)
    first_page = pages[0]
    replacements = {
        "__IMAGE_NAME__": escape(first_page["imageName"]),
        "__IMAGE_WIDTH__": str(first_page["width"]),
        "__IMAGE_HEIGHT__": str(first_page["height"]),
        "__PAGES_JSON__": json.dumps(pages, ensure_ascii=False).replace("</", "<\\/"),
    }
    for marker, replacement in replacements.items():
        template = template.replace(marker, replacement)
    return template


def main():
    """Tạo giao diện annotation từ đúng snapshot OCR đã dùng trong HTML cũ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rule_output_directory", type=Path)
    arguments = parser.parse_args()
    directory = arguments.rule_output_directory.resolve()
    source_html = (directory / "rule_induction.html").read_text(encoding="utf-8")
    pages_source = source_html.split("const pages = ", 1)[1].lstrip()
    pages, _ = json.JSONDecoder().raw_decode(pages_source)
    for page in pages:
        page["ocrFingerprint"] = hashlib.sha256(
            json.dumps(
                page,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        ).hexdigest()
    (directory / "annotation_manifest.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = directory / "annotate_page3.html"
    html_path.write_text(build_annotation_html(pages), encoding="utf-8")
    print(f"Đã tạo giao diện cho {len(pages)} phiếu: {html_path}")
    print(f'Chạy server: python scripts/serve_page3_annotations.py "{directory}"')


if __name__ == "__main__":
    main()
