# Mục đích: phục vụ giao diện và lưu dataset annotation page 3.
# Chức năng: đọc/lưu phiếu đã xác nhận, giữ snapshot OCR và nhóm quan hệ.
# Luồng xử lý: GET giao diện/dataset → lọc ảnh tùy chọn → POST một phiếu → ghi annotations.json.
# Cấu hình/parameter: thư mục output, folder ảnh, file annotation và --port.
# Quan hệ: phục vụ output của annotate_page3.py cho train_page3_text_layout.py.
# Input/output: JSON phiên bản 1, pages ánh xạ image_name → phiếu đã xác nhận.
# Contract: không lưu draft; role hợp lệ được kiểm tra; relation chỉ nối KEY → VALUE.
# Tương thích dữ liệu cũ: cho phép inline KEY + VALUE cùng với INLINE_KEY_VALUE.

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


PRIMARY_ROLES = {
    "KEY",
    "VALUE",
    "HEADER",
    "FIELD_TABLE",
    "NOTE",
    "PAGE_NUMBER",
    "FORM_NUMBER",
    "INLINE_KEY_VALUE",
    "OTHER",
}
LABELS = PRIMARY_ROLES | {"UNLABELED"}
LEGACY_INLINE_ROLES = {"KEY", "VALUE"}


def _filter_manifest_pages(pages, directory, image_directory):
    """Chỉ phục vụ các page có ảnh còn tồn tại trong folder được chọn."""
    image_directory = image_directory.resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Không tìm thấy folder ảnh: {image_directory}")
    if directory != image_directory and directory not in image_directory.parents:
        raise ValueError("Folder ảnh phải nằm trong folder output.")
    available_names = {
        path.name
        for path in image_directory.iterdir()
        if path.is_file()
    }
    selected_pages = []
    for page in pages:
        image_name = Path(page["imageName"]).name
        if image_name not in available_names:
            continue
        page["imageName"] = (
            (image_directory / image_name)
            .relative_to(directory)
            .as_posix()
        )
        selected_pages.append(page)
    if not selected_pages:
        raise FileNotFoundError(
            f"Không có page nào có ảnh trong: {image_directory}"
        )
    return selected_pages


class AnnotationRequestHandler(SimpleHTTPRequestHandler):
    """Lưu đồng bộ annotation và relation của từng phiếu."""

    def send_json(self, value):
        """Trả JSON UTF-8 và không dùng cache trình duyệt cho annotation."""
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Đọc dataset đã lưu hoặc phục vụ ảnh và giao diện."""
        if self.path == "/api/annotations":
            with self.server.annotation_lock:
                dataset = {"schema_version": 1, "pages": {}}
                if self.server.dataset_path.is_file():
                    dataset = json.loads(
                        self.server.dataset_path.read_text(encoding="utf-8"),
                    )
            self.send_json(dataset)
            return
        if self.path == "/":
            self.path = "/annotate_page3.html"
        super().do_GET()

    def do_POST(self):
        """Chỉ xác nhận sau khi đã lưu phiếu với các ID và nhãn nhất quán."""
        if self.path != "/api/annotations":
            self.send_error(404)
            return
        record = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        page = self.server.pages[record["image_name"]]
        if record["ocr_fingerprint"] != page["ocrFingerprint"]:
            raise ValueError("Snapshot OCR không khớp manifest của server.")
        annotations = record["annotations"]
        region_ids = {region["id"] for region in page["regions"]}
        if set(annotations) != region_ids:
            raise ValueError("Annotation phải có đủ và đúng ID region của phiếu.")
        for roles in annotations.values():
            if (
                not isinstance(roles, list)
                or not roles
                or not set(roles) <= LABELS
                or len(set(roles)) != len(roles)
                or (
                    len(roles) > 1
                    and set(roles) != LEGACY_INLINE_ROLES
                )
            ):
                raise ValueError(
                    "Role của region không hợp lệ. Cho phép một role "
                    "hoặc inline legacy KEY + VALUE; danh sách role hợp lệ: "
                    f"{', '.join(sorted(LABELS))}."
                )
            if ("UNLABELED" in roles or "OTHER" in roles) and len(roles) != 1:
                raise ValueError("OTHER và UNLABELED phải đứng riêng.")
        linked = {role: set() for role in ["KEY", "VALUE"]}
        for relation in record["relations"]:
            if relation.get("code_region_ids") not in (None, []):
                raise ValueError("Dataset mới không hỗ trợ CODE.")
            for role in linked:
                ids = relation[f"{role.lower()}_region_ids"]
                if not ids or any(role not in annotations[region_id] for region_id in ids):
                    raise ValueError("Relation chứa ID không có nhãn tương ứng.")
                linked[role].update(ids)
        for role, ids in linked.items():
            if ids != {region_id for region_id, roles in annotations.items() if role in roles}:
                raise ValueError("KEY/VALUE phải thuộc một nhóm quan hệ.")
        saved_record = {
            "pdf_name": page["pdfName"],
            "physical_page": page["physicalPage"],
            "image_name": page["imageName"],
            "image_width": page["width"],
            "image_height": page["height"],
            "ocr_fingerprint": page["ocrFingerprint"],
            "regions": page["regions"],
            "annotations": annotations,
            "relations": record["relations"],
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.server.annotation_lock:
            dataset = {"schema_version": 1, "pages": {}}
            if self.server.dataset_path.is_file():
                dataset = json.loads(
                    self.server.dataset_path.read_text(encoding="utf-8"),
                )
            dataset["pages"][page["imageName"]] = saved_record
            temporary_path = self.server.dataset_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self.server.dataset_path)
        self.send_json(saved_record)


def main():
    """Chạy server annotation cục bộ, không khởi tạo OCR hoặc model layout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--image-directory", type=Path)
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path("annotations.json"),
    )
    parser.add_argument("--port", type=int, default=8001)
    arguments = parser.parse_args()
    directory = arguments.output_directory.resolve()
    manifest = json.loads(
        (directory / "annotation_manifest.json").read_text(encoding="utf-8"),
    )
    manifest_pages = manifest["pages"]
    if arguments.image_directory is not None:
        manifest_pages = _filter_manifest_pages(
            manifest_pages,
            directory,
            arguments.image_directory,
        )
        from scripts.annotation.annotate_page3 import build_annotation_html

        (directory / "annotate_page3.html").write_text(
            build_annotation_html(manifest_pages),
            encoding="utf-8",
        )
    server = ThreadingHTTPServer(
        ("127.0.0.1", arguments.port),
        partial(AnnotationRequestHandler, directory=str(directory)),
    )
    server.annotation_lock = Lock()
    server.dataset_path = (directory / arguments.annotations_file).resolve()
    server.pages = {page["imageName"]: page for page in manifest_pages}
    print(f"Giao diện: http://127.0.0.1:{arguments.port}/", flush=True)
    print(f"Dataset: {server.dataset_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
