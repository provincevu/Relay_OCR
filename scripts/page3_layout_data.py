# Mục đích: chuẩn bị cùng dữ liệu PaddleOCR page 3 cho hai nhánh LayoutLMv3.
# Chức năng: OCR ảnh page 3, đọc manifest và tạo nhãn region/liên kết.
# Luồng xử lý: ảnh → manifest OCR → region KEY/VALUE và cạnh key → value.
# Cấu hình/parameter: header_labels tùy chọn ánh xạ image_name → danh sách ID.
# Quan hệ: được scripts.compare_page3_layout sử dụng cho cả A và B.
# Input/output: JSON manifest, annotation → danh sách page có nhãn.
# Contract: bbox pixel; ppocr_text là đầu vào; một region có thể có nhiều vai trò.
# Contract: OCR mới dùng ID Oxxx và cần annotation tương ứng với manifest mới.

import json
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Lưu dữ liệu thử nghiệm bằng JSON tiếng Việt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def recognize_pages(image_directory, output_path):
    """Tạo manifest PaddleOCR thuần từ các ảnh *_page_3.png đã render."""
    import cv2
    import numpy as np
    from paddleocr import TextDetection, TextRecognition

    detector = TextDetection(
        model_name="PP-OCRv6_small_det",
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
        cpu_threads=6,
    )
    recognizer = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
        cpu_threads=6,
    )
    pages = []
    for image_path in sorted(image_directory.glob("*_page_3.png")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        regions = []
        for result in detector.predict(
            input=image,
            limit_side_len=1600,
            limit_type="max",
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
        ):
            for polygon in result["dt_polys"]:
                points = np.asarray(polygon)
                left, top = points.min(axis=0).astype(int)
                right, bottom = points.max(axis=0).astype(int)
                left, top = max(0, left), max(0, top)
                right, bottom = min(width, right), min(height, bottom)
                recognition = recognizer.predict(
                    input=image[top:bottom, left:right],
                    batch_size=1,
                )[0]
                regions.append({
                    "bbox": [int(left), int(top), int(right), int(bottom)],
                    "ppocr_text": recognition["rec_text"],
                    "ppocr_score": float(recognition["rec_score"]),
                })
        regions.sort(key=lambda region: (
            region["bbox"][1] + region["bbox"][3],
            region["bbox"][0] + region["bbox"][2],
        ))
        for index, region in enumerate(regions, start=1):
            region["id"] = f"O{index:03d}"
        pages.append({
            "image_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "image_width": width,
            "image_height": height,
            "regions": regions,
        })
        print(f"Đã OCR: {image_path.name} ({len(regions)} region)", flush=True)
    write_json(output_path, {"pages": pages})


def load_pages(manifest_path):
    """Đọc manifest và giữ text PaddleOCR gốc cho cả hai nhánh."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest["pages"]
    for page in pages:
        image_path = Path(page["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_DIRECTORY / image_path
        page["image_path"] = str(image_path)
        for region in page["regions"]:
            region["text"] = region["ppocr_text"]
    return pages


def label_pages(pages, annotations_path, header_labels_path=None):
    """Gán đa nhãn và cạnh region từ annotation, không suy diễn nhãn header."""
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    header_labels = None
    roles = ["KEY", "VALUE"]
    if header_labels_path is not None:
        header_labels = json.loads(
            header_labels_path.read_text(encoding="utf-8"),
        )
        roles.append("HEADER")
    for page in pages:
        indices = {
            region["id"]: index
            for index, region in enumerate(page["regions"])
        }
        labels = [[0.0] * len(roles) for _ in page["regions"]]
        links = set()
        for pair in annotations[page["image_name"]]:
            for key_id in pair["key_region_ids"]:
                labels[indices[key_id]][0] = 1.0
                for value_id in pair["value_region_ids"]:
                    links.add((indices[key_id], indices[value_id]))
            for value_id in pair["value_region_ids"]:
                labels[indices[value_id]][1] = 1.0
        if header_labels is not None:
            for region_id in header_labels[page["image_name"]]:
                labels[indices[region_id]][2] = 1.0
        page["role_labels"] = labels
        page["link_labels"] = sorted(links)
    return roles
