# Mục đích: chuẩn bị dữ liệu page 3 cho GPT và tạo ground truth từ kết quả ghép region.
# Chức năng: đọc cache page/detector, chạy recognition PP-OCR + VietOCR
# và ghép key-value.
# Luồng xử lý: prepare tạo manifest/ảnh overlay; build đọc kết quả GPT và tạo dataset.
# Cấu hình/parameter: hai lệnh prepare và build; toàn bộ ảnh và tọa độ giữ ở 400 DPI.
# Quan hệ: dùng cache của test_rule_induction_page3_plus.py và quy tắc
# hiệu chỉnh OCR hiện có.
# Input/output: đọc PDF/cache/ảnh 400 DPI; ghi dữ liệu GPT và ground truth JSON.
# Contract: GPT trả key_region_ids/value_region_ids; key_text/value_text là tùy
# chọn khi key và value cần tách từ cùng một polygon.

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

import cv2
import numpy as np
from paddleocr import TextRecognition

from scr.services.recognitor import Recognitor
from test_rule_induction_page3_plus import DETECTOR_CACHE_VERSION
from test_table_key_value_page3_plus import correct_text_with_vietocr


PDF_DIRECTORY = PROJECT_DIRECTORY / "data" / "pdf"
RULE_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "rule_induction" / "pdf"
PAGE_CACHE_PATH = RULE_OUTPUT_DIRECTORY / "page_selection_cache.json"
DETECTOR_CACHE_DIRECTORY = RULE_OUTPUT_DIRECTORY / "detector_cache"
GPT_INPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "page3_gpt_input"
OVERLAY_DIRECTORY = GPT_INPUT_DIRECTORY / "overlays"
MANIFEST_PATH = GPT_INPUT_DIRECTORY / "manifest.json"
GROUND_TRUTH_PATH = PROJECT_DIRECTORY / "data" / "page3_ground_truth_generated.json"
DPI = 400
DETECTION_PARAMETERS = {
    "limit_side_len": 1600,
    "limit_type": "max",
    "thresh": 0.3,
    "box_thresh": 0.6,
    "unclip_ratio": 1.5,
}


def parse_arguments():
    """Đọc flow cần chạy và đường dẫn kết quả GPT khi build."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="flow", required=True)
    subparsers.add_parser("prepare")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("gpt_result_path", type=Path)
    return parser.parse_args()


def load_detector_polygons(image):
    """Đọc polygon detector từ đúng cache đã tạo cho ảnh và tham số hiện tại."""
    cache_key = hashlib.sha256()
    cache_key.update(DETECTOR_CACHE_VERSION.encode("utf-8"))
    cache_key.update(str(image.shape).encode("utf-8"))
    cache_key.update(str(image.dtype).encode("utf-8"))
    cache_key.update(image.tobytes())
    cache_key.update(
        json.dumps(
            DETECTION_PARAMETERS,
            sort_keys=True,
        ).encode("utf-8")
    )
    cache_path = DETECTOR_CACHE_DIRECTORY / f"{cache_key.hexdigest()}.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy detector cache: {cache_path}")
    return json.loads(cache_path.read_text(encoding="utf-8"))


def recognize_regions(image, polygons, text_recognizer, vietocr_recognizer):
    """Recognition từng polygon và hiệu chỉnh text bằng VietOCR."""
    image_height, image_width = image.shape[:2]
    regions = []
    crops = []

    for polygon_data in polygons:
        polygon = np.array(polygon_data, dtype=np.float32)
        x_min, y_min = map(int, polygon.min(axis=0))
        x_max, y_max = map(int, polygon.max(axis=0))
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, image_width)
        y_max = min(y_max, image_height)
        crops.append(image[y_min:y_max, x_min:x_max])
        regions.append(
            {
                "polygon": [
                    [int(x_coordinate), int(y_coordinate)]
                    for x_coordinate, y_coordinate in polygon
                ],
                "bbox": [x_min, y_min, x_max, y_max],
                "center": [
                    float(polygon[:, 0].mean()),
                    float(polygon[:, 1].mean()),
                ],
            }
        )

    indexed_crops = list(enumerate(crops))
    indexed_crops.sort(
        key=lambda item: item[1].shape[1] / item[1].shape[0],
    )
    sorted_crops = [crop for _, crop in indexed_crops]
    recognition_results = text_recognizer.predict(
        input=sorted_crops,
        batch_size=8,
    )
    indexed_results = [
        (original_index, recognition_result)
        for (original_index, _), recognition_result in zip(
            indexed_crops,
            recognition_results,
            strict=True,
        )
    ]
    indexed_results.sort(key=lambda item: item[0])

    for region, (crop_index, recognition_result) in zip(
        regions,
        indexed_results,
        strict=True,
    ):
        ppocr_text = recognition_result["rec_text"].strip()
        vietocr_text = vietocr_recognizer.recognize(crops[crop_index]).strip()
        region["ppocr_text"] = ppocr_text
        region["vietocr_text"] = vietocr_text
        region["text"] = correct_text_with_vietocr(
            ppocr_text,
            vietocr_text,
        )
        region["ppocr_score"] = float(recognition_result["rec_score"])

    regions.sort(
        key=lambda region: (
            region["center"][1],
            region["center"][0],
        )
    )
    for index, region in enumerate(regions, start=1):
        region["id"] = f"R{index:03d}"
        region.pop("center")
    return regions


def create_overlay(image, regions, output_path):
    """Vẽ polygon và ID lên một ảnh riêng để GPT đối chiếu region."""
    overlay = image.copy()
    for region in regions:
        polygon = np.array(region["polygon"], dtype=np.int32)
        cv2.polylines(
            overlay,
            [polygon],
            True,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        x_min, y_min, _, _ = region["bbox"]
        label = region["id"]
        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_y = max(y_min - 4, label_size[1] + baseline + 2)
        cv2.rectangle(
            overlay,
            (x_min, label_y - label_size[1] - baseline - 2),
            (x_min + label_size[0] + 4, label_y + 2),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (x_min + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    saved = cv2.imwrite(str(output_path), overlay)
    if not saved:
        raise RuntimeError(f"Không lưu được ảnh overlay: {output_path}")


def prepare():
    """Tạo manifest OCR và ảnh overlay từ cache page 3/detector."""
    page_cache = json.loads(PAGE_CACHE_PATH.read_text(encoding="utf-8"))
    OVERLAY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    text_recognizer = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
        cpu_threads=4,
    )
    vietocr_recognizer = Recognitor({})
    pages = []

    for pdf_path in sorted(PDF_DIRECTORY.glob("*.pdf"), key=lambda path: path.name):
        physical_page = page_cache[pdf_path.name]["physical_page"]
        if physical_page is None:
            continue

        image_name = f"{pdf_path.stem}_page_3.png"
        image_path = RULE_OUTPUT_DIRECTORY / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh 400 DPI: {image_path}")

        polygons = load_detector_polygons(image)
        regions = recognize_regions(
            image,
            polygons,
            text_recognizer,
            vietocr_recognizer,
        )
        overlay_path = OVERLAY_DIRECTORY / image_name
        create_overlay(image, regions, overlay_path)
        image_height, image_width = image.shape[:2]
        pages.append(
            {
                "pdf_name": pdf_path.name,
                "physical_page": physical_page,
                "image_name": image_name,
                "image_path": image_path.relative_to(PROJECT_DIRECTORY).as_posix(),
                "overlay_path": overlay_path.relative_to(
                    PROJECT_DIRECTORY
                ).as_posix(),
                "image_width": image_width,
                "image_height": image_height,
                "regions": regions,
            }
        )
        print(f"Đã chuẩn bị {image_name}: {len(regions)} region")

    manifest = {
        "dpi": DPI,
        "coordinate_origin": "Góc trên bên trái của ảnh",
        "coordinate_format": "polygon gồm các điểm [x, y] theo pixel",
        "gpt_output_contract": {
            "key_region_ids": ["R001"],
            "value_region_ids": ["R002"],
            "key_text": "Nội dung key tùy chọn",
            "value_text": "Nội dung value tùy chọn",
        },
        "pages": pages,
    }
    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest cho GPT: {MANIFEST_PATH}")


def merge_region_boxes(regions):
    """Tạo bbox bao ngoài một hoặc nhiều region."""
    return [
        min(region["bbox"][0] for region in regions),
        min(region["bbox"][1] for region in regions),
        max(region["bbox"][2] for region in regions),
        max(region["bbox"][3] for region in regions),
    ]


def build(gpt_result_path):
    """Tạo ground truth từ danh sách region key/value do GPT trả về."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    gpt_result = json.loads(gpt_result_path.read_text(encoding="utf-8"))
    ground_truth = {}

    for page in manifest["pages"]:
        image_name = page["image_name"]
        region_by_id = {
            region["id"]: region
            for region in page["regions"]
        }
        records = []
        for pair in gpt_result[image_name]:
            key_region_ids = pair["key_region_ids"]
            value_region_ids = pair["value_region_ids"]
            if not key_region_ids or not value_region_ids:
                raise ValueError(
                    f"Cặp key-value của {image_name} không được để trống region ID."
                )
            key_regions = [region_by_id[region_id] for region_id in key_region_ids]
            value_regions = [
                region_by_id[region_id]
                for region_id in value_region_ids
            ]
            records.append(
                {
                    "key": pair.get(
                        "key_text",
                        " ".join(region["text"] for region in key_regions),
                    ),
                    "value": pair.get(
                        "value_text",
                        " ".join(
                            region["text"]
                            for region in value_regions
                        ),
                    ),
                    "key_region_ids": key_region_ids,
                    "value_region_ids": value_region_ids,
                    "key_polygons": [
                        region["polygon"]
                        for region in key_regions
                    ],
                    "value_polygons": [
                        region["polygon"]
                        for region in value_regions
                    ],
                    "key_bbox": merge_region_boxes(key_regions),
                    "value_bbox": merge_region_boxes(value_regions),
                }
            )
        ground_truth[image_name] = records

    GROUND_TRUTH_PATH.write_text(
        json.dumps(
            ground_truth,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ground truth đã tạo: {GROUND_TRUTH_PATH}")


def main():
    """Điều phối flow prepare hoặc build."""
    arguments = parse_arguments()
    if arguments.flow == "prepare":
        prepare()
        return
    build(arguments.gpt_result_path)


if __name__ == "__main__":
    main()
