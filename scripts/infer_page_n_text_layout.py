# Mục đích: suy luận CODE/KEY/VALUE trên trang vật lý n của nhiều PDF.
# Chức năng: render trang, chạy PP-OCR + VietOCR hiệu chỉnh, rồi chạy model layout.
# Luồng xử lý: folder PDF → trang n → detector/recognition cache → Layout model → JSON.
# Cấu hình/parameter: folder PDF, số trang vật lý, checkpoint, output và device.
# Quan hệ: tái sử dụng Page3PlusKeyValueExtractor và predict_page hiện có.
# Input/output: đọc PDF; ghi ảnh trang, cache detector/recognition và predictions.json.
# Contract: trang n đánh số từ 1; trang không tồn tại bị bỏ qua; link mặc định là row-link.

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

from scripts.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    predict_page,
)
from scripts.train_page3_text_layout import prepare_page
from test_rule_induction_page3_plus import (
    CachedTextDetector,
    _recognize_regions_with_cache,
)
from test_table_key_value_page3_plus import Page3PlusKeyValueExtractor


DPI = 400


def render_physical_page(renderer, pdf_path, page_number, output_path):
    """Render một trang vật lý; trả False nếu PDF không có trang đó."""
    temporary_path = output_path.with_suffix(".tmp.png")
    if temporary_path.is_file():
        temporary_path.unlink()
    command = [
        renderer,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        f"-r{DPI}",
        f"-dFirstPage={page_number}",
        f"-dLastPage={page_number}",
        f"-sOutputFile={temporary_path}",
        str(pdf_path),
    ]
    subprocess.run(command, check=False)
    if not temporary_path.is_file():
        return False
    temporary_path.replace(output_path)
    return True


def load_layout_model(checkpoint_directory, device):
    """Tải checkpoint Layout model và tokenizer."""
    checkpoint = torch.load(
        checkpoint_directory / "model.pt",
        map_location=device,
        weights_only=True,
    )
    config = LayoutLMv3Config.from_dict(checkpoint["encoder_config"])
    model = Page3LayoutModel(
        LayoutLMv3Model(config),
        len(checkpoint["roles"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(checkpoint_directory)
    return model, tokenizer, checkpoint["roles"]


def sort_regions(regions):
    """Sắp xếp region theo thứ tự đọc để ghép text ổn định."""
    return sorted(
        regions,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )


def join_region_text(region_by_id, region_ids):
    """Ghép text của các region theo tọa độ mà vẫn giữ danh sách ID."""
    regions = [region_by_id[region_id] for region_id in region_ids]
    return " ".join(
        region["text"].strip()
        for region in sort_regions(regions)
        if region["text"].strip()
    )


def add_link_texts(prediction):
    """Thêm text dẫn xuất vào link nhưng giữ nguyên ID region làm dữ liệu chính."""
    region_by_id = {
        region["id"]: region
        for region in prediction["regions"]
    }
    decorated_links = []
    for link in prediction["links"]:
        decorated_links.append({
            **link,
            "key_text": join_region_text(
                region_by_id,
                [link["key_region_id"]],
            ),
            "value_text": join_region_text(
                region_by_id,
                [link["value_region_id"]],
            ),
        })
    decorated_code_key_links = []
    for link in prediction["code_key_links"]:
        decorated_code_key_links.append({
            **link,
            "code_text": join_region_text(
                region_by_id,
                [link["code_region_id"]],
            ),
            "key_text": join_region_text(
                region_by_id,
                [link["key_region_id"]],
            ),
        })
    decorated_triples = []
    for link in prediction["code_key_value_links"]:
        decorated_triples.append({
            **link,
            "code_text": join_region_text(
                region_by_id,
                [link["code_region_id"]],
            ),
            "key_text": join_region_text(
                region_by_id,
                [link["key_region_id"]],
            ),
            "value_text": join_region_text(
                region_by_id,
                [link["value_region_id"]],
            ),
        })
    prediction["links"] = decorated_links
    prediction["code_key_links"] = decorated_code_key_links
    prediction["code_key_value_links"] = decorated_triples
    prediction["code_key_links_by_mode"] = {
        "row": decorated_code_key_links,
    }
    prediction["code_key_value_links_by_mode"] = {
        "row": decorated_triples,
    }
    prediction["triples"] = decorated_triples
    return prediction


def infer_page(
    pdf_path,
    page_number,
    renderer,
    output_directory,
    model,
    tokenizer,
    roles,
    device,
    threshold,
    extractor,
):
    """Render, OCR và suy luận một PDF; trả None nếu thiếu trang n."""
    image_directory = output_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    image_path = image_directory / f"{pdf_path.stem}_page_{page_number}.png"
    if not render_physical_page(
        renderer,
        pdf_path,
        page_number,
        image_path,
    ):
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    image_height, image_width = image.shape[:2]
    recognized_regions, recognition_cache_hit = (
        _recognize_regions_with_cache(
            extractor,
            image,
            output_directory / "recognition_cache",
        )
    )
    page = prepare_page({
        "image_width": image_width,
        "image_height": image_height,
        "regions": recognized_regions,
    })
    encoded = encode_page(
        page,
        tokenizer,
        None,
        "B",
        device,
    )
    with torch.inference_mode():
        prediction = predict_page(
            model,
            encoded,
            page,
            roles,
            threshold,
            "row",
        )
    ocr_scores = {
        region["id"]: region["score"]
        for region in recognized_regions
    }
    for region in prediction["regions"]:
        region["ocr_score"] = ocr_scores[region["id"]]
    prediction = add_link_texts(prediction)
    prediction.update({
        "pdf_name": pdf_path.name,
        "physical_page": page_number,
        "image_name": image_path.name,
        "image_path": image_path.relative_to(output_directory).as_posix(),
        "image_width": image_width,
        "image_height": image_height,
        "recognition_source": (
            "cache recognition"
            if recognition_cache_hit
            else "PP-OCR + VietOCR hiệu chỉnh"
        ),
    })
    return prediction


def parse_arguments():
    """Đọc tham số folder PDF và trang vật lý cần suy luận."""
    parser = argparse.ArgumentParser(
        description="Suy luận Layout model trên trang vật lý n của các PDF.",
    )
    parser.add_argument("--pdf-directory", type=Path, required=True)
    parser.add_argument("--page-number", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    """Chạy detector, recognition và Layout model trên các trang đã chọn."""
    arguments = parse_arguments()
    if arguments.page_number < 1:
        raise ValueError("--page-number phải bắt đầu từ 1.")
    if not arguments.pdf_directory.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy folder PDF: {arguments.pdf_directory}"
        )
    pdf_files = sorted(
        arguments.pdf_directory.glob("*.pdf"),
        key=lambda path: path.name,
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"Không tìm thấy PDF trong: {arguments.pdf_directory}"
        )
    renderer = shutil.which("gswin64c")
    if not renderer:
        raise FileNotFoundError("Không tìm thấy gswin64c trong PATH.")

    arguments.output.mkdir(parents=True, exist_ok=True)
    print("Đang khởi tạo PP-OCR, VietOCR và Layout model...", flush=True)
    extractor = Page3PlusKeyValueExtractor()
    extractor.text_detector = CachedTextDetector(
        extractor.text_detector,
        arguments.output / "detector_cache",
    )
    model, tokenizer, roles = load_layout_model(
        arguments.checkpoint,
        arguments.device,
    )

    pages = []
    total_started = perf_counter()
    total_files = len(pdf_files)
    for file_index, pdf_path in enumerate(pdf_files, start=1):
        progress = f"[{file_index}/{total_files}] {pdf_path.name}"
        item_started = perf_counter()
        print(
            f"{progress} | Đang xử lý trang vật lý "
            f"{arguments.page_number}...",
            flush=True,
        )
        prediction = infer_page(
            pdf_path,
            arguments.page_number,
            renderer,
            arguments.output,
            model,
            tokenizer,
            roles,
            arguments.device,
            arguments.threshold,
            extractor,
        )
        if prediction is None:
            print(
                f"{progress} | Bỏ qua: không có trang "
                f"{arguments.page_number}.",
                flush=True,
            )
            continue
        pages.append(prediction)
        print(
            f"{progress} | Hoàn tất {len(prediction['regions'])} region, "
            f"{len(prediction['triples'])} triple trong "
            f"{perf_counter() - item_started:.1f}s",
            flush=True,
        )

    if not pages:
        raise FileNotFoundError(
            f"Không có PDF nào chứa trang {arguments.page_number}."
        )
    result = {
        "schema_version": 1,
        "page_number": arguments.page_number,
        "dpi": DPI,
        "link_mode": "row",
        "threshold": arguments.threshold,
        "checkpoint": str(arguments.checkpoint),
        "device": arguments.device,
        "ocr_contract": "PP-OCR + VietOCR hiệu chỉnh khi cần",
        "pages": pages,
    }
    output_path = arguments.output / "predictions.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Đã lưu {len(pages)}/{total_files} PDF: {output_path} "
        f"({perf_counter() - total_started:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
