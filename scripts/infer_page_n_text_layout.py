# Mục đích: suy luận KEY/VALUE trên toàn bộ page dữ liệu 3+.
# Chức năng: dùng quy tắc chọn page dữ liệu, chạy PP-OCR + VietOCR hiệu chỉnh, rồi chạy model layout.
# Luồng xử lý: folder PDF → chọn page dữ liệu → detector/recognition cache → Layout model → JSON.
# Cấu hình/parameter: folder PDF, checkpoint, output, recognition cache,
# threshold và device nhận từ command line.
# Quan hệ: dùng cùng luồng render/OCR của test_rule_induction_page3_plus.py.
# Input/output: đọc PDF; ghi ảnh page 3+, cache detector/recognition và predictions.json.
# Contract: page 3+ bắt đầu từ page dữ liệu đầu tiên sau page 1; link mặc định là row-link.

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_pages_3plus import DPI
from scripts.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    predict_page,
)
from scripts.page_n_prediction_filter import (
    filter_prediction_pages,
    filter_unlinked_regions,
)
from scripts.train_page3_text_layout import prepare_page
from test_rule_induction_page3_plus import (
    _create_extractor,
    _process_image,
    _render_pages_3plus,
)


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
        checkpoint.get("head_dropout", 0.3),
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
    """Ghép text của các region theo tọa độ và giữ ID region trong link."""
    regions = [region_by_id[region_id] for region_id in region_ids]
    return " ".join(
        region["text"].strip()
        for region in sort_regions(regions)
        if region["text"].strip()
    )


def add_link_texts(prediction):
    """Thêm text dẫn xuất vào link nhưng giữ region ID làm dữ liệu chính."""
    region_by_id = {
        region["id"]: region
        for region in prediction["regions"]
    }
    prediction["links"] = [
        {
            **link,
            "key_text": join_region_text(
                region_by_id,
                [link["key_region_id"]],
            ),
            "value_text": join_region_text(
                region_by_id,
                [link["value_region_id"]],
            ),
        }
        for link in prediction["links"]
    ]
    return prediction


def infer_page(
    page_record,
    model,
    tokenizer,
    roles,
    device,
    threshold,
):
    """Chạy Layout model trên một page đã được OCR."""
    page = prepare_page({
        "image_width": page_record["width"],
        "image_height": page_record["height"],
        "regions": page_record["regions"],
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
        for region in page_record["regions"]
    }
    for region in prediction["regions"]:
        region["ocr_score"] = ocr_scores[region["id"]]
    prediction = add_link_texts(prediction)
    prediction = filter_unlinked_regions(prediction)
    prediction.update({
        "pdf_name": page_record["pdfName"],
        "physical_page": page_record["physicalPage"],
        "image_name": page_record["imageName"],
        "image_path": page_record["imageName"],
        "image_width": page_record["width"],
        "image_height": page_record["height"],
    })
    return prediction


def recognition_source(extractor, cache_hit):
    """Mô tả nguồn recognition theo cache detector/recognition hiện tại."""
    if cache_hit:
        return "cache recognition"
    detector_source = (
        "cache detector"
        if extractor.text_detector.last_cache_hit
        else "detector mới"
    )
    return f"recognition mới, {detector_source}"


def parse_arguments():
    """Đọc folder PDF và các tham số suy luận page 3+."""
    parser = argparse.ArgumentParser(
        description="Suy luận Layout model trên toàn bộ page dữ liệu 3+.",
    )
    parser.add_argument("--pdf-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recognition-cache-directory", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    """Render, OCR và suy luận tuần tự trên mọi page dữ liệu 3+ của folder PDF."""
    arguments = parse_arguments()
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
    image_directory = arguments.output / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    print("Đang khởi tạo PP-OCR, VietOCR và Layout model...", flush=True)
    extractor = _create_extractor(arguments.output)
    model, tokenizer, roles = load_layout_model(
        arguments.checkpoint,
        arguments.device,
    )

    pages = []
    total_started = perf_counter()
    total_files = len(pdf_files)
    processed_pages = 0
    for file_index, pdf_path in enumerate(pdf_files, start=1):
        progress = f"[{file_index}/{total_files}] {pdf_path.name}"
        print(f"{progress} | Đang render các page dữ liệu 3+...", flush=True)
        rendered_pages = _render_pages_3plus(
            renderer,
            pdf_path,
            image_directory,
        )
        if not rendered_pages:
            print(f"{progress} | Bỏ qua: không có page dữ liệu 3+.", flush=True)
            continue

        for physical_page, image_path in rendered_pages:
            page_started = perf_counter()
            print(
                f"{progress} | Page vật lý {physical_page}: "
                "đang đọc cache/chạy recognition...",
                flush=True,
            )
            page_record, cache_hit = _process_image(
                extractor,
                image_path,
                arguments.output,
                pdf_path.name,
                physical_page,
                recognition_cache_directory=(
                    arguments.recognition_cache_directory
                ),
            )
            prediction = infer_page(
                page_record,
                model,
                tokenizer,
                roles,
                arguments.device,
                arguments.threshold,
            )
            prediction["recognition_source"] = recognition_source(
                extractor,
                cache_hit,
            )
            pages.append(prediction)
            processed_pages += 1
            print(
                f"{progress} | Page {physical_page} hoàn tất: "
                f"{len(prediction['regions'])} region, "
                f"{len(prediction['links'])} link KEY → VALUE trong "
                f"{perf_counter() - page_started:.1f}s",
                flush=True,
            )

    if not pages:
        raise FileNotFoundError("Không có PDF nào chứa page dữ liệu 3+.")
    filter_prediction_pages(pages)
    for page in pages:
        add_link_texts(page)
    result = {
        "schema_version": 1,
        "page_range": "data_pages_3plus",
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
        f"Đã lưu {processed_pages} page dữ liệu 3+ từ {total_files} PDF: "
        f"{output_path} ({perf_counter() - total_started:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
