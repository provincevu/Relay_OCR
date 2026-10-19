# Mục đích: tạo cache region embedding từ annotation page 3+ đã xác nhận.
# Chức năng: chạy LayoutLMv3 encoder một lần cho mỗi page và lưu embedding trên CPU.
# Luồng xử lý: annotations.json → encode page → gộp embedding theo region → cache .pt.
# Cấu hình/parameter: dataset, output cache, encoder, device và giới hạn số page.
# Quan hệ: dùng chung encode_page và encoder của page3_layout_model.
# Input/output: nhận annotations.json; tạo index.json và một file .pt cho mỗi page.
# Contract: không sửa annotation; cache giữ region ID, nhãn, relation và embedding.

import argparse
import json
from pathlib import Path

import torch
from transformers import LayoutLMv3TokenizerFast

from scripts.page3_layout_model import (
    encode_page,
    encode_region_embeddings,
    load_pretrained,
)
from scripts.train_page3_text_layout import ROLES, _load_pages


def parse_arguments():
    """Đọc tham số tạo cache embedding."""
    parser = argparse.ArgumentParser(
        description="Tạo cache region embedding cho page3 LayoutLM.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="microsoft/layoutlmv3-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Chỉ xử lý tối đa số page này để kiểm tra cache.",
    )
    return parser.parse_args()


def _page_payload(page, embedding):
    """Tạo dữ liệu page độc lập để lưu vào file cache."""
    return {
        "pdf_name": page["pdf_name"],
        "physical_page": page["physical_page"],
        "image_name": page["image_name"],
        "image_width": page["image_width"],
        "image_height": page["image_height"],
        "ocr_fingerprint": page["ocr_fingerprint"],
        "confirmed_at": page["confirmed_at"],
        "regions": page["regions"],
        "annotations": page["annotations"],
        "relations": page["relations"],
        "region_embeddings": embedding.detach().cpu(),
    }


def prepare_embeddings(arguments):
    """Chạy encoder một lần trên từng page và ghi cache CPU."""
    pages = _load_pages(arguments.dataset)
    if arguments.max_pages is not None:
        if arguments.max_pages <= 0:
            raise ValueError("--max-pages phải lớn hơn 0.")
        pages = pages[:arguments.max_pages]
    if not pages:
        raise ValueError("Dataset không có page đã xác nhận để tạo embedding.")

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.encoder)
    model = load_pretrained(
        arguments.encoder,
        "B",
        len(ROLES),
        arguments.seed,
    ).to(arguments.device)
    model.eval()
    arguments.output.mkdir(parents=True, exist_ok=True)
    index_pages = []

    with torch.inference_mode():
        for page_index, page in enumerate(pages, start=1):
            print(
                f"Đang tạo embedding page {page_index}/{len(pages)}: "
                f"{page['image_name']}",
                flush=True,
            )
            encoded = encode_page(
                page,
                tokenizer,
                None,
                "B",
                arguments.device,
            )
            embedding = encode_region_embeddings(
                model.encoder,
                encoded,
                len(page["regions"]),
            )
            cache_name = f"page_{page_index:06d}.pt"
            torch.save(
                _page_payload(page, embedding),
                arguments.output / cache_name,
            )
            index_pages.append({
                "file": cache_name,
                "pdf_name": page["pdf_name"],
                "physical_page": page["physical_page"],
                "image_name": page["image_name"],
                "region_count": len(page["regions"]),
            })
            del encoded, embedding

    index = {
        "schema_version": 1,
        "cache_type": "page3_region_embeddings",
        "encoder": arguments.encoder,
        "variant": "B",
        "roles": ROLES,
        "role_mode": "single_class",
        "link_mode": "row",
        "embedding_dimension": model.encoder.config.hidden_size,
        "pages": index_pages,
    }
    (arguments.output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Đã lưu cache {len(index_pages)} page tại: {arguments.output}",
        flush=True,
    )


def main():
    """Điều phối việc tạo cache embedding."""
    prepare_embeddings(parse_arguments())


if __name__ == "__main__":
    main()
