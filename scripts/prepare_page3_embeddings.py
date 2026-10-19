# Mục đích: tạo cache region embedding từ annotation page 3+ đã xác nhận.
# Chức năng: chạy LayoutLMv3 encoder một lần cho mỗi page và lưu embedding trên CPU.
# Luồng xử lý: annotations.json → encode page → gộp embedding theo region → cache .pt.
# Cấu hình/parameter: dataset, output cache, encoder, device, AMP và giới hạn số page.
# Quan hệ: dùng chung iter_page_chunks và encoder của page3_layout_model.
# Input/output: nhận annotations.json; tạo index.json và một file .pt cho mỗi page.
# Contract: không sửa annotation; cache giữ region ID, nhãn, relation và embedding.

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import LayoutLMv3TokenizerFast

from scripts.page3_layout_model import (
    iter_page_chunks,
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
        "--amp",
        action="store_true",
        help="Dùng float16 khi chạy encoder trên CUDA.",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=100,
        help="Thời gian nghỉ giữa các chunk khi chạy CUDA.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Chỉ xử lý tối đa số page này để kiểm tra cache.",
    )
    return parser.parse_args()


def _stream_region_embeddings(
    encoder,
    page,
    tokenizer,
    device,
    use_amp,
    pause_ms,
):
    """Chạy từng chunk, chuyển hidden về CPU và giải phóng GPU ngay."""
    pooled = None
    counts = None
    for chunk in iter_page_chunks(page, tokenizer, device):
        inputs = {
            name: value
            for name, value in chunk.items()
            if name != "region_indices"
        }
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            hidden = encoder(**inputs).last_hidden_state
        text_hidden = hidden[0, :chunk["input_ids"].shape[1]].float().cpu()
        indices = chunk["region_indices"].detach().cpu()
        valid = indices >= 0
        if pooled is None:
            pooled = text_hidden.new_zeros(
                (len(page["regions"]), text_hidden.shape[-1]),
            )
            counts = text_hidden.new_zeros((len(page["regions"]), 1))
        pooled.index_add_(0, indices[valid], text_hidden[valid])
        counts.index_add_(
            0,
            indices[valid],
            text_hidden.new_ones((int(valid.sum()), 1)),
        )
        del chunk, inputs, hidden, text_hidden, indices, valid
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if pause_ms:
                time.sleep(pause_ms / 1000)
    return pooled / counts


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
    if arguments.pause_ms < 0:
        raise ValueError("--pause-ms không được âm.")
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
    use_amp = arguments.amp and arguments.device.startswith("cuda")
    arguments.output.mkdir(parents=True, exist_ok=True)
    index_pages = []

    with torch.inference_mode():
        for page_index, page in enumerate(pages, start=1):
            print(
                f"Đang tạo embedding page {page_index}/{len(pages)}: "
                f"{page['image_name']}",
                flush=True,
            )
            embedding = _stream_region_embeddings(
                model.encoder,
                page,
                tokenizer,
                arguments.device,
                use_amp,
                arguments.pause_ms,
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
            del embedding
            if arguments.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    index = {
        "schema_version": 1,
        "cache_type": "page3_region_embeddings",
        "encoder": arguments.encoder,
        "variant": "B",
        "roles": ROLES,
        "role_mode": "single_class",
        "link_mode": "row",
        "embedding_dimension": model.encoder.config.hidden_size,
        "chunk_streaming": True,
        "amp": use_amp,
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
