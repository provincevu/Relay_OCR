# Mục đích: fine-tune LayoutLMv3 chỉ dùng text + bbox từ annotation page 3.
# Chức năng: học CODE/KEY/VALUE/HEADER/OTHER và liên kết KEY → VALUE.
# Luồng xử lý: phiếu đã xác nhận → mã hóa region → loss có mask → checkpoint.
# Cấu hình/parameter: python -m scripts.train_page3_text_layout --help.
# Quan hệ: dùng annotations.json của serve_page3_annotations và model B hiện có.
# Input/output: train nhận dataset đã chia bên ngoài; predict nhận manifest OCR.
# Contract: UNLABELED chỉ làm ngữ cảnh, không là target hoặc cạnh âm/dương.
# Contract: giữ nhóm trong dataset; head liên kết học mọi cạnh giữa hai nhóm.
# Contract: không chia train/test/val; visual_embed=False, không phải bản distilled.

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

from scripts.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    load_pretrained,
    predict_page,
)


ROLES = ["CODE", "KEY", "VALUE", "HEADER", "OTHER"]


def prepare_page(record):
    """Đổi tên bbox nội bộ, không thay đổi snapshot và nhóm ID trong dataset."""
    return {
        **record,
        "regions": [
            {**region, "bbox": region["box"]}
            for region in record["regions"]
        ],
    }


def masked_training_loss(role_logits, link_logits, page):
    """Bỏ qua region UNLABELED và mọi cạnh có ít nhất một đầu chưa gán nhãn."""
    region_ids = [region["id"] for region in page["regions"]]
    labels = [page["annotations"][region_id] for region_id in region_ids]
    known = torch.tensor(
        ["UNLABELED" not in roles for roles in labels],
        dtype=torch.bool,
        device=role_logits.device,
    )
    role_targets = role_logits.new_tensor([
        [float(role in labels_for_region) for role in ROLES]
        for labels_for_region in labels
    ])
    # Tổng rỗng vẫn có gradient bằng 0 khi toàn bộ trang bị bỏ qua.
    role_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        role_logits,
        role_targets,
        reduction="none",
    )
    role_loss = role_losses[known].sum() / (known.sum() * len(ROLES)).clamp(min=1)
    indices = {region_id: index for index, region_id in enumerate(region_ids)}
    link_targets = torch.zeros_like(link_logits)
    for relation in page["relations"]:
        for key_id in relation["key_region_ids"]:
            for value_id in relation["value_region_ids"]:
                link_targets[indices[key_id], indices[value_id]] = 1
    link_mask = known[:, None] & known[None, :]
    positives = link_targets[link_mask].sum()
    positive_weight = ((link_mask.sum() - positives) / positives.clamp(min=1)).clamp(min=1)
    link_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        link_logits,
        link_targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    link_loss = link_losses[link_mask].sum() / link_mask.sum().clamp(min=1)
    return role_loss + link_loss


def train(arguments):
    """Huấn luyện nhánh B trên các phiếu đã xác nhận, lưu tokenizer và hai head."""
    dataset = json.loads(arguments.dataset.read_text(encoding="utf-8"))
    pages = [
        prepare_page(record)
        for record in dataset["pages"].values()
        if record["confirmed_at"]
        and any("UNLABELED" not in labels for labels in record["annotations"].values())
    ]
    if not pages:
        raise ValueError("Dataset chưa có phiếu đã xác nhận với nhãn dùng được.")
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.encoder)
    model = load_pretrained(
        arguments.encoder,
        "B",
        len(ROLES),
        arguments.seed,
    ).to(arguments.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    model.train()
    for epoch in range(arguments.epochs):
        random.shuffle(pages)
        total_loss = 0.0
        for page in pages:
            encoded = encode_page(page, tokenizer, None, "B", arguments.device)
            optimizer.zero_grad()
            role_logits, link_logits = model(encoded, len(page["regions"]))
            loss = masked_training_loss(role_logits, link_logits, page)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        print(f"Epoch {epoch + 1}/{arguments.epochs}: loss = {total_loss / len(pages):.6f}", flush=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(arguments.output)
    torch.save(
        {
            "roles": ROLES,
            "encoder_config": model.encoder.config.to_dict(),
            "model_state": model.state_dict(),
        },
        arguments.output / "model.pt",
    )
    print(f"Đã lưu checkpoint text + bbox: {arguments.output}")


def predict(arguments):
    """Dự đoán nhãn và cạnh KEY → VALUE theo ID, không dùng annotation làm đầu vào."""
    checkpoint = torch.load(
        arguments.checkpoint / "model.pt",
        map_location=arguments.device,
        weights_only=True,
    )
    config = LayoutLMv3Config.from_dict(checkpoint["encoder_config"])
    model = Page3LayoutModel(LayoutLMv3Model(config), len(checkpoint["roles"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(arguments.device).eval()
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.checkpoint)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    predictions = []
    with torch.no_grad():
        for record in manifest["pages"]:
            page = prepare_page({
                "image_width": record["width"],
                "image_height": record["height"],
                "regions": record["regions"],
            })
            encoded = encode_page(page, tokenizer, None, "B", arguments.device)
            prediction = predict_page(
                model,
                encoded,
                page,
                checkpoint["roles"],
                arguments.threshold,
            )
            # OTHER không đi cùng nhãn thực thể; KEY/VALUE cùng ID vẫn hợp lệ cho inline.
            for region in prediction["regions"]:
                region["labels"] = [role for role in region["labels"] if role != "OTHER"] or ["OTHER"]
            predictions.append({
                "pdf_name": record["pdfName"],
                "physical_page": record["physicalPage"],
                "image_name": record["imageName"],
                "ocr_fingerprint": record["ocrFingerprint"],
                **prediction,
            })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"pages": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu dự đoán {len(predictions)} phiếu: {arguments.output}")


def main():
    """Đọc lệnh train/predict; tập dữ liệu được người dùng chia bằng script riêng."""
    parser = argparse.ArgumentParser(description="LayoutLMv3 text + bbox, không dùng ảnh.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Fine-tune từ annotations.json đã xác nhận")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--encoder", default="microsoft/layoutlmv3-base")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    predict_parser = commands.add_parser("predict", help="Dự đoán từ annotation_manifest.json")
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    for command in [train_parser, predict_parser]:
        command.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        predict(arguments)


if __name__ == "__main__":
    main()
