# Mục đích: fine-tune LayoutLMv3 chỉ dùng text + bbox từ annotation page 3.
# Chức năng: học một role cho mỗi region; liên kết KEY → VALUE dùng row-link.
# Luồng xử lý: phiếu đã xác nhận → mã hóa region → role loss → checkpoint tốt nhất.
# Cấu hình/parameter: python -m scripts.train_page3_text_layout --help.
# Quan hệ: dùng annotations.json của serve_page3_annotations và model B hiện có.
# Input/output: train nhận dataset đã chia bên ngoài; predict nhận manifest OCR.
# Contract: UNLABELED và inline KEY VALUE legacy không là target role single class.
# Contract: không có learned-link head/loss; relation production là row-link.
# Contract: không chia train/test/val; visual_embed=False, không phải bản distilled.

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3Model,
    LayoutLMv3TokenizerFast,
    get_linear_schedule_with_warmup,
)

from scripts.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    load_pretrained,
    predict_page,
)


ROLES = [
    "KEY",
    "VALUE",
    "HEADER",
    "FIELD_TABLE",
    "NOTE",
    "PAGE_NUMBER",
    "FORM_NUMBER",
    "OTHER",
]


def prepare_page(record):
    """Đổi tên bbox nội bộ, không thay đổi snapshot và nhóm ID trong dataset."""
    return {
        **record,
        "regions": [
            {**region, "bbox": region["box"]}
            for region in record["regions"]
        ],
    }


def _is_trainable_annotation(labels):
    """Kiểm tra annotation có đúng một role để đưa vào single-class loss."""
    return "UNLABELED" not in labels and set(labels) != {"KEY", "VALUE"}


def masked_training_loss(role_logits, page):
    """Tính single-class role loss và bỏ qua nhãn không thể biểu diễn."""
    region_ids = [region["id"] for region in page["regions"]]
    labels = [page["annotations"][region_id] for region_id in region_ids]
    target_indices = []
    known_values = []
    for labels_for_region in labels:
        labels_set = set(labels_for_region)
        known_values.append(_is_trainable_annotation(labels_set))
        target_role = next(
            (role for role in ROLES if role in labels_set),
            "OTHER",
        )
        target_indices.append(ROLES.index(target_role))
    known = torch.tensor(
        known_values,
        dtype=torch.bool,
        device=role_logits.device,
    )
    role_targets = torch.tensor(
        target_indices,
        dtype=torch.long,
        device=role_logits.device,
    )
    if not known.any():
        return role_logits.sum() * 0
    return torch.nn.functional.cross_entropy(
        role_logits[known],
        role_targets[known],
    )


def _update_counts(counts, gold, predicted):
    """Cộng TP/FP/FN của hai tập ID."""
    counts[0] += len(gold & predicted)
    counts[1] += len(predicted - gold)
    counts[2] += len(gold - predicted)


def _f1_from_counts(counts):
    """Tính F1 từ TP/FP/FN."""
    true_positive, false_positive, false_negative = counts
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    return 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0


def _normalize_prediction_labels(labels):
    """Không cho OTHER đi cùng role chính trong metric."""
    labels = set(labels)
    if labels.intersection(set(ROLES) - {"OTHER"}):
        labels.discard("OTHER")
    return labels or {"OTHER"}


def validation_metrics(pages, model, tokenizer, device, threshold):
    """Đánh giá role F1 và row-link F1 để early stopping."""
    role_counts = {role: [0, 0, 0] for role in ["KEY", "VALUE"]}
    row_link_counts = [0, 0, 0]
    model.eval()
    with torch.inference_mode():
        for page in pages:
            encoded = encode_page(page, tokenizer, None, "B", device)
            prediction = predict_page(
                model,
                encoded,
                page,
                ROLES,
                threshold,
                "row",
            )
            labels = page["annotations"]
            known_ids = {
                region["id"]
                for region in page["regions"]
                if _is_trainable_annotation(labels[region["id"]])
            }
            predicted_labels = {
                region["id"]: _normalize_prediction_labels(region["labels"])
                for region in prediction["regions"]
            }
            for role in role_counts:
                gold = {
                    region_id
                    for region_id in known_ids
                    if role in labels[region_id]
                }
                predicted = {
                    region_id
                    for region_id in known_ids
                    if role in predicted_labels[region_id]
                }
                _update_counts(role_counts[role], gold, predicted)

            gold_links = {
                (key_id, value_id)
                for relation in page["relations"]
                for key_id in relation["key_region_ids"]
                for value_id in relation["value_region_ids"]
                if key_id in known_ids and value_id in known_ids
            }
            predicted_links = {
                (link["key_region_id"], link["value_region_id"])
                for link in prediction["links"]
                if link["key_region_id"] in known_ids
                and link["value_region_id"] in known_ids
            }
            _update_counts(row_link_counts, gold_links, predicted_links)

    key_f1 = _f1_from_counts(role_counts["KEY"])
    value_f1 = _f1_from_counts(role_counts["VALUE"])
    row_link_f1 = _f1_from_counts(row_link_counts)
    role_f1 = (key_f1 + value_f1) / 2
    val_f1 = 0.5 * role_f1 + 0.5 * row_link_f1
    model.train()
    return {
        "key_f1": key_f1,
        "value_f1": value_f1,
        "role_f1": role_f1,
        "row_link_f1": row_link_f1,
        "val_f1": val_f1,
    }


def _load_pages(dataset_path):
    """Đọc các page đã xác nhận và có annotation dùng được."""
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        prepare_page(record)
        for record in dataset["pages"].values()
        if record["confirmed_at"]
        and any(
            _is_trainable_annotation(labels)
            for labels in record["annotations"].values()
        )
    ]


def _optimizer_parameters(model, weight_decay):
    """Tách parameter decay và no-decay theo quy tắc AdamW chuẩn."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _save_checkpoint(output, tokenizer, model, epoch, metrics, head_dropout):
    """Lưu tokenizer và checkpoint tốt nhất hiện tại."""
    if (output / "model.pt").exists():
        raise FileExistsError(
            f"Checkpoint đã tồn tại: {output / 'model.pt'}. "
            "Hãy chọn thư mục output mới cho model single class."
        )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    torch.save(
        {
            "roles": ROLES,
            "role_mode": "single_class",
            "encoder_config": model.encoder.config.to_dict(),
            "model_state": model.state_dict(),
            "head_dropout": head_dropout,
            "link_mode": "row",
            "best_epoch": epoch,
            "best_val_metrics": metrics,
        },
        output / "model.pt",
    )


def train(arguments):
    """Train role heads, chọn checkpoint theo row-link validation F1."""
    pages = _load_pages(arguments.dataset)
    validation_pages = _load_pages(arguments.validation_dataset)
    if not pages:
        raise ValueError("Dataset chưa có phiếu đã xác nhận với nhãn dùng được.")
    if not validation_pages:
        raise ValueError("Validation dataset chưa có phiếu đã xác nhận dùng được.")
    if not 0 <= arguments.warmup_ratio <= 1:
        raise ValueError("warmup_ratio phải nằm trong khoảng 0 đến 1.")
    if arguments.weight_decay < 0:
        raise ValueError("weight_decay không được âm.")
    if arguments.max_grad_norm <= 0:
        raise ValueError("max_grad_norm phải lớn hơn 0.")
    if arguments.head_dropout < 0 or arguments.head_dropout >= 1:
        raise ValueError("head_dropout phải nằm trong khoảng [0, 1).")
    if arguments.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience phải ít nhất là 1.")
    if arguments.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta không được âm.")
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.encoder)
    model = load_pretrained(
        arguments.encoder,
        "B",
        len(ROLES),
        arguments.seed,
        arguments.head_dropout,
    ).to(arguments.device)
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, arguments.weight_decay),
        lr=arguments.learning_rate,
    )
    total_steps = arguments.epochs * len(pages)
    warmup_steps = int(total_steps * arguments.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    best_val_f1 = float("-inf")
    best_state = None
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    for epoch in range(arguments.epochs):
        model.train()
        random.shuffle(pages)
        total_loss = 0.0
        for page in pages:
            encoded = encode_page(page, tokenizer, None, "B", arguments.device)
            optimizer.zero_grad()
            role_logits = model(encoded, len(page["regions"]))
            loss = masked_training_loss(role_logits, page)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                arguments.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().item()
        metrics = validation_metrics(
            validation_pages,
            model,
            tokenizer,
            arguments.device,
            arguments.threshold,
        )
        average_loss = total_loss / len(pages)
        print(
            f"Epoch {epoch + 1}/{arguments.epochs}: "
            f"loss={average_loss:.6f}, "
            f"val_f1={metrics['val_f1']:.4f}, "
            f"KEY={metrics['key_f1']:.4f}, "
            f"VALUE={metrics['value_f1']:.4f}, "
            f"row-link={metrics['row_link_f1']:.4f}",
            flush=True,
        )
        if metrics["val_f1"] > best_val_f1 + arguments.early_stopping_min_delta:
            best_val_f1 = metrics["val_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= arguments.early_stopping_patience:
                print(
                    f"Early stopping tại epoch {epoch + 1}; "
                    f"best epoch={best_epoch}, val_f1={best_val_f1:.4f}",
                    flush=True,
                )
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Không tạo được checkpoint tốt nhất theo validation F1.")
    model.load_state_dict(best_state)
    _save_checkpoint(
        arguments.output,
        tokenizer,
        model,
        best_epoch,
        best_metrics,
        arguments.head_dropout,
    )
    print(
        f"Đã lưu best checkpoint text + bbox: {arguments.output} "
        f"(epoch {best_epoch}, val_f1={best_val_f1:.4f})"
    )


def predict(arguments):
    """Dự đoán nhãn và các cạnh region theo ID, không dùng annotation làm đầu vào."""
    checkpoint = torch.load(
        arguments.checkpoint / "model.pt",
        map_location=arguments.device,
        weights_only=True,
    )
    config = LayoutLMv3Config.from_dict(checkpoint["encoder_config"])
    model = Page3LayoutModel(
        LayoutLMv3Model(config),
        len(checkpoint["roles"]),
        checkpoint.get("head_dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(arguments.device).eval()
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.checkpoint)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    predictions = []

    def add_link_texts(prediction):
        """Thêm text region để viewer hiển thị link thay vì chỉ hiện ID."""
        region_by_id = {
            region["id"]: region
            for region in prediction["regions"]
        }
        prediction["links"] = [
            {
                **link,
                "key_text": region_by_id[link["key_region_id"]]["text"],
                "value_text": region_by_id[link["value_region_id"]]["text"],
            }
            for link in prediction["links"]
        ]
        return prediction

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
                arguments.link_mode,
            )
            prediction = add_link_texts(prediction)
            # OTHER không đi cùng role chính; KEY/VALUE cùng ID vẫn hợp lệ cho inline.
            for region in prediction["regions"]:
                region["labels"] = [role for role in region["labels"] if role != "OTHER"] or ["OTHER"]
            predictions.append({
                "pdf_name": record["pdfName"],
                "physical_page": record["physicalPage"],
                "image_name": record["imageName"],
                "image_path": record["imageName"],
                "image_width": record["width"],
                "image_height": record["height"],
                "ocr_fingerprint": record["ocrFingerprint"],
                "recognition_source": "annotation manifest",
                **prediction,
            })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "link_mode": "row",
                "threshold": arguments.threshold,
                "checkpoint": str(arguments.checkpoint),
                "device": arguments.device,
                "pages": predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--validation-dataset", type=Path, required=True)
    train_parser.add_argument("--weight-decay", type=float, default=0.03)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--head-dropout", type=float, default=0.3)
    train_parser.add_argument("--early-stopping-patience", type=int, default=5)
    train_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
    )
    train_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser = commands.add_parser("predict", help="Dự đoán từ annotation_manifest.json")
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument(
        "--link-mode",
        choices=["row"],
        default="row",
        help="Cách tạo liên kết KEY → VALUE; pipeline chỉ dùng row-link.",
    )
    for command in [train_parser, predict_parser]:
        command.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        predict(arguments)


if __name__ == "__main__":
    main()
