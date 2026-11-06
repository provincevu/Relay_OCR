# Mục đích: fine-tune LayoutLMv3 chỉ dùng text + bbox từ annotation page 3.
# Chức năng: học một role cho mỗi region; liên kết KEY → VALUE dùng row-link.
# Luồng xử lý: phiếu đã xác nhận → mã hóa region → role loss → checkpoint tốt nhất;
# hoặc embedding cache → role head → checkpoint.
# Cấu hình/parameter: python -m scripts.training.train_page3_text_layout --help;
# hỗ trợ AMP, freeze encoder và gradient checkpointing.
# Quan hệ: dùng annotations.json của serve_page3_annotations và model B hiện có.
# Input/output: train nhận dataset đã chia bên ngoài; train-cached nhận embedding cache;
# predict nhận manifest OCR.
# Contract: INLINE_KEY_VALUE là role single class; UNLABELED và inline KEY VALUE legacy không là target.
# Contract: không có learned-link head/loss; relation production là row-link.
# Contract: không chia train/test/val; visual_embed=False, không phải bản distilled.

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3Model,
    LayoutLMv3TokenizerFast,
)

from scripts.training.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    load_pretrained,
    predict_page,
)
from scripts.inference.page_n_prediction_filter import (
    filter_prediction_pages,
)
from scripts.inference.role_change_metadata import write_role_change_metadata
from scripts.training.metrics import (
    ROLES,
    cached_validation_metrics as _cached_validation_metrics,
    is_trainable_annotation as _is_trainable_annotation,
    masked_training_loss,
    validation_metrics,
)
from scripts.training.training_utils import (
    create_scheduler,
    optimizer_parameters,
    run_training_epochs,
    save_checkpoint,
)


def prepare_page(record):
    """Đổi tên bbox nội bộ, không thay đổi snapshot và nhóm ID trong dataset."""
    return {
        **record,
        "regions": [
            {**region, "bbox": region["box"]}
            for region in record["regions"]
        ],
    }


def _load_embedding_pages(cache_path):
    """Đọc index và các page embedding đã tạo từ cache."""
    index_path = cache_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("roles") != ROLES:
        raise ValueError("Cache embedding không khớp danh sách role hiện tại.")
    pages = [
        torch.load(
            cache_path / item["file"],
            map_location="cpu",
            weights_only=True,
        )
        for item in index["pages"]
    ]
    if not pages:
        raise ValueError("Cache embedding không có page.")
    return index, pages


def _cached_role_logits(role_head, dropout, page, device):
    """Tính role logits từ embedding cache của một page."""
    embeddings = page["region_embeddings"].to(device)
    return role_head(dropout(embeddings)), embeddings


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
    use_amp = arguments.amp and str(arguments.device).startswith("cuda")
    model = load_pretrained(
        arguments.encoder,
        "B",
        len(ROLES),
        arguments.seed,
        arguments.head_dropout,
        arguments.gradient_checkpointing and not arguments.freeze_encoder,
    ).to(arguments.device)
    if arguments.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        optimizer_parameters(model, arguments.weight_decay),
        lr=arguments.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = create_scheduler(
        optimizer,
        arguments.epochs,
        len(pages),
        arguments.warmup_ratio,
    )
    def train_epoch():
        model.train()
        if arguments.freeze_encoder:
            model.encoder.eval()
        random.shuffle(pages)
        total_loss = 0.0
        for page in pages:
            encoded = encode_page(page, tokenizer, None, "B", arguments.device)
            optimizer.zero_grad()
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                role_logits = model(encoded, len(page["regions"]))
                loss = masked_training_loss(role_logits, page)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                arguments.max_grad_norm,
            )
            if use_amp:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = scaler.get_scale() >= previous_scale
            else:
                optimizer.step()
                optimizer_updated = True
            if optimizer_updated:
                scheduler.step()
            total_loss += loss.detach().item()
        return total_loss / len(pages)

    def validate():
        return validation_metrics(
            validation_pages,
            model,
            tokenizer,
            arguments.device,
            arguments.threshold,
        )

    early_stopping = run_training_epochs(
        arguments.epochs,
        {"model": model},
        train_epoch,
        validate,
        arguments.early_stopping_patience,
        arguments.early_stopping_min_delta,
    )

    early_stopping.restore_best({"model": model})
    save_checkpoint(
        arguments.output,
        tokenizer,
        model,
        early_stopping.best_epoch,
        early_stopping.best_metrics,
        arguments.head_dropout,
    )
    print(
        f"Đã lưu best checkpoint text + bbox: {arguments.output} "
        f"(epoch {early_stopping.best_epoch}, "
        f"val_f1={early_stopping.best_score:.4f})"
    )


def train_cached(arguments):
    """Train role head trên embedding cache mà không chạy lại encoder."""
    train_index, train_pages = _load_embedding_pages(arguments.dataset)
    validation_index, validation_pages = _load_embedding_pages(
        arguments.validation_dataset,
    )
    if train_index["embedding_dimension"] != validation_index["embedding_dimension"]:
        raise ValueError("Train và validation cache khác embedding dimension.")
    encoder_config = LayoutLMv3Config.from_pretrained(arguments.encoder)
    if encoder_config.hidden_size != train_index["embedding_dimension"]:
        raise ValueError("Embedding cache không khớp hidden size của encoder.")
    if arguments.epochs <= 0:
        raise ValueError("epochs phải lớn hơn 0.")
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

    torch.manual_seed(arguments.seed)
    embedding_dimension = train_index["embedding_dimension"]
    role_head = torch.nn.Linear(
        embedding_dimension,
        len(ROLES),
    ).to(arguments.device)
    dropout = torch.nn.Dropout(arguments.head_dropout).to(arguments.device)
    optimizer = torch.optim.AdamW(
        list(role_head.parameters()),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        arguments.epochs,
        len(train_pages),
        arguments.warmup_ratio,
    )
    def train_epoch():
        role_head.train()
        dropout.train()
        total_loss = 0.0
        for page in train_pages:
            optimizer.zero_grad()
            role_logits, embeddings = _cached_role_logits(
                role_head,
                dropout,
                page,
                arguments.device,
            )
            loss = masked_training_loss(role_logits, page)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                role_head.parameters(),
                arguments.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().item()
            del role_logits, embeddings, loss
        return total_loss / len(train_pages)

    def validate():
        return _cached_validation_metrics(
            validation_pages,
            role_head,
            dropout,
            arguments.device,
            _cached_role_logits,
        )

    early_stopping = run_training_epochs(
        arguments.epochs,
        {"role_head": role_head},
        train_epoch,
        validate,
        arguments.early_stopping_patience,
        arguments.early_stopping_min_delta,
    )

    early_stopping.restore_best({"role_head": role_head})

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(arguments.encoder)
    base_model = load_pretrained(
        arguments.encoder,
        "B",
        len(ROLES),
        arguments.seed,
        arguments.head_dropout,
    )
    base_model.role_head.load_state_dict(role_head.state_dict())
    save_checkpoint(
        arguments.output,
        tokenizer,
        base_model,
        early_stopping.best_epoch,
        early_stopping.best_metrics,
        arguments.head_dropout,
    )
    print(
        f"Đã lưu best checkpoint từ embedding cache: {arguments.output} "
        f"(epoch {early_stopping.best_epoch}, "
        f"val_f1={early_stopping.best_score:.4f})",
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
            # OTHER không đi cùng role chính; dữ liệu inline legacy vẫn giữ tương thích.
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
    filter_prediction_pages(predictions)
    for prediction in predictions:
        prediction = add_link_texts(prediction)
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
    write_role_change_metadata(arguments.output.parent)
    print(f"Đã lưu dự đoán {len(predictions)} phiếu: {arguments.output}")


def _add_training_arguments(parser):
    """Đăng ký một lần các tham số dùng chung cho hai chế độ train."""
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)


def main():
    """Đọc lệnh train/predict; tập dữ liệu được người dùng chia bằng script riêng."""
    parser = argparse.ArgumentParser(description="LayoutLMv3 text + bbox, không dùng ảnh.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Fine-tune từ annotations.json đã xác nhận")
    _add_training_arguments(train_parser)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument(
        "--amp",
        action="store_true",
        help="Dùng mixed precision float16 khi train trên CUDA.",
    )
    train_parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Đóng băng LayoutLMv3 và chỉ train role head.",
    )
    train_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Bật gradient checkpointing cho LayoutLMv3 encoder.",
    )
    cached_train_parser = commands.add_parser(
        "train-cached",
        help="Train role head từ embedding cache đã tạo.",
    )
    _add_training_arguments(cached_train_parser)
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
    for command in [train_parser, cached_train_parser, predict_parser]:
        command.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments)
    elif arguments.command == "train-cached":
        train_cached(arguments)
    else:
        predict(arguments)


if __name__ == "__main__":
    main()
