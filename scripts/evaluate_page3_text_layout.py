# Mục đích: đánh giá checkpoint LayoutLMv3 text + bbox trên dataset page 3+.
# Chức năng: đo nhận diện KEY/VALUE/OTHER và các liên kết KEY → VALUE.
# Luồng xử lý: tải checkpoint → dự đoán từng phiếu → so sánh region/cạnh với nhãn thật → ghi metric.
# Cấu hình/parameter: dataset, checkpoint, device, threshold và output nhận từ command line.
# Quan hệ: dùng chung Page3LayoutModel, tokenizer và encode_page với scripts.train_page3_text_layout.
# Input/output: nhận annotations.json cùng thư mục checkpoint; ghi evaluation JSON.
# Contract: bỏ qua region UNLABELED; liên kết nhiều region được tính theo từng cạnh KEY → VALUE.

import argparse
import json
from pathlib import Path

import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

from scripts.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    predict_page,
)
from scripts.train_page3_text_layout import prepare_page


EVALUATED_ROLES = ["KEY", "VALUE", "OTHER"]


def set_counts(gold, predicted):
    """Tính số đúng, dương giả và âm giả của hai tập ID."""
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    return true_positive, false_positive, false_negative


def metric_from_counts(true_positive, false_positive, false_negative):
    """Đổi số TP/FP/FN thành precision, recall và F1."""
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "support": true_positive + false_negative,
        "predicted": true_positive + false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def normalize_prediction_labels(labels):
    """Giữ OTHER riêng với KEY hoặc VALUE giống pipeline predict."""
    labels = set(labels)
    if labels.intersection({"KEY", "VALUE"}):
        labels.discard("OTHER")
    return labels or {"OTHER"}


def load_model(checkpoint_directory, device):
    """Tải model và tokenizer từ thư mục checkpoint."""
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


def evaluate(arguments):
    """Đánh giá nhận diện region và các liên kết KEY/VALUE."""
    dataset = json.loads(arguments.dataset.read_text(encoding="utf-8"))
    records = [
        record
        for record in dataset["pages"].values()
        if record["confirmed_at"]
    ]
    model, tokenizer, roles = load_model(arguments.checkpoint, arguments.device)
    role_counts = {role: [0, 0, 0] for role in EVALUATED_ROLES}
    link_modes = ["learned", "row"] if arguments.link_mode == "both" \
        else [arguments.link_mode]
    link_counts = {mode: [0, 0, 0] for mode in link_modes}
    ignored_unlabeled = 0

    with torch.inference_mode():
        for record in records:
            page = prepare_page(record)
            labels = page["annotations"]
            known_ids = {
                region["id"]
                for region in page["regions"]
                if "UNLABELED" not in labels[region["id"]]
            }
            ignored_unlabeled += len(page["regions"]) - len(known_ids)
            encoded = encode_page(
                page,
                tokenizer,
                None,
                "B",
                arguments.device,
            )
            prediction = predict_page(
                model,
                encoded,
                page,
                roles,
                arguments.threshold,
                arguments.link_mode,
            )
            predicted_labels = {
                region["id"]: normalize_prediction_labels(region["labels"])
                for region in prediction["regions"]
            }
            for role in EVALUATED_ROLES:
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
                counts = set_counts(gold, predicted)
                role_counts[role] = [
                    left + right
                    for left, right in zip(role_counts[role], counts)
                ]

            gold_links = {
                (key_id, value_id)
                for relation in page["relations"]
                for key_id in relation["key_region_ids"]
                for value_id in relation["value_region_ids"]
                if key_id in known_ids and value_id in known_ids
            }
            links_by_mode = prediction.get(
                "links_by_mode",
                {arguments.link_mode: prediction["links"]},
            )
            for mode in link_modes:
                predicted_links = {
                    (link["key_region_id"], link["value_region_id"])
                    for link in links_by_mode[mode]
                    if link["key_region_id"] in known_ids
                    and link["value_region_id"] in known_ids
                }
                counts = set_counts(gold_links, predicted_links)
                link_counts[mode] = [
                    left + right
                    for left, right in zip(link_counts[mode], counts)
                ]

    role_metrics = {
        role: metric_from_counts(*counts)
        for role, counts in role_counts.items()
    }
    link_metrics_by_mode = {
        mode: metric_from_counts(*counts)
        for mode, counts in link_counts.items()
    }
    result = {
        "dataset": str(arguments.dataset),
        "checkpoint": str(arguments.checkpoint),
        "device": arguments.device,
        "threshold": arguments.threshold,
        "link_mode": arguments.link_mode,
        "pages": len(records),
        "ignored_unlabeled_regions": ignored_unlabeled,
        "region_metrics": role_metrics,
        "key_value_link_metrics_by_mode": link_metrics_by_mode,
    }
    if arguments.link_mode != "both":
        result["key_value_link_metrics"] = link_metrics_by_mode[arguments.link_mode]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã đánh giá {len(records)} phiếu: {arguments.output}")
    for role, metrics in role_metrics.items():
        print(
            f"{role}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
        )
    for mode, metrics in link_metrics_by_mode.items():
        print(
            f"KEY → VALUE [{mode}-link]: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
        )


def main():
    """Đọc tham số và chạy đánh giá checkpoint."""
    parser = argparse.ArgumentParser(
        description="Đánh giá LayoutLMv3 text + bbox trên dataset page 3+.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--link-mode",
        choices=["learned", "row", "both"],
        default="row",
        help="Cách tạo liên kết KEY → VALUE; mặc định dùng row-link.",
    )
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    evaluate(arguments)


if __name__ == "__main__":
    main()
