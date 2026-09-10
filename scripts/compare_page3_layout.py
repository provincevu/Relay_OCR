# Mục đích: huấn luyện và so sánh hai pipeline LayoutLMv3 trên page 3.
# Chức năng: A dùng ảnh + text + bbox; B chỉ dùng text + bbox.
# Luồng xử lý: ocr tùy chọn → train A/B → compare trên cùng tập giữ lại.
# Cấu hình/parameter: chạy python -m scripts.compare_page3_layout --help.
# Quan hệ: dùng page3_layout_data, page3_layout_model và manifest PaddleOCR.
# Input/output: manifest/annotation → checkpoint, metrics JSON và HTML đối chiếu.
# Contract: cùng seed, tập train/test và head; không inference bằng head chưa học.
# Contract: F1 đo region và cạnh có hướng, chưa đo cặp thực thể gộp nhiều region.
# Contract: HEADER chỉ có khi --header-labels chứa nhãn tường minh cho mọi page.

import argparse
import hashlib
import json
import os
import random
from html import escape
from pathlib import Path
from time import perf_counter

from scripts.page3_layout_data import (
    PROJECT_DIRECTORY,
    label_pages,
    load_pages,
    recognize_pages,
    write_json,
)


DEFAULT_MANIFEST = PROJECT_DIRECTORY / "data/page3_gpt_input/manifest.json"
DEFAULT_ANNOTATIONS = PROJECT_DIRECTORY / "data/page3_ground_truth_generated.json"
DEFAULT_OUTPUT = PROJECT_DIRECTORY / "output/page3_layout_comparison"


def parse_arguments():
    """Đọc lệnh OCR, huấn luyện hoặc so sánh hai checkpoint đã học."""
    parser = argparse.ArgumentParser(
        description="So sánh nhãn và liên kết region của LayoutLMv3 A/B.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ocr = commands.add_parser(
        "ocr",
        help="OCR ảnh page 3 đã render; cần annotation theo ID mới",
    )
    ocr.add_argument("image_directory", type=Path)
    ocr.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "ocr_manifest.json",
    )
    train = commands.add_parser("train", help="Fine-tune hai nhánh từ cùng dữ liệu")
    compare = commands.add_parser("compare", help="Đánh giá trên tập giữ lại")
    for command in [train, compare]:
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
        command.add_argument("--header-labels", type=Path)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--device", default="cpu")
    train.add_argument("--encoder", default="microsoft/layoutlmv3-base")
    train.add_argument("--variant", choices=["A", "B", "both"], default="both")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--limit",
        type=int,
        help="Chỉ lấy N page đầu để kiểm tra luồng; bỏ qua khi so sánh chính thức",
    )
    compare.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def dataset_fingerprint(arguments):
    """Nhận diện chính xác dữ liệu đã dùng để chia tập và huấn luyện."""
    paths = [arguments.manifest, arguments.annotations]
    if arguments.header_labels is not None:
        paths.append(arguments.header_labels)
    return [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]


def train_models(arguments):
    """Huấn luyện hai nhánh độc lập với cùng nhãn và cách chia page."""
    import torch

    from scripts.page3_layout_model import (
        encode_page,
        load_preprocessors,
        load_pretrained,
        training_loss,
    )

    pages = load_pages(arguments.manifest)
    if arguments.limit is not None:
        pages = pages[:arguments.limit]
    roles = label_pages(pages, arguments.annotations, arguments.header_labels)
    if len(pages) < 2 or not 0 < arguments.test_fraction < 1:
        raise ValueError("Cần ít nhất hai page và test-fraction nằm giữa 0 và 1.")
    if arguments.epochs < 1:
        raise ValueError("Cần huấn luyện ít nhất một epoch trước khi so sánh.")
    ordered_pages = sorted(pages, key=lambda page: page["image_name"])
    random.Random(arguments.seed).shuffle(ordered_pages)
    test_count = max(
        1,
        min(len(pages) - 1, round(len(pages) * arguments.test_fraction)),
    )
    test_pages = ordered_pages[:test_count]
    train_pages = ordered_pages[test_count:]
    tokenizer, image_processor = load_preprocessors(arguments.encoder)
    variants = ["A", "B"] if arguments.variant == "both" else [arguments.variant]
    for variant in variants:
        torch.manual_seed(arguments.seed)
        model = load_pretrained(
            arguments.encoder,
            variant,
            len(roles),
            arguments.seed,
        ).to(arguments.device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=arguments.learning_rate,
        )
        print(
            f"Nhánh {variant}: {len(train_pages)} page train, "
            f"{len(test_pages)} page test; nhãn {roles}",
            flush=True,
        )
        for epoch in range(arguments.epochs):
            epoch_pages = list(train_pages)
            random.Random(arguments.seed + epoch).shuffle(epoch_pages)
            for page_index, page in enumerate(epoch_pages, start=1):
                encoded = encode_page(
                    page,
                    tokenizer,
                    image_processor,
                    variant,
                    arguments.device,
                )
                optimizer.zero_grad(set_to_none=True)
                role_logits, link_logits = model(encoded, len(page["regions"]))
                loss = training_loss(role_logits, link_logits, page)
                loss.backward()
                optimizer.step()
                print(
                    f"{variant} · epoch {epoch + 1}/{arguments.epochs} · "
                    f"{page_index}/{len(epoch_pages)} · {page['image_name']} · "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        directory = arguments.output / variant
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(directory)
        image_processor.save_pretrained(directory)
        torch.save(model.state_dict(), directory / "model.pt")
        write_json(directory / "metadata.json", {
            "variant": variant,
            "encoder": arguments.encoder,
            "encoder_config": model.encoder.config.to_dict(),
            "roles": roles,
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "learning_rate": arguments.learning_rate,
            "fingerprint": dataset_fingerprint(arguments),
            "train_pages": [page["image_name"] for page in train_pages],
            "test_pages": [page["image_name"] for page in test_pages],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        })
        del optimizer, model, loss, role_logits, link_logits, encoded
        if torch.device(arguments.device).type == "cuda":
            torch.cuda.empty_cache()


def classification_counts(expected, predicted):
    """Đếm đúng, dư và thiếu trên tập nhãn hoặc cạnh."""
    return [
        len(expected & predicted),
        len(predicted - expected),
        len(expected - predicted),
    ]


def summarize_counts(counts):
    """Tính precision, recall và F1; mẫu số rỗng có giá trị 0."""
    true_positive, false_positive, false_negative = counts
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / max(denominator, 1),
    }


def render_page(page, prediction, image_url):
    """Vẽ nhãn và cạnh lên ảnh gốc để đối chiếu từng trang."""
    colors = {
        "KEY": "#15803d",
        "VALUE": "#0369a1",
        "HEADER": "#a16207",
        "OTHER": "#64748b",
    }
    region_by_id = {region["id"]: region for region in prediction["regions"]}
    shapes = []
    for link in prediction["links"]:
        key_box = region_by_id[link["key_region_id"]]["bbox"]
        value_box = region_by_id[link["value_region_id"]]["bbox"]
        shapes.append(
            f'<line x1="{(key_box[0] + key_box[2]) / 2}" '
            f'y1="{(key_box[1] + key_box[3]) / 2}" '
            f'x2="{(value_box[0] + value_box[2]) / 2}" '
            f'y2="{(value_box[1] + value_box[3]) / 2}" '
            'stroke="#c026d3" stroke-width="5" />'
        )
    for region in prediction["regions"]:
        left, top, right, bottom = region["bbox"]
        label = "/".join(region["labels"])
        color = colors[region["labels"][0]]
        title = escape(f"{region['id']} · {label} · {region['text']}")
        shapes.append(
            f'<g><title>{title}</title><rect x="{left}" y="{top}" '
            f'width="{right - left}" height="{bottom - top}" '
            f'fill="none" stroke="{color}" stroke-width="4" />'
            f'<text x="{left}" y="{max(top - 5, 22)}" fill="{color}" '
            f'font-size="24">{escape(region["id"])} {label}</text></g>'
        )
    link_rows = "".join(
        f'<tr><td>{escape(link["key_region_id"])}</td>'
        f'<td>{escape(link["value_region_id"])}</td>'
        f'<td>{link["score"]:.3f}</td></tr>'
        for link in prediction["links"]
    )
    return (
        f'<div class="page"><img src="{escape(image_url, quote=True)}">'
        f'<svg viewBox="0 0 {page["image_width"]} {page["image_height"]}">'
        f'{"".join(shapes)}</svg></div>'
        '<details><summary>Liên kết region</summary><table>'
        '<tr><th>Key ID</th><th>Value ID</th><th>Điểm</th></tr>'
        f'{link_rows}</table></details>'
    )


def write_comparison_html(output_path, pages, results):
    """Xuất ảnh hai nhánh cạnh nhau cùng số đo trên tập giữ lại."""
    sections = []
    for index, page in enumerate(pages):
        image_url = Path(os.path.relpath(
            page["image_path"],
            output_path.parent,
        )).as_posix()
        columns = []
        for variant in ["A", "B"]:
            prediction = results[variant]["pages"][index]
            columns.append(
                f'<article><h3>Nhánh {variant} · {prediction["seconds"]:.3f}s</h3>'
                + render_page(page, prediction, image_url)
                + '</article>'
            )
        sections.append(
            f'<section><h2>{escape(page["image_name"])}</h2>'
            f'<div class="columns">{"".join(columns)}</div></section>'
        )
    summaries = []
    for variant in ["A", "B"]:
        metrics = results[variant]["metrics"]
        summaries.append(
            f'<tr><td>{variant}</td><td>{metrics["roles_micro"]["f1"]:.3f}</td>'
            f'<td>{metrics["links"]["precision"]:.3f}</td>'
            f'<td>{metrics["links"]["recall"]:.3f}</td>'
            f'<td>{metrics["links"]["f1"]:.3f}</td>'
            f'<td>{metrics["mean_seconds"]:.3f}</td></tr>'
        )
    output_path.write_text(
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<title>Đối chiếu LayoutLMv3 page 3</title><style>'
        'body{font-family:Arial;margin:24px;background:#f1f5f9;color:#172033}'
        '.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}'
        'article{min-width:0;background:white;padding:12px}'
        '.page{position:relative}.page img{width:100%;display:block}'
        'svg{position:absolute;inset:0;width:100%;height:100%}'
        'table{border-collapse:collapse}td,th{border:1px solid #94a3b8;padding:8px}'
        '</style><h1>LayoutLMv3: ảnh + text + bbox so với text + bbox</h1>'
        '<p>A dùng ảnh; B bỏ nhánh ảnh. Cả hai dùng cùng PaddleOCR và tập giữ lại.</p>'
        '<p>F1 tính theo region OCR và cạnh có hướng, không phải cặp thực thể gộp. '
        'Thời gian gồm chuẩn bị đầu vào và inference, không gồm OCR hoặc tải model.</p>'
        '<p>Cạnh cùng ID biểu thị inline, chưa tách nội dung trong region. '
        'Tập giữ lại chia theo page, chưa tách theo họ mẫu phiếu. '
        'Điểm số phụ thuộc độ đúng và đầy đủ của annotation.</p>'
        '<p>Màu xanh lá: KEY; xanh dương: VALUE; vàng: HEADER nếu có nhãn; '
        'xám: OTHER; tím: liên kết. Di chuột vào bbox để xem text.</p>'
        '<table><tr><th>Nhánh</th><th>F1 nhãn</th><th>Precision cạnh</th>'
        '<th>Recall cạnh</th><th>F1 cạnh</th><th>Giây/page</th></tr>'
        + "".join(summaries) + '</table>' + "".join(sections) + '</html>',
        encoding="utf-8",
    )


def compare_models(arguments):
    """Đánh giá hai checkpoint đã fine-tune trên cùng page chưa dùng để học."""
    import torch
    from transformers import LayoutLMv3Config, LayoutLMv3Model

    from scripts.page3_layout_model import (
        Page3LayoutModel,
        encode_page,
        load_preprocessors,
        predict_page,
    )

    metadata = {
        variant: json.loads(
            (arguments.output / variant / "metadata.json").read_text(encoding="utf-8"),
        )
        for variant in ["A", "B"]
    }
    for field in [
        "encoder", "roles", "seed", "epochs", "learning_rate",
        "fingerprint", "train_pages", "test_pages",
    ]:
        if metadata["A"][field] != metadata["B"][field]:
            raise ValueError(f"Hai nhánh không cùng điều kiện so sánh: {field}")
    if dataset_fingerprint(arguments) != metadata["A"]["fingerprint"]:
        raise ValueError("Manifest hoặc annotation đã khác dữ liệu dùng khi train.")
    pages_by_name = {
        page["image_name"]: page for page in load_pages(arguments.manifest)
    }
    pages = [pages_by_name[name] for name in metadata["A"]["test_pages"]]
    roles = label_pages(pages, arguments.annotations, arguments.header_labels)
    results = {}
    for variant in ["A", "B"]:
        directory = arguments.output / variant
        config = LayoutLMv3Config(**metadata[variant]["encoder_config"])
        model = Page3LayoutModel(LayoutLMv3Model(config), len(roles))
        model.load_state_dict(torch.load(
            directory / "model.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.to(arguments.device).eval()
        tokenizer, image_processor = load_preprocessors(directory)
        predictions = []
        counts = {role: [0, 0, 0] for role in roles}
        counts["links"] = [0, 0, 0]
        with torch.inference_mode():
            warmup = encode_page(
                pages[0],
                tokenizer,
                image_processor,
                variant,
                arguments.device,
            )
            model(warmup, len(pages[0]["regions"]))
            for page in pages:
                if torch.device(arguments.device).type == "cuda":
                    torch.cuda.synchronize()
                started = perf_counter()
                encoded = encode_page(
                    page,
                    tokenizer,
                    image_processor,
                    variant,
                    arguments.device,
                )
                prediction = predict_page(
                    model,
                    encoded,
                    page,
                    roles,
                    arguments.threshold,
                )
                if torch.device(arguments.device).type == "cuda":
                    torch.cuda.synchronize()
                prediction["seconds"] = perf_counter() - started
                prediction["image_name"] = page["image_name"]
                predictions.append(prediction)
                for role_index, role in enumerate(roles):
                    expected = {
                        region["id"]
                        for index, region in enumerate(page["regions"])
                        if page["role_labels"][index][role_index]
                    }
                    predicted = {
                        region["id"] for region in prediction["regions"]
                        if role in region["labels"]
                    }
                    counts[role] = [
                        total + current for total, current in zip(
                            counts[role],
                            classification_counts(expected, predicted),
                            strict=True,
                        )
                    ]
                expected_links = {
                    (page["regions"][key]["id"], page["regions"][value]["id"])
                    for key, value in page["link_labels"]
                }
                predicted_links = {
                    (link["key_region_id"], link["value_region_id"])
                    for link in prediction["links"]
                }
                counts["links"] = [
                    total + current for total, current in zip(
                        counts["links"],
                        classification_counts(expected_links, predicted_links),
                        strict=True,
                    )
                ]
                print(f"Đã đánh giá {variant}: {page['image_name']}", flush=True)
        metrics = {name: summarize_counts(value) for name, value in counts.items()}
        metrics["roles_micro"] = summarize_counts([
            sum(counts[role][index] for role in roles) for index in range(3)
        ])
        metrics["mean_seconds"] = (
            sum(page["seconds"] for page in predictions) / len(predictions)
        )
        metrics["parameter_count"] = metadata[variant]["parameter_count"]
        results[variant] = {"metrics": metrics, "pages": predictions}
        del model, warmup, encoded
        if torch.device(arguments.device).type == "cuda":
            torch.cuda.empty_cache()
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output / "comparison.json", {
        "evaluation_unit": "region OCR và cạnh có hướng; không phải cặp thực thể gộp",
        "split_unit": "page; chưa tách theo họ mẫu phiếu",
        "threshold": arguments.threshold,
        "roles": roles,
        "fingerprint": metadata["A"]["fingerprint"],
        "results": results,
    })
    html_path = arguments.output / "comparison.html"
    write_comparison_html(html_path, pages, results)
    print(f"Kết quả đối chiếu: {html_path}", flush=True)


def main():
    """Chạy đúng bước được yêu cầu trong thử nghiệm A/B."""
    arguments = parse_arguments()
    if arguments.command == "ocr":
        recognize_pages(arguments.image_directory, arguments.output)
    elif arguments.command == "train":
        train_models(arguments)
    else:
        compare_models(arguments)


if __name__ == "__main__":
    main()
