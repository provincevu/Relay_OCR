# Mục đích: chia annotations.json thành các tập train, validation và test.
# Chức năng: xáo trộn các phiếu theo seed và ghi ba dataset cùng schema đầu vào.
# Luồng xử lý: đọc dataset → chia theo tỷ lệ → ghi dataset_train.json, dataset_val.json, dataset_test.json.
# Cấu hình/parameter: tỷ lệ train/val/test và seed nhận từ command line.
# Quan hệ: đầu ra được dùng bởi scripts.training.train_page3_text_layout.
# Input/output: nhận annotations.json; ghi ba file JSON vào thư mục output.
# Contract: giữ schema_version, khóa pages và toàn bộ nội dung từng phiếu; chỉ thay đổi tập phiếu.

import argparse
import json
import random
from pathlib import Path


def split_pages(dataset, train_ratio, val_ratio, test_ratio, seed):
    """Chia các phiếu thành ba tập theo seed cố định."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Tổng train_ratio, val_ratio và test_ratio phải bằng 1.")

    pages = list(dataset["pages"].items())
    random.Random(seed).shuffle(pages)
    train_end = int(len(pages) * train_ratio)
    val_end = train_end + int(len(pages) * val_ratio)
    return {
        "train": pages[:train_end],
        "val": pages[train_end:val_end],
        "test": pages[val_end:],
    }


def write_dataset(path, schema_version, pages):
    """Ghi một tập dữ liệu và giữ nguyên nội dung từng phiếu."""
    path.write_text(
        json.dumps(
            {"schema_version": schema_version, "pages": dict(pages)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    """Đọc annotations.json và tạo ba file dataset."""
    parser = argparse.ArgumentParser(
        description="Chia annotations.json thành train/val/test theo phiếu."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    dataset = json.loads(arguments.dataset.read_text(encoding="utf-8"))
    splits = split_pages(
        dataset,
        arguments.train_ratio,
        arguments.val_ratio,
        arguments.test_ratio,
        arguments.seed,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": arguments.output / "dataset_train.json",
        "val": arguments.output / "dataset_val.json",
        "test": arguments.output / "dataset_test.json",
    }
    for name, pages in splits.items():
        write_dataset(paths[name], dataset["schema_version"], pages)
        print(f"{name}: {len(pages)} phiếu → {paths[name]}")


if __name__ == "__main__":
    main()
