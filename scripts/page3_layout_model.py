# Mục đích: so sánh LayoutLMv3 có ảnh và LayoutLMv3 chỉ dùng text + bbox.
# Chức năng: mã hóa OCR, dự đoán đa nhãn region và liên kết key → value.
# Luồng xử lý: chia token thành chunk → encoder → gộp region → hai head.
# Cấu hình/parameter: variant A/B, checkpoint encoder, danh sách role.
# Quan hệ: dùng chung bởi lệnh train và compare trong compare_page3_layout.
# Input/output: một page OCR → logits [region, role] và [region, region].
# Contract: A dùng ảnh; B visual_embed=False; cùng tokenizer/head/cách pooling.
# Contract: không cắt bỏ token cuối trang; cạnh có thể nối region khác chunk.

import math

import torch
from PIL import Image
from torch import nn
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3TokenizerFast,
)


class Page3LayoutModel(nn.Module):
    """Học vai trò và liên kết region bằng cùng kiến trúc cho hai nhánh."""

    def __init__(self, encoder, role_count):
        """Tạo head đa nhãn và head chấm điểm cạnh có hướng."""
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.role_head = nn.Linear(hidden_size, role_count)
        self.key_head = nn.Linear(hidden_size, 128)
        self.value_head = nn.Linear(hidden_size, 128)

    def forward(self, encoded, region_count):
        """Gộp mọi subtoken theo ID region trước khi chấm điểm liên kết."""
        pooled = None
        counts = None
        for chunk in encoded["chunks"]:
            inputs = {
                name: value
                for name, value in chunk.items()
                if name != "region_indices"
            }
            if "pixel_values" in encoded:
                inputs["pixel_values"] = encoded["pixel_values"]
            hidden = self.encoder(**inputs).last_hidden_state
            text_hidden = hidden[0, :chunk["input_ids"].shape[1]]
            indices = chunk["region_indices"]
            valid = indices >= 0
            if pooled is None:
                pooled = text_hidden.new_zeros((region_count, hidden.shape[-1]))
                counts = text_hidden.new_zeros((region_count, 1))
            pooled = pooled.index_add(0, indices[valid], text_hidden[valid])
            counts = counts.index_add(
                0,
                indices[valid],
                text_hidden.new_ones((int(valid.sum()), 1)),
            )
        region_hidden = pooled / counts
        role_logits = self.role_head(region_hidden)
        key_hidden = torch.tanh(self.key_head(region_hidden))
        value_hidden = torch.tanh(self.value_head(region_hidden))
        link_logits = key_hidden @ value_hidden.T / math.sqrt(128)
        return role_logits, link_logits


def load_pretrained(encoder_path, variant, role_count, seed):
    """Khởi tạo encoder pretrained và head mới cần được huấn luyện."""
    config = LayoutLMv3Config.from_pretrained(encoder_path)
    config.visual_embed = variant == "A"
    encoder = LayoutLMv3Model.from_pretrained(
        encoder_path,
        config=config,
    )
    torch.manual_seed(seed)
    return Page3LayoutModel(encoder, role_count)


def load_preprocessors(encoder_path):
    """Đọc tokenizer và bộ xử lý ảnh, tắt OCR nội bộ của Transformers."""
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(encoder_path)
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(
        encoder_path,
        apply_ocr=False,
    )
    return tokenizer, image_processor


def encode_page(page, tokenizer, image_processor, variant, device):
    """Chuẩn hóa bbox về 0–1000 và giữ đầy đủ token của trang."""
    width, height = page["image_width"], page["image_height"]
    boxes = [
        [
            max(0, min(1000, int(1000 * coordinate / dimension)))
            for coordinate, dimension in zip(
                region["bbox"],
                [width, height, width, height],
                strict=True,
            )
        ]
        for region in page["regions"]
    ]
    tokens = tokenizer(
        [region["text"].strip() or tokenizer.unk_token for region in page["regions"]],
        boxes=boxes,
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        padding=False,
    )
    chunks = []
    for chunk_index in range(len(tokens["input_ids"])):
        chunk = {
            name: torch.tensor([tokens[name][chunk_index]], device=device)
            for name in ["input_ids", "attention_mask", "bbox"]
        }
        chunk["region_indices"] = torch.tensor(
            [
                -1 if index is None else index
                for index in tokens.word_ids(batch_index=chunk_index)
            ],
            device=device,
        )
        chunks.append(chunk)
    encoded = {"chunks": chunks}
    if variant == "A":
        with Image.open(page["image_path"]) as image:
            encoded["pixel_values"] = image_processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            )["pixel_values"].to(device)
    return encoded


def training_loss(role_logits, link_logits, page):
    """Tính loss đa nhãn và cân bằng cạnh đúng với các cạnh không liên kết."""
    roles = role_logits.new_tensor(page["role_labels"])
    links = torch.zeros_like(link_logits)
    for key_index, value_index in page["link_labels"]:
        links[key_index, value_index] = 1.0
    positive_count = links.sum()
    positive_weight = (links.numel() - positive_count) / positive_count.clamp(min=1)
    role_loss = nn.functional.binary_cross_entropy_with_logits(role_logits, roles)
    link_loss = nn.functional.binary_cross_entropy_with_logits(
        link_logits,
        links,
        pos_weight=positive_weight,
    )
    return role_loss + link_loss


def same_row_score(first, second):
    """Tính độ phù hợp cùng hàng Y từ phần giao nhau và tâm bbox."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    overlap = max(
        0,
        min(first_box[3], second_box[3])
        - max(first_box[1], second_box[1]),
    )
    first_height = max(first_box[3] - first_box[1], 1)
    second_height = max(second_box[3] - second_box[1], 1)
    overlap_ratio = overlap / min(first_height, second_height)
    first_center = (first_box[1] + first_box[3]) / 2
    second_center = (second_box[1] + second_box[3]) / 2
    center_gap = abs(first_center - second_center)
    center_limit = max(first_height, second_height) * 0.5
    if overlap_ratio >= 0.5:
        return min(overlap_ratio, 1.0)
    if center_gap <= center_limit:
        return max(0.0, 1.0 - center_gap / center_limit)
    return 0.0


def row_links(regions):
    """Tạo cạnh giữa mọi KEY và VALUE cùng hàng Y."""
    keys = [region for region in regions if "KEY" in region["labels"]]
    values = [region for region in regions if "VALUE" in region["labels"]]
    links = []
    for key in keys:
        for value in values:
            score = same_row_score(key, value)
            if score:
                links.append({
                    "key_region_id": key["id"],
                    "value_region_id": value["id"],
                    "score": score,
                })
    return links


def predict_page(model, encoded, page, roles, threshold, link_mode="learned"):
    """Trả vai trò và cạnh region theo learned-link, row-link hoặc cả hai."""
    if link_mode not in {"learned", "row", "both"}:
        raise ValueError("link_mode phải là learned, row hoặc both.")
    role_logits, link_logits = model(encoded, len(page["regions"]))
    role_scores = role_logits.sigmoid().cpu().tolist()
    link_scores = link_logits.sigmoid().cpu().tolist()
    regions = []
    for region, scores in zip(page["regions"], role_scores, strict=True):
        regions.append({
            "id": region["id"],
            "text": region["text"],
            "bbox": region["bbox"],
            "labels": [
                role for role, score in zip(roles, scores, strict=True)
                if score >= threshold
            ] or ["OTHER"],
            "scores": dict(zip(roles, scores, strict=True)),
        })
    learned_links = []
    for key_index, key in enumerate(regions):
        if "KEY" not in key["labels"]:
            continue
        for value_index, value in enumerate(regions):
            score = link_scores[key_index][value_index]
            if "VALUE" in value["labels"] and score >= threshold:
                learned_links.append({
                    "key_region_id": key["id"],
                    "value_region_id": value["id"],
                    "score": score,
                })
    links_by_mode = {
        "learned": learned_links,
        "row": row_links(regions),
    }
    result = {
        "regions": regions,
        "link_mode": link_mode,
        "links": links_by_mode["learned" if link_mode == "both" else link_mode],
    }
    if link_mode == "both":
        result["links_by_mode"] = links_by_mode
    return result
