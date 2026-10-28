# Mục đích: LayoutLMv3 chỉ dùng text + bbox cho page 3+.
# Chức năng: mã hóa OCR, dự đoán một role cho mỗi region và row-link giữa các region.
# Luồng xử lý: chia token thành chunk → encoder → gộp region → role head.
# Cấu hình/parameter: variant A/B, checkpoint encoder, danh sách role,
# gradient checkpointing.
# Quan hệ: dùng chung bởi lệnh train và compare trong compare_page3_layout.
# Input/output: một page OCR → logits [region, role] và liên kết KEY → VALUE;
# value được nối theo quy tắc cùng hàng.
# Contract: A dùng ảnh; B visual_embed=False; model training hiện tại dùng B.
# Contract: không cắt bỏ token cuối trang; cạnh có thể nối region khác chunk.

import torch
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3TokenizerFast,
)

from scripts.inference.page_n_prediction_filter import (
    filter_distant_same_row_candidates,
    filter_prediction_links,
)
from scripts.inference.role_change_metadata import record_role_change


class Page3LayoutModel(nn.Module):
    """Học vai trò region; liên kết KEY -> VALUE dùng row-link."""

    def __init__(
        self,
        encoder,
        role_count,
        head_dropout=0.3,
        gradient_checkpointing=False,
    ):
        """Tạo head role với dropout; không học learned-link."""
        super().__init__()
        self.encoder = encoder
        self.gradient_checkpointing = gradient_checkpointing
        hidden_size = encoder.config.hidden_size
        self.head_dropout = nn.Dropout(head_dropout)
        self.role_head = nn.Linear(hidden_size, role_count)

    def forward(self, encoded, region_count):
        """Gộp mọi subtoken theo ID region rồi dự đoán role."""
        region_hidden = encode_region_embeddings(
            self.encoder,
            encoded,
            region_count,
            self._encode_chunk,
        )
        role_logits = self.role_head(self.head_dropout(region_hidden))
        return role_logits

    def _encode_chunk(self, inputs):
        """Chạy một chunk bằng checkpoint thủ công khi đang huấn luyện."""
        if not (self.gradient_checkpointing and self.training):
            return self.encoder(**inputs).last_hidden_state
        return checkpoint(
            lambda *values: self.encoder(
                **dict(zip(inputs.keys(), values, strict=True))
            ).last_hidden_state,
            *inputs.values(),
            use_reentrant=False,
        )


def encode_region_embeddings(
    encoder,
    encoded,
    region_count,
    encode_chunk=None,
):
    """Chạy encoder và gộp hidden state theo từng region."""
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
        hidden = (
            encode_chunk(inputs)
            if encode_chunk is not None
            else encoder(**inputs).last_hidden_state
        )
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
    return pooled / counts


def load_pretrained(
    encoder_path,
    variant,
    role_count,
    seed,
    head_dropout=0.3,
    gradient_checkpointing=False,
):
    """Khởi tạo encoder pretrained và head mới cần được huấn luyện."""
    config = LayoutLMv3Config.from_pretrained(encoder_path)
    config.visual_embed = variant == "A"
    encoder = LayoutLMv3Model.from_pretrained(
        encoder_path,
        config=config,
    )
    torch.manual_seed(seed)
    return Page3LayoutModel(
        encoder,
        role_count,
        head_dropout,
        gradient_checkpointing,
    )


def load_preprocessors(encoder_path):
    """Đọc tokenizer và bộ xử lý ảnh, tắt OCR nội bộ của Transformers."""
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(encoder_path)
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(
        encoder_path,
        apply_ocr=False,
    )
    return tokenizer, image_processor


def iter_page_chunks(page, tokenizer, device):
    """Tokenize page và tạo từng chunk trên thiết bị được chọn."""
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
        yield chunk


def encode_page(page, tokenizer, image_processor, variant, device):
    """Chuẩn hóa bbox về 0–1000 và giữ đầy đủ token của trang."""
    chunks = list(iter_page_chunks(page, tokenizer, device))
    encoded = {"chunks": chunks}
    if variant == "A":
        with Image.open(page["image_path"]) as image:
            encoded["pixel_values"] = image_processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            )["pixel_values"].to(device)
    return encoded


def training_loss(role_logits, page):
    """Tính single-class loss từ chỉ số role của mỗi region."""
    role_targets = torch.as_tensor(
        page["role_labels"],
        dtype=torch.long,
        device=role_logits.device,
    )
    known = role_targets != -100
    if not known.any():
        return role_logits.sum() * 0
    return nn.functional.cross_entropy(
        role_logits[known],
        role_targets[known],
    )


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


def row_links(regions, source_role, target_role, source_field, target_field):
    """Tạo cạnh giữa hai vai trò cùng hàng Y."""
    sources = [
        region for region in regions if source_role in region["labels"]
    ]
    targets = [
        region for region in regions if target_role in region["labels"]
    ]
    links = []
    for source in sources:
        for target in targets:
            score = same_row_score(source, target)
            if score:
                links.append({
                    source_field: source["id"],
                    target_field: target["id"],
                    "score": score,
                })
    return links


def predict_page(model, encoded, page, roles, threshold, link_mode="row"):
    """Chọn role có score cao nhất và tạo cạnh row-link."""
    if link_mode != "row":
        raise ValueError("Pipeline hiện tại chỉ hỗ trợ row-link.")
    role_logits = model(encoded, len(page["regions"]))
    role_scores = role_logits.softmax(dim=-1).cpu().tolist()
    regions = []
    role_change_codes = {}
    for region, scores in zip(page["regions"], role_scores, strict=True):
        best_role_index = max(range(len(scores)), key=scores.__getitem__)
        role = roles[best_role_index]
        if region["text"].strip().lower() in {"item", "value"}:
            role = "FIELD_TABLE"
        regions.append({
            "id": region["id"],
            "text": region["text"],
            "bbox": region["bbox"],
            "labels": [role],
            "scores": dict(zip(roles, scores, strict=True)),
        })
        if role == "OTHER":
            record_role_change(role_change_codes, region["id"], 1)
    filter_distant_same_row_candidates(
        regions,
        role_change_codes=role_change_codes,
    )
    row_key_value_links = row_links(
        regions,
        "KEY",
        "VALUE",
        "key_region_id",
        "value_region_id",
    )
    result = {
        "regions": regions,
        "role_change_codes": role_change_codes,
        "link_mode": link_mode,
        "links": row_key_value_links,
    }
    return filter_prediction_links(result)
