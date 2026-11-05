# Mục đích: cung cấp service suy luận Layout cho một hoặc nhiều page 3+.
# Chức năng: load checkpoint, dự đoán role và tạo link KEY → VALUE từ page OCR.
# Luồng xử lý: page OCR → encode → Layout model → prediction → hậu xử lý page.
# Cấu hình/parameter: page3_checkpoint, page3_device và page3_threshold.
# Quan hệ: dùng Page3LayoutModel hiện có và TextPostprocessing.
# Input/output: nhận page record đã OCR; trả prediction giữ metadata page.
# Contract: giữ region ID, text, bbox và metadata pdf_name/physical_page/document_page_number.

from pathlib import Path

import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

from scripts.training.page3_layout_model import (
    Page3LayoutModel,
    encode_page,
    predict_page,
)

from .text_postprocessing import TextPostprocessing


class Page3Inference:
    """Suy luận role và link cho page OCR thuộc page 3+."""

    def __init__(self, config):
        self.config = config
        self.device = config.get("page3_device", "cpu")
        self.threshold = config.get("page3_threshold", 0.5)
        self.model, self.tokenizer, self.roles = self._load_model(
            Path(config.get("page3_checkpoint")),
        )
        self.text_postprocessing = TextPostprocessing(config)

    def infer(self, page_record):
        """Dự đoán một page đã có region OCR."""
        page = self._prepare_page(page_record)
        encoded = encode_page(
            page,
            self.tokenizer,
            None,
            "B",
            self.device,
        )
        with torch.inference_mode():
            prediction = predict_page(
                self.model,
                encoded,
                page,
                self.roles,
                self.threshold,
                "row",
            )

        ocr_scores = {
            region["id"]: region.get("score")
            for region in page_record["regions"]
        }
        for region in prediction["regions"]:
            region["ocr_score"] = ocr_scores[region["id"]]
        self.text_postprocessing.process_page(prediction)
        self._add_link_texts(prediction)
        prediction.update(
            {
                "pdf_name": page_record.get("pdf_name", page_record.get("pdfName")),
                "physical_page": page_record.get(
                    "physical_page",
                    page_record.get("physicalPage"),
                ),
                "document_page_number": page_record.get(
                    "document_page_number",
                ),
                "image_name": page_record.get(
                    "image_name",
                    page_record.get("imageName"),
                ),
                "image_path": page_record.get(
                    "image_path",
                    page_record.get("imageName"),
                ),
                "image_width": page["image_width"],
                "image_height": page["image_height"],
            },
        )
        return prediction

    def infer_pages(self, page_records):
        """Suy luận tuần tự và hậu xử lý liên tục nhiều page."""
        predictions = [self.infer(page_record) for page_record in page_records]
        self.text_postprocessing.process_pages(predictions)
        for prediction in predictions:
            self._add_link_texts(prediction)
        return predictions

    def _load_model(self, checkpoint_directory):
        """Tải checkpoint Layout model và tokenizer."""
        checkpoint = torch.load(
            checkpoint_directory / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        encoder_config = LayoutLMv3Config.from_dict(
            checkpoint["encoder_config"],
        )
        model = Page3LayoutModel(
            LayoutLMv3Model(encoder_config),
            len(checkpoint["roles"]),
            checkpoint.get("head_dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device).eval()
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            checkpoint_directory,
        )
        return model, tokenizer, checkpoint["roles"]

    def _prepare_page(self, page_record):
        """Chuẩn hóa tên bbox nhưng giữ nguyên dữ liệu OCR đầu vào."""
        return {
            **page_record,
            "regions": [
                {
                    **region,
                    "bbox": region.get("bbox", region.get("box")),
                }
                for region in page_record["regions"]
            ],
            "image_width": page_record.get(
                "image_width",
                page_record.get("width"),
            ),
            "image_height": page_record.get(
                "image_height",
                page_record.get("height"),
            ),
        }

    def _add_link_texts(self, prediction):
        """Thêm text của KEY và VALUE vào link theo region ID."""
        region_by_id = {
            region["id"]: region
            for region in prediction["regions"]
        }
        for link in prediction["links"]:
            link["key_text"] = self._join_region_text(
                region_by_id,
                [link["key_region_id"]],
            )
            link["value_text"] = self._join_region_text(
                region_by_id,
                [link["value_region_id"]],
            )

    def _join_region_text(self, region_by_id, region_ids):
        """Ghép text của các region theo thứ tự tọa độ."""
        regions = sorted(
            (region_by_id[region_id] for region_id in region_ids),
            key=lambda region: (region["bbox"][1], region["bbox"][0]),
        )
        return " ".join(
            region["text"].strip()
            for region in regions
            if region["text"].strip()
        )
