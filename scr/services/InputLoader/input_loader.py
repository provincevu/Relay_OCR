# Mục đích: đọc một PDF chứa nhiều phiếu và chuẩn bị ảnh cho pipeline OCR.
# Chức năng: đếm page, xác định ranh giới phiếu, phân loại page, render ảnh và ghi metadata.
# Luồng xử lý: PDF gốc → OCR evidence → segment phiếu → ảnh PNG → metadata.json.
# Cấu hình/parameter: output_directory, input_loader_render_dpi và input_loader_renderer.
# Quan hệ: dùng PageEvidenceService để đọc evidence; không tạo PDF con và không sửa PDF gốc.
# Input/output: nhận một đường dẫn PDF; ghi thư mục phiếu và input_metadata.json.
# Contract: mỗi ảnh có image_path, source_pdf_page_number và document_page_number; chỉ số ảnh bắt đầu từ 001.

import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import cv2
from pypdf import PdfReader

from scr.services.PageEvidence import PageEvidenceService


class InputLoader:
    """Đọc PDF nhiều phiếu và tạo ảnh, metadata cho từng phiếu."""

    def __init__(self, config, page_evidence=None):
        self.config = config
        self.output_directory = Path(
            config.get(
                "input_loader_output",
                "output/inference/splitted_pdf",
            ),
        )
        self.render_dpi = config.get("input_loader_render_dpi", 200)
        self.renderer = config.get("input_loader_renderer")
        self.page_evidence = page_evidence or PageEvidenceService(config)

    def load(self, pdf_path):
        """Đọc một PDF và ghi ảnh, metadata cho các phiếu tìm được."""
        source_pdf = Path(pdf_path).resolve()
        page_count = len(PdfReader(str(source_pdf)).pages)
        if page_count == 0:
            raise ValueError(f"PDF không có page: {source_pdf}")

        self.output_directory.mkdir(parents=True, exist_ok=True)
        renderer = self._find_renderer()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            pages = []
            for source_page_number in range(1, page_count + 1):
                image_path = self._render_page(
                    renderer,
                    source_pdf,
                    source_page_number,
                    temporary_directory,
                )
                ocr_regions = self._recognize_page(image_path)
                pages.append(
                    self._build_page_evidence(
                        source_page_number,
                        image_path,
                        ocr_regions,
                    ),
                )

            documents = self._split_pages(pages)
            document_metadata = []
            for index, document_pages in enumerate(documents, start=1):
                document_metadata.append(
                    self._write_document(
                        index,
                        source_pdf,
                        page_count,
                        document_pages,
                        self.output_directory,
                    ),
                )

        metadata = {
            "schema_version": 1,
            "source_pdf": {
                "path": str(source_pdf),
                "name": source_pdf.name,
                "page_count": page_count,
            },
            "documents": document_metadata,
        }
        (self.output_directory / "input_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata

    def _find_renderer(self):
        """Tìm Ghostscript dùng để render page PDF."""
        if self.renderer:
            return self.renderer
        for renderer in ("gswin64c", "gswin32c", "gs"):
            found = shutil.which(renderer)
            if found:
                return found
        raise FileNotFoundError("Không tìm thấy Ghostscript trong PATH.")

    def _render_page(
        self,
        renderer,
        source_pdf,
        source_page_number,
        temporary_directory,
    ):
        """Render một page PDF thành ảnh tạm."""
        image_path = temporary_directory / (
            f"source_page_{source_page_number:04d}.png"
        )
        command = [
            renderer,
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{self.render_dpi}",
            f"-dFirstPage={source_page_number}",
            f"-dLastPage={source_page_number}",
            f"-sOutputFile={image_path}",
            str(source_pdf),
        ]
        subprocess.run(command, check=True)
        return image_path

    def _recognize_page(self, image_path):
        """Nhận dạng và giữ lại các vùng text của một page."""
        results = self.page_evidence.recognize(str(image_path))
        return self._recover_pagination(image_path, results)

    def _build_page_evidence(
        self,
        source_page_number,
        image_path,
        ocr_regions,
    ):
        """Tạo evidence phân đoạn và giữ metadata OCR của page."""
        ocr_regions = self._normalize_ocr_regions(ocr_regions)
        text = " ".join(region["text"] for region in ocr_regions)
        current_page, total_pages = self._read_page_number(text)
        normalized_text = self._normalize(text)
        has_cover_signature = any(
            phrase in normalized_text
            for phrase in (
                "phieu chinh dinh ro le bao ve",
                "mo ta chung",
                "thiet bi duoc bao ve",
                "nguyen tac hoat dong",
            )
        )
        return {
            "source_page_number": source_page_number,
            "image_path": str(image_path),
            "current_page": current_page,
            "total_pages": total_pages,
            "is_form_start": current_page == 1 or has_cover_signature,
            "ocr_regions": ocr_regions,
        }

    def _recover_pagination(self, image_path, ocr_regions):
        """OCR lại vùng pagination khi kết quả hiện tại chưa đủ số page."""
        text = " ".join(result["text"] for result in ocr_regions)
        current_page, total_pages = self._read_page_number(text)
        if current_page is not None and total_pages is not None:
            return ocr_regions

        pagination_box = self._find_pagination_box(image_path, ocr_regions)
        if pagination_box is None:
            return ocr_regions

        recovered = self.page_evidence.recognize_region(
            str(image_path),
            pagination_box,
        )
        recovered_text = " ".join(
            [text, recovered["text"]],
        )
        current_page, total_pages = self._read_page_number(recovered_text)
        if current_page is None or total_pages is None:
            return ocr_regions

        return [
            result
            for result in ocr_regions
            if not self._boxes_overlap(result["box"], pagination_box)
        ] + [recovered]

    def _find_pagination_box(self, image_path, ocr_regions):
        """Tìm dòng nghi ngờ chứa pagination để OCR lại."""
        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]
        markers = [
            result
            for result in ocr_regions
            if self._box_center_y(result["box"]) <= image_height * 0.45
            and re.search(
                r"trang|page|/|\bof\b|\btren\b",
                self._normalize(result["text"]),
            )
        ]
        if not markers:
            return None

        marker_box = min(
            (result["box"] for result in markers),
            key=self._box_center_y,
        )
        marker_y = sum(point[1] for point in marker_box) / len(marker_box)
        marker_height = max(
            point[1] for point in marker_box
        ) - min(point[1] for point in marker_box)
        line_results = [
            result
            for result in ocr_regions
            if abs(self._box_center_y(result["box"]) - marker_y)
            <= max(marker_height * 2, image_height * 0.01)
        ]
        if not line_results:
            return None

        x_min = min(
            point[0]
            for result in line_results
            for point in result["box"]
        )
        y_min = min(
            point[1]
            for result in line_results
            for point in result["box"]
        )
        x_max = max(
            point[0]
            for result in line_results
            for point in result["box"]
        )
        y_max = max(
            point[1]
            for result in line_results
            for point in result["box"]
        )
        return [
            [
                max(0, x_min - image_width * 0.01),
                max(0, y_min - image_height * 0.004),
            ],
            [
                min(image_width - 1, x_max + image_width * 0.12),
                max(0, y_min - image_height * 0.004),
            ],
            [
                min(image_width - 1, x_max + image_width * 0.12),
                min(image_height - 1, y_max + image_height * 0.004),
            ],
            [
                max(0, x_min - image_width * 0.01),
                min(image_height - 1, y_max + image_height * 0.004),
            ],
        ]

    def _normalize_ocr_regions(self, ocr_regions):
        """Chuẩn hóa region OCR trước khi ghi metadata."""
        normalized_regions = []
        for index, region in enumerate(ocr_regions, start=1):
            box = region["box"]
            normalized_regions.append(
                {
                    "id": f"region_{index:04d}",
                    "text": region["text"],
                    "box": box,
                    "bbox": self._box_to_bbox(box),
                    "score": region.get("score"),
                },
            )
        return normalized_regions

    def _box_to_bbox(self, box):
        """Đổi polygon OCR thành bounding box."""
        return [
            min(point[0] for point in box),
            min(point[1] for point in box),
            max(point[0] for point in box),
            max(point[1] for point in box),
        ]

    def _box_center_y(self, box):
        """Tính tọa độ dọc trung tâm của polygon."""
        return sum(point[1] for point in box) / len(box)

    def _boxes_overlap(self, first_box, second_box):
        """Kiểm tra hai box có giao nhau hay không."""
        first = self._box_to_bbox(first_box)
        second = self._box_to_bbox(second_box)
        return not (
            first[2] < second[0]
            or second[2] < first[0]
            or first[3] < second[1]
            or second[3] < first[1]
        )

    def _read_page_number(self, text):
        """Đọc số page hiện tại và tổng số page từ text OCR."""
        normalized_text = self._normalize(text)
        match = re.search(
            r"(?:trang|page)\s*[:.-]?\s*(\d+)\s*(?:/|of|tren)\s*(\d+)",
            normalized_text,
        )
        if match:
            return int(match.group(1)), int(match.group(2))

        match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", normalized_text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def _split_pages(self, pages):
        """Tách danh sách page thành các phiếu theo evidence đã thu được."""
        documents = []
        document_pages = []

        for page in pages:
            if document_pages and page["is_form_start"]:
                documents.append(document_pages)
                document_pages = []

            document_pages.append(page)
            if (
                page["current_page"] is not None
                and page["total_pages"] is not None
                and page["current_page"] == page["total_pages"]
            ):
                documents.append(document_pages)
                document_pages = []

        if document_pages:
            documents.append(document_pages)
        return documents

    def _write_document(
        self,
        document_index,
        source_pdf,
        source_page_count,
        pages,
        output_directory,
    ):
        """Ghi ảnh và metadata cho một phiếu."""
        document_id = f"{document_index:04d}"
        document_directory = output_directory / document_id
        document_directory.mkdir(parents=True, exist_ok=True)
        page_metadata = []

        for image_index, page in enumerate(pages, start=1):
            image_name = f"{image_index:03d}.png"
            output_image = document_directory / image_name
            shutil.copy2(page["image_path"], output_image)
            document_page_number = image_index
            is_comment_page = (
                self._is_comment_page(page["image_path"])
                if document_page_number == 2
                else False
            )
            page_metadata.append(
                {
                    "image_path": f"{document_id}/{image_name}",
                    "image_name": image_name,
                    "image_index": image_index,
                    "source_pdf_page_number": page["source_page_number"],
                    "document_page_number": document_page_number,
                    "current_page": page["current_page"],
                    "total_pages": page["total_pages"],
                    "ocr_regions": page["ocr_regions"],
                    "page_type": self._classify_page(
                        document_page_number,
                        is_comment_page,
                    ),
                },
            )

        metadata = {
            "schema_version": 1,
            "document_id": document_id,
            "source_pdf": {
                "path": str(source_pdf),
                "name": source_pdf.name,
                "page_count": source_page_count,
            },
            "source_page_start": pages[0]["source_page_number"],
            "source_page_end": pages[-1]["source_page_number"],
            "document_page_count": len(pages),
            "pages": page_metadata,
        }
        (document_directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "document_id": document_id,
            "metadata_path": f"{document_id}/metadata.json",
            "source_page_start": metadata["source_page_start"],
            "source_page_end": metadata["source_page_end"],
            "page_count": metadata["document_page_count"],
        }

    def _classify_page(self, document_page_number, is_comment_page):
        """Phân loại page theo vị trí trong phiếu."""
        if document_page_number == 1:
            return "PAGE_1"
        if document_page_number == 2:
            return "PAGE_2_COMMENT" if is_comment_page else "PAGE_2_DATA"
        return "PAGE_3_PLUS"

    def _is_comment_page(self, image_path):
        """Phân loại page comment bằng logic hình ảnh hiện có."""
        from scripts.data.export_pages_3plus import _is_comment_page

        return _is_comment_page(image_path)

    def _normalize(self, text):
        """Chuẩn hóa text OCR để tìm pattern không phụ thuộc dấu."""
        decomposed = unicodedata.normalize("NFD", text.lower())
        plain_text = "".join(
            character
            for character in decomposed
            if unicodedata.category(character) != "Mn"
        )
        return plain_text.replace("đ", "d")
