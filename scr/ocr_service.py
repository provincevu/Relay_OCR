import json

import cv2

from .services.PageEvidence import PageEvidenceService
from .page1.table_1_analysis import Table1Analysis

# Mục đích: cung cấp dịch vụ OCR kết hợp tiền xử lý, detection và recognition.
# Chức năng: OCR ảnh, trích xuất TABLE 01 page 1 và điều phối PDF nhiều phiếu.
# Luồng xử lý: ảnh → preprocessing → OCR hoặc PDF → InputLoader → page services.
# Cấu hình: nhận một dict config và truyền cho các service bên dưới.
# Quan hệ: phối hợp PageEvidenceService và Table1Analysis.
# Input/output: nhận ảnh hoặc PDF; trả kết quả trong bộ nhớ.
# Contract: giữ detect_text(), recognize_text(), analyze_page1() và thêm process_pdf().

INPUT_LOADER_RENDER_DPI = 400


class OCRService:
    """Dịch vụ OCR kết hợp phát hiện và nhận dạng văn bản."""

    # Khởi tạo các service và dùng chung model text cho TABLE 01.
    def __init__(self, config):
        self.config = config
        self.page_evidence = PageEvidenceService(config)
        self.image_preprocessing = self.page_evidence.image_preprocessing
        self.detector = self.page_evidence.detector
        self.recognitor = self.page_evidence.recognitor
        self.table_1_analysis = Table1Analysis(
            config,
            self.detector,
            self.recognitor,
        )
        self._page3_inference = None

    # Phát hiện các vùng text trên ảnh đã tiền xử lý.
    def detect_text(self, image):
        # Tiền xử lý ảnh trước khi phát hiện vùng văn bản
        processed_image = self.image_preprocessing.preprocess(image)
        results = self.detector.detect(processed_image)
        return results

    # Nhận dạng text trong toàn bộ polygon được phát hiện.
    def recognize_text(self, image):
        # Tiền xử lý ảnh trước khi detection và recognition
        return self.page_evidence.recognize(image)

    # Trích xuất field TABLE 01 từ ảnh page 1.
    def analyze_page1(self, image_path):
        """Trả fields và table của page 1 mà không tạo output file."""
        image = self.image_preprocessing.preprocess(image_path)
        return self.table_1_analysis.analyze(image)

    def process_pdf(self, pdf_path, output_directory=None):
        """Xử lý một PDF nhiều phiếu và trả kết quả theo từng phiếu."""
        from .services.InputLoader import InputLoader

        input_loader_config = dict(self.config)
        if output_directory is not None:
            input_loader_config["input_loader_output"] = str(
                output_directory,
            )
        input_loader_config.setdefault(
            "input_loader_render_dpi",
            INPUT_LOADER_RENDER_DPI,
        )
        input_loader = InputLoader(
            input_loader_config,
            self.page_evidence,
        )
        input_metadata = input_loader.load(pdf_path)
        documents = []

        for document in input_metadata["documents"]:
            document_metadata = self._read_document_metadata(
                input_loader.output_directory,
                document["document_id"],
            )
            page1_result = None
            page3_records = []

            for page in document_metadata["pages"]:
                image_path = (
                    input_loader.output_directory / page["image_path"]
                )
                if page["page_type"] == "PAGE_1":
                    page1_result = self.analyze_page1(image_path)
                elif page["page_type"] == "PAGE_3_PLUS":
                    page3_records.append(
                        self._build_page3_record(
                            image_path,
                            page,
                            input_metadata["source_pdf"]["name"],
                        ),
                    )

            page3_result = []
            if page3_records:
                if self._page3_inference is None:
                    from .page3 import Page3Inference

                    self._page3_inference = Page3Inference(self.config)
                page3_result = self._page3_inference.infer_pages(page3_records)

            documents.append(
                {
                    "document_id": document_metadata["document_id"],
                    "source_page_start": document_metadata[
                        "source_page_start"
                    ],
                    "source_page_end": document_metadata[
                        "source_page_end"
                    ],
                    "page1": page1_result,
                    "page3_plus": page3_result,
                },
            )

        return {
            "source_pdf": input_metadata["source_pdf"],
            "documents": documents,
            "_debug_cache": {
                "kind": "ocr_evidence",
                "hit": input_loader.last_ocr_cache_hit,
                "key": input_loader.last_ocr_cache_key,
                "reused": input_loader.reuse_ocr_cache,
            },
        }

    def _read_document_metadata(self, output_directory, document_id):
        """Đọc metadata của một phiếu đã được InputLoader tạo."""
        metadata_path = output_directory / document_id / "metadata.json"
        return json.loads(
            metadata_path.read_text(encoding="utf-8"),
        )

    def _build_page3_record(self, image_path, page, pdf_name):
        """Tạo page record chuẩn từ metadata OCR đã có."""
        image = cv2.imread(str(image_path))
        regions = []

        for region in page["ocr_regions"]:
            regions.append(
                {
                    "id": region["id"],
                    "text": region["text"],
                    "bbox": region["bbox"],
                    "score": region.get("score"),
                },
            )

        return {
            "pdf_name": pdf_name,
            "physical_page": page["source_pdf_page_number"],
            "document_page_number": page["document_page_number"],
            "image_name": page["image_name"],
            "image_path": str(image_path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "regions": regions,
        }
