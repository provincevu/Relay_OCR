# Mục đích: công khai service tạo OCR evidence cho page PDF.
# Chức năng: export PageEvidenceService.
# Luồng xử lý: import package → dùng PageEvidenceService.recognize().
# Cấu hình/parameter: nhận config model OCR.
# Quan hệ: được InputLoader sử dụng; không phụ thuộc OCRService.
# Input/output: nhận ảnh page; trả danh sách vùng text đã nhận dạng.
# Contract: giữ tên import PageEvidenceService.

from .page_evidence_service import PageEvidenceService

__all__ = ["PageEvidenceService"]
