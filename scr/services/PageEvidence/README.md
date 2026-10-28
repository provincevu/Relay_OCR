# PageEvidenceService

`PageEvidenceService` tạo text evidence từ ảnh page để `InputLoader` xác định
ranh giới phiếu và phân loại page.

Service này chỉ thực hiện preprocessing, detection và recognition. Nó không
đọc PDF, không tạo metadata và không phụ thuộc `OCRService`.
