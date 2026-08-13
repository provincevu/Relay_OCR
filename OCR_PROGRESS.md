# Ghi chú tiến độ dự án Relay OCR

Ngày cập nhật: 13/08/2026

Tài liệu này ghi lại những phần đã xây dựng, cách các thành phần đang kết nối với nhau và những phần chưa triển khai.

## Mục tiêu hiện tại

Dự án đang xây dựng pipeline OCR cho ảnh phiếu/chỉ thị rơ-le:

```text
Ảnh đầu vào
→ Tiền xử lý ảnh bằng PaddleOCR
→ Phát hiện vùng văn bản bằng PaddleOCR
→ Cắt từng vùng văn bản
→ Nhận dạng từng vùng bằng VietOCR VGG Transformer
```

## Các file đã có

### `scr/services/image_preprocessing.py`

Class `ImagePreprocessing` đóng gói các bước tiền xử lý ảnh độc lập của PaddleOCR:

- `DocImgOrientationClassification`: xác định hướng toàn bộ tài liệu và xoay ảnh về đúng hướng.
- `TextImageUnwarping`: hiệu chỉnh ảnh tài liệu bị méo/phối cảnh bằng model `UVDoc`.
- `TextLineOrientationClassification`: xác định dòng chữ bị xoay 180 độ trước khi đưa vào VietOCR.

Cấu hình mặc định:

```python
{
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "debug": False,
}
```

Khi `debug=False`, ảnh sau tiền xử lý chỉ được giữ trong bộ nhớ và không tạo ảnh trung gian.

Khi `debug=True`, ảnh sau tiền xử lý được lưu vào:

```text
output/preprocessing/
```

Nhãn textline của PaddleOCR có dạng `0_degree` hoặc `180_degree`. Code hiện tại chuyển `180_degree` thành góc xoay 180 độ.

### `scr/services/detector.py`

Class `Detector` dùng PaddleOCR `TextDetection` để phát hiện vùng chứa văn bản.

Cấu hình mặc định:

```python
{
    "model_name": "PP-OCRv6_medium_det",
    "device": "cpu",
    "engine": "paddle_static",
    "cpu_threads": 2,
    "enable_mkldnn": False,
    "limit_side_len": 1600,
    "limit_type": "max",
    "thresh": 0.3,
    "box_thresh": 0.6,
    "unclip_ratio": 2.0,
}
```

Phương thức chính:

```python
Detector(config).detect(image)
```

`image` có thể là đường dẫn ảnh hoặc ảnh `numpy.ndarray`. Kết quả trả về là kết quả detection gốc của PaddleOCR, trong đó có `dt_polys`.

### `scr/services/recognitor.py`

Class `Recognitor` dùng VietOCR với model mặc định:

```text
vgg_transformer
```

Cấu hình:

```python
{
    "recognition_model": "vgg_transformer",
    "recognition_device": "cpu",
}
```

Phương thức chính:

```python
Recognitor(config).recognize(crop)
```

Input là ảnh crop dạng OpenCV BGR. Kết quả là chuỗi văn bản nhận dạng được.

### `scr/ocr_service.py`

Class `OCRService` kết hợp ba thành phần:

```text
ImagePreprocessing → Detector → Recognitor
```

Các phương thức hiện có:

```python
service.detect_text(image)
```

Thực hiện tiền xử lý và detection, trả về kết quả detection gốc của PaddleOCR.

```python
service.recognize_text(image)
```

Thực hiện:

1. Tiền xử lý ảnh.
2. Detection các vùng văn bản.
3. Lấy `dt_polys`.
4. Cắt hình chữ nhật bao quanh từng polygon.
5. Xử lý hướng textline nếu được bật.
6. Recognition từng crop bằng VietOCR.

Kết quả hiện tại có dạng:

```python
[
    {
        "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "text": "Nội dung nhận dạng",
    }
]
```

## Các script kiểm tra hiện có

### `test_detection.py`

Script detection cơ bản theo PaddleOCR:

- Đọc ảnh trong `data/image/`.
- Dùng `TextDetection` với model `PP-OCRv6_medium_det`.
- Lưu ảnh detection vào `output/image/`.
- Lưu JSON tọa độ vào `output/json/`.

### `test_recognition.py`

Script kết hợp PaddleOCR detection và VietOCR recognition:

- Detection các vùng chữ.
- Cắt từng vùng chữ.
- Recognition bằng VietOCR `vgg_transformer`.
- Vẽ polygon và text nhận dạng lên ảnh.
- Lưu ảnh vào `output/recognition/`.

### `test_table_analysis_page1.py`

File đã được tạo nhưng hiện chưa có code triển khai phân tích bảng.

## Cấu hình mẫu cho `OCRService`

Ví dụ tối thiểu:

```python
config = {
    "device": "cpu",
    "engine": "paddle_static",
}

service = OCRService(config)
results = service.recognize_text("./data/image/sample.png")
```

Bật đầy đủ tiền xử lý:

```python
config = {
    "device": "cpu",
    "engine": "paddle_static",
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": True,
    "use_textline_orientation": True,
    "debug": True,
}
```

## Dependency hiện tại

Các dependency chính trong `requirements.txt`:

```text
paddleocr==3.7.0
paddlepaddle==3.3.0
vietocr==0.3.13
torch==2.13.0
torchvision==0.28.0
```

Project sử dụng môi trường `.venv` hiện tại. Không cần tạo môi trường mới nếu môi trường này vẫn hoạt động bình thường.

## Cách chạy các script

Mở PowerShell tại thư mục project:

```powershell
cd C:\Users\tinhv\MyProject\A_WORK\Relay_OCR
.\.venv\Scripts\Activate.ps1
```

Chạy detection:

```powershell
python test_detection.py
```

Chạy recognition:

```powershell
python test_recognition.py
```

Ảnh đầu vào đặt trong:

```text
data/image/
```

## Những phần chưa triển khai

- Xác định vùng bảng bằng model table detection.
- Xác định các cell vật lý trong từng bảng.
- Gán text box vào đúng cell.
- Nhóm text box thành dòng.
- Xác định thứ tự dòng/cột của bảng.
- Xử lý `rowspan` và `colspan`.
- Xuất kết quả bảng thành JSON có cấu trúc hàng/cột.
- Tạo ảnh debug trực quan cho table/cell/line/text box.
- Tích hợp pipeline phân tích bảng vào `OCRService`.

## Hướng phát triển tiếp theo

Pipeline phân tích bảng dự kiến:

```text
Ảnh đầu vào
→ Tiền xử lý ảnh
→ Xác định vùng bảng
→ Xác định cell
→ Text detection
→ Gán text box vào cell
→ Nhóm text box thành dòng
→ VietOCR recognition
→ Xuất JSON và ảnh debug
```

Khi phát triển phần bảng, nên giữ nguyên ba class hiện tại và thêm thành phần xử lý bảng riêng để tránh làm phức tạp `Detector` và `Recognitor`.

## Ghi chú về debug

Chỉ bật debug khi cần kiểm tra ảnh trung gian:

```python
config["debug"] = True
```

Ở chế độ bình thường:

```python
config["debug"] = False
```

Pipeline không lưu ảnh tiền xử lý ra đĩa.

## Quy ước cập nhật tài liệu

Khi có yêu cầu ghi thêm tiến độ hoặc thay đổi mới, bổ sung nội dung vào cuối file trong phần `Nhật ký cập nhật`, không cần viết lại các phần cũ nếu không được yêu cầu.

Mỗi nội dung mới dùng mẫu sau:

```markdown
## [NGÀY/THÁNG/NĂM] Tên nội dung cập nhật

### Đã làm
- ...

### File liên quan
- `...`

### Kết quả
- ...

### Việc tiếp theo
- ...
```

## Nhật ký cập nhật

Các thay đổi mới sẽ được ghi nối tiếp tại đây theo thứ tự thời gian, mục mới nhất đặt ở cuối file.
