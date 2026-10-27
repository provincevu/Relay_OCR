# Pipeline xử lý Relay OCR

Thư mục này chứa các bước chuẩn bị dữ liệu, annotation, huấn luyện, inference
và các công cụ kiểm tra kết quả. Các lệnh bên dưới được chạy từ thư mục gốc
của repository:

```text
C:\Users\tinhv\MyProject\A_WORK\Relay_OCR
```

Pipeline chính dành cho page 3+ có luồng tổng quát:

```text
PDF
  ↓
Render page dữ liệu
  ↓
Annotation
  ↓
Chia train / val / test
  ↓
Training LayoutLMv3
  ↓
Inference
  ↓
Lọc prediction và xem kết quả
```

## 1. Chuẩn bị input

Đặt các file PDF cần xử lý vào:

```text
data/pdf/
```

Kiểm tra các dependency trong `requirements.txt`. Một số bước cần thêm công cụ
ngoài Python:

- Ghostscript để render PDF.
- PaddleOCR và VietOCR cho OCR.
- PyTorch và Transformers cho Layout model.
- `OPENAI_API_KEY` nếu tạo ground truth TABLE 01 bằng OpenAI API.

## 2. Render page 3+

Để tự nhận diện page dữ liệu và render toàn bộ page 3+:

```bash
python -m scripts.data.export_pages_3plus
```

Ảnh được ghi vào:

```text
data/pdf/page3plus_png/
```

Nếu đã có cache chọn page và chỉ cần render page 3:

```bash
python -m scripts.data.export_page3
```

Ảnh được ghi vào `data/page3/`.

## 3. Tạo dataset page 3+

Có thể render và chia dataset trong một bước:

```bash
python -m scripts.data.build_page3plus_dataset
```

Ví dụ thay đổi DPI và seed:

```bash
python -m scripts.data.build_page3plus_dataset --dpi 400 --seed 42
```

Kết quả mặc định nằm trong:

```text
data/page3plus_dataset/
├── train/
├── val/
├── test/
└── manifest.json
```

Nếu cần tạo lại thư mục output:

```bash
python -m scripts.data.build_page3plus_dataset --overwrite
```

Nếu đã có `annotations.json` và chỉ muốn chia các phiếu thành ba tập:

```bash
python -m scripts.data.split_page3_dataset \
  --dataset path/to/annotations.json \
  --output path/to/split_dataset
```

## 4. Tạo annotation page 3+

### 4.1. Tạo giao diện annotation

Từ thư mục output Rule Induction:

```bash
python -m scripts.annotation.annotate_page3 \
  path/to/rule_induction_output
```

Có thể chỉ định thư mục ảnh page 3+:

```bash
python -m scripts.annotation.annotate_page3 \
  path/to/rule_induction_output \
  --image-directory path/to/page3_images
```

Lệnh này tạo giao diện HTML và `annotation_manifest.json`.

### 4.2. Chạy server annotation

```bash
python -m scripts.annotation.serve_page3_annotations \
  path/to/rule_induction_output \
  --image-directory path/to/page3_images
```

Mở địa chỉ localhost được in trong terminal. Kết quả xác nhận được ghi vào
`annotations.json`.

### 4.3. Bổ sung header

```bash
python -m scripts.annotation.resolve_page3_headers \
  path/to/annotations.json
```

Script này cập nhật metadata header và tạo bản sao lưu nếu truyền thêm
`--backup`.

### 4.4. Chuyển annotation schema cũ

Khi dataset còn annotation inline theo schema cũ:

```bash
python -m scripts.annotation.migrate_inline_key_value_annotations
```

Tham số cụ thể của script có thể xem bằng:

```bash
python -m scripts.annotation.migrate_inline_key_value_annotations --help
```

## 5. Training Layout model

### 5.1. Huấn luyện trực tiếp từ annotation

```bash
python -m scripts.training.train_page3_text_layout train \
  --dataset path/to/train_dataset \
  --validation-dataset path/to/val_dataset \
  --output output/checkpoints/page3_layout
```

Có thể xem toàn bộ tham số:

```bash
python -m scripts.training.train_page3_text_layout --help
```

Model mặc định dùng `microsoft/layoutlmv3-base`. Checkpoint được ghi vào
thư mục truyền qua `--output`.

### 5.2. Tạo embedding cache

```bash
python -m scripts.training.prepare_page3_embeddings \
  --dataset path/to/train_dataset \
  --output output/page3_embeddings
```

### 5.3. Huấn luyện từ embedding cache

```bash
python -m scripts.training.train_page3_text_layout train-cached \
  --dataset output/page3_embeddings/train \
  --validation-dataset output/page3_embeddings/val \
  --output output/checkpoints/page3_layout_cached
```

Tên subcommand và tham số phải được kiểm tra lại bằng `--help` nếu dataset
được tạo theo cấu trúc khác.

### 5.4. Đánh giá checkpoint

```bash
python -m scripts.training.evaluate_page3_text_layout \
  --dataset path/to/test_dataset \
  --checkpoint output/checkpoints/page3_layout \
  --output output/page3_evaluation.json
```

## 6. Inference page 3+

Chạy OCR và Layout model trên các PDF:

```bash
python -m scripts.inference.infer_page_n_text_layout \
  --pdf-directory data/pdf \
  --checkpoint output/checkpoints/page3_layout \
  --output output/page3_predictions
```

Có thể chỉ định recognition cache, threshold và device:

```bash
python -m scripts.inference.infer_page_n_text_layout \
  --pdf-directory data/pdf \
  --checkpoint output/checkpoints/page3_layout \
  --output output/page3_predictions \
  --recognition-cache-directory output/recognition_cache \
  --threshold 0.5 \
  --device cpu
```

Kết quả chính là `predictions.json`, cùng các ảnh và cache OCR trong thư mục
output.

## 7. Lọc prediction

Sau inference, áp dụng các quy tắc lọc link và role:

```bash
python -m scripts.inference.filter_page_n_predictions \
  output/page3_predictions/predictions.json
```

Script giữ nguyên schema prediction và cập nhật các link hoặc role theo quy
tắc hiện tại. File metadata lý do thay đổi role được ghi cùng thư mục output.

## 8. Xem kết quả inference

```bash
python -m scripts.tools.serve_page_n_predictions \
  output/page3_predictions
```

Mở địa chỉ localhost được in trong terminal để xem ảnh, bbox, role và link
KEY → VALUE.

## 9. Pipeline TABLE 01

TABLE 01 là nhánh riêng, không phụ thuộc vào training Layout model page 3+.

### 9.1. Crop TABLE 01

```bash
python -m scripts.data.split_table_01
```

### 9.2. Tạo ground truth

Đảm bảo biến môi trường `OPENAI_API_KEY` đã được thiết lập, sau đó xem tham số
của script:

```bash
python -m scripts.tools.create_table_01_ground_truth --help
```

### 9.3. Đánh giá TABLE 01

```bash
python -m scripts.tools.evaluate_table_01
```

Script đọc prediction và ground truth TABLE 01, sau đó tạo báo cáo HTML.

## 10. Công cụ nghiên cứu khác

| Tệp | Khi sử dụng |
|---|---|
| `scripts/tools/experiment_detector_page3.py` | So sánh nhiều cấu hình text detector. |
| `scripts/tools/serve_rule_induction.py` | Xem và xác nhận kết quả Rule Induction. |
| `scripts/data/group_table_01_sources_by_relay.py` | Nhóm source TABLE 01 theo tên rơ-le. |

Các công cụ này không bắt buộc trong luồng training/inference thông thường.

## 11. Thứ tự thực hiện ngắn gọn

```bash
# 1. Đặt PDF vào data/pdf/
python -m scripts.data.build_page3plus_dataset

# 2. Tạo và xác nhận annotation
python -m scripts.annotation.annotate_page3 <rule_output_directory>
python -m scripts.annotation.serve_page3_annotations <rule_output_directory>

# 3. Chia dataset nếu cần
python -m scripts.data.split_page3_dataset \
  --dataset <annotations.json> \
  --output <split_output>

# 4. Huấn luyện
python -m scripts.training.train_page3_text_layout train \
  --dataset <train_dataset> \
  --validation-dataset <val_dataset> \
  --output <checkpoint_directory>

# 5. Inference
python -m scripts.inference.infer_page_n_text_layout \
  --pdf-directory data/pdf \
  --checkpoint <checkpoint_directory> \
  --output <prediction_directory>

# 6. Lọc và xem prediction
python -m scripts.inference.filter_page_n_predictions \
  <prediction_directory>/predictions.json
python -m scripts.tools.serve_page_n_predictions <prediction_directory>
```

Các đường dẫn `<...>` là giá trị cần thay theo dữ liệu và checkpoint thực tế.
