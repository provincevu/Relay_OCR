# RELAY OCR

cung cấp pipeline OCR phiếu chỉnh định rơ-le cho các trạm biến áp.

## Tạo dataset page 3+

Render các page dữ liệu từ `data/pdf`, sau đó chia theo phiếu/PDF thành ba tập
`train`, `val`, `test`. Mỗi phiếu được shuffle riêng với seed 42 rồi chia page
theo rule: 2 page thì train/test mỗi tập một page; từ 3 page trở lên chia đều,
ưu tiên phần dư theo thứ tự train, test, val.

```bash
python -m scripts.data.build_page3plus_dataset
```

Kết quả nằm trong `data/page3plus_dataset/`, gồm ảnh PNG, `manifest.json` ở
gốc và manifest riêng trong từng split. Có thể thay đổi DPI hoặc seed:

```bash
python -m scripts.data.build_page3plus_dataset --dpi 400 --seed 42
```

Nếu chạy lại để tạo mới dataset, thêm `--overwrite`.

Script cần Ghostscript (`gswin64c` trên Windows) trong `PATH` hoặc truyền
đường dẫn bằng `--renderer`.

## Cài đặt

```bash
git clone https://github.com/provincevu/Relay_OCR.git
cd Relay_OCR

# Cài dependencies
pip install -r requirements.txt
```

## Kiểm tra cài đặt thành công chưa

```bash
python -c "import paddle; print(paddle.__version__)"
```

chỉ xuất ảnh khi ở chế độ debug
