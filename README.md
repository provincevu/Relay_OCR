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

## API OCR local (Phase 1)

API nhận PDF, lưu theo job và đưa vào Redis Queue. Trong Phase 1, worker OCR
chưa được nối; job sẽ ở trạng thái `queued` cho đến khi hoàn thành Phase 2.

### Khởi động Redis

Đảm bảo Docker Desktop đang chạy, sau đó mở PowerShell tại thư mục repo:

```powershell
docker compose up -d redis
docker compose ps
docker compose exec redis redis-cli ping
```

Lệnh cuối phải trả `PONG`. Redis chỉ publish cổng trên `127.0.0.1` và dùng
named volume `relay_ocr_redis_data` để giữ dữ liệu.

### Cài dependencies và chạy API

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OCR_API_TOKEN = "replace-this-with-a-local-secret"
.\.venv\Scripts\python.exe -m uvicorn scr.api.app:app --host 127.0.0.1 --port 8010
```

Token chỉ có hiệu lực trong cửa sổ PowerShell hiện tại. Không commit token thật
vào repo. API mặc định chỉ nghe trên loopback.

### Gọi API

Mở PowerShell thứ hai tại repo, đặt cùng token rồi gửi PDF:

```powershell
$env:OCR_API_TOKEN = "replace-this-with-a-local-secret"
$headers = @{ Authorization = "Bearer $env:OCR_API_TOKEN" }
$job = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8010/api/v1/jobs `
  -Headers $headers `
  -Form @{ file = Get-Item "C:\path\to\sample.pdf" }
$job
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/jobs/$($job.job_id)" -Headers $headers
```

API trả `202 Accepted` cùng `job_id`. Polling và lấy kết quả OCR hoạt động sau
khi hoàn thành Phase 2. Health checks không cần token:

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health/live
Invoke-RestMethod http://127.0.0.1:8010/health/ready
```
