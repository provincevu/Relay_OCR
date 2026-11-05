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

## API OCR local

API nhận PDF, lưu theo job và đưa vào Redis Queue. Worker chạy OCR tuần tự
bằng pipeline hiện có rồi ghi JSON kết quả vào thư mục của job.

### Khởi động Redis

Đảm bảo Docker Desktop đang chạy, sau đó mở PowerShell tại thư mục repo:

```powershell
docker compose up -d redis
docker compose ps
docker compose exec redis redis-cli ping
```

Lệnh cuối phải trả `PONG`. Redis chỉ publish cổng trên `127.0.0.1` và dùng
named volume `relay_ocr_redis_data` để giữ dữ liệu.

### Cài dependencies và chạy API/worker

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OCR_API_TOKEN = "set-the-same-secret-in-the-client-terminal"
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 18010
```

Token chỉ có hiệu lực trong cửa sổ PowerShell hiện tại. Không commit token thật
vào repo. API mặc định chỉ nghe trên loopback và dùng cổng `18010`.

Mở PowerShell thứ hai tại root repo để chạy worker. API và worker phải cùng
`REDIS_URL`, `OCR_JOB_ROOT`, `OCR_CONFIG_PATH` nếu bạn đã đổi các giá trị mặc
định:

```powershell
.\.venv\Scripts\python.exe -m src.api.worker
```

Worker dùng RQ `SimpleWorker` cùng `TimerDeathPenalty` trên Windows và xử lý
mỗi lần một job. Model OCR
được nạp trong worker khi job đầu tiên bắt đầu, sau đó dùng lại cho các job sau.

Mỗi thành phần chạy trong cửa sổ/process riêng: Redis trong Docker Desktop, API
trong một PowerShell, worker trong PowerShell thứ hai, và các lệnh client trong
PowerShell thứ ba (hoặc một cửa sổ đang rảnh). API và client phải đặt cùng
`OCR_API_TOKEN`; worker không cần token này.

### Gọi API

Mở PowerShell thứ hai tại repo, đặt cùng token rồi gửi PDF:

```powershell
$env:OCR_API_TOKEN = "replace-this-with-a-local-secret"
$headers = @{ Authorization = "Bearer $env:OCR_API_TOKEN" }
$job = curl.exe -sS -X POST `
  http://127.0.0.1:18010/api/v1/jobs `
  -H "Authorization: Bearer $env:OCR_API_TOKEN" `
  -F "file=@C:\path\to\sample.pdf;type=application/pdf" | ConvertFrom-Json
$job
Invoke-RestMethod "http://127.0.0.1:18010/api/v1/jobs/$($job.job_id)" -Headers $headers
Invoke-RestMethod "http://127.0.0.1:18010/api/v1/jobs/$($job.job_id)/result" -Headers $headers
```

API trả `202 Accepted` cùng `job_id`. Khi worker hoàn tất, polling trả trạng thái
`succeeded` và endpoint result trả JSON OCR. Health checks không cần token:

```powershell
Invoke-RestMethod http://127.0.0.1:18010/health
```

### Retention và cleanup

RQ giữ metadata của job thành công/thất bại trong `JOB_RETENTION_DAYS` ngày
(mặc định 7 ngày). Cleanup file job được gọi thủ công bằng module bên dưới.
Lệnh chỉ xét thư mục job quá hạn; job queued/running được giữ lại.

Có thể xem trước các thư mục đủ điều kiện bị dọn bằng lệnh sau; mặc định lệnh
này không xóa dữ liệu:

```powershell
.\.venv\Scripts\python.exe -m src.api.cleanup
```

Khi cần dọn thủ công, trước tiên xem trước với đúng `OCR_JOB_ROOT` và
`JOB_RETENTION_DAYS`. Chỉ thêm `--apply` khi danh sách preview đúng:

```powershell
.\.venv\Scripts\python.exe -m src.api.cleanup --apply
```

Có thể truyền `--job-root` và `--retention-days` để kiểm tra trên một thư mục
thử nghiệm riêng. API không cung cấp endpoint xóa job.
