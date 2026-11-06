# Relay OCR API Contract

Tài liệu này mô tả HTTP contract được triển khai trong `src/api/app.py`.
Đây là API nhận PDF bất đồng bộ: client gửi PDF, lưu `job_id`, poll trạng thái,
rồi tải JSON OCR khi job thành công.

## Thông tin chung

- Base URL mặc định khi chạy local: `http://127.0.0.1:18010`.
- Encoding: JSON dùng UTF-8; upload dùng `multipart/form-data`.
- Các endpoint `/api/v1/jobs` yêu cầu header:

  ```http
  Authorization: Bearer <OCR_API_TOKEN>
  ```

- `/health` không yêu cầu token.
- `job_id` do OCR API tạo; client cần lưu giá trị này.
- Hiện không có endpoint liệt kê, retry hoặc xóa job.

## `GET /health`

Health check kết nối Redis bằng lệnh `PING`.

Thành công — `200 OK`:

```json
{"status": "healthy"}
```

Nếu Redis không khả dụng, request sẽ lỗi; endpoint hiện không chuyển lỗi Redis
sang một response JSON riêng.

## `POST /api/v1/jobs`

Gửi một file PDF vào queue để xử lý bất đồng bộ.

### Request

- Content-Type: `multipart/form-data`
- Field bắt buộc: `file` (một PDF)
- Authorization: Bearer token

```bash
curl -X POST "http://127.0.0.1:18010/api/v1/jobs" \
  -H "Authorization: Bearer <OCR_API_TOKEN>" \
  -F "file=@sample.pdf;type=application/pdf"
```

API kiểm tra chữ ký `%PDF-`, khả năng đọc PDF và số trang. Mặc định giới hạn
upload là 100 MB và tối đa 300 trang; cấu hình qua `MAX_UPLOAD_MB` và
`MAX_PDF_PAGES`.

### Response thành công

`202 Accepted`:

```json
{
  "job_id": "65a9e989d81845398887eb72de104343",
  "status": "queued",
  "page_count": 11
}
```

Response hiện không trả `status_url` hoặc `result_url`. Client tạo URL từ
`job_id` theo các endpoint bên dưới.

### Lỗi

FastAPI trả body dạng `{"detail": "..."}` cho các lỗi HTTP do API phát sinh.

| HTTP status | Ý nghĩa |
|---|---|
| `400 Bad Request` | PDF rỗng, không đọc được, hoặc số trang không trong giới hạn. |
| `401 Unauthorized` | Token thiếu, không hợp lệ hoặc chưa cấu hình trên API. |
| `413 Payload Too Large` | File vượt `MAX_UPLOAD_MB`. |
| `415 Unsupported Media Type` | Nội dung không có chữ ký PDF `%PDF-`. |

## `GET /api/v1/jobs/{job_id}`

Trả trạng thái job cùng metadata upload. Yêu cầu Bearer token.

```http
GET /api/v1/jobs/65a9e989d81845398887eb72de104343
Authorization: Bearer <OCR_API_TOKEN>
```

Response — `200 OK`:

```json
{
  "job_id": "65a9e989d81845398887eb72de104343",
  "status": "running",
  "source_filename": "sample.pdf",
  "page_count": 11,
  "created_at": "2026-11-06T03:57:59.921191+00:00"
}
```

`created_at` là thời điểm tạo job dạng ISO 8601 UTC. Các trạng thái API:

| Status | Ý nghĩa |
|---|---|
| `queued` | Đang chờ worker xử lý. |
| `running` | Worker đang xử lý. |
| `succeeded` | OCR hoàn tất thành công. |
| `failed` | Job kết thúc với lỗi. |

Lỗi chính: `401 Unauthorized` nếu token không hợp lệ; `404 Not Found` nếu
job không còn được tìm thấy trong Redis. Response hiện không có trường mô tả
chi tiết nguyên nhân job thất bại.

## `GET /api/v1/jobs/{job_id}/result`

Trả JSON OCR trực tiếp, không bọc trong envelope HTTP tùy chỉnh. Yêu cầu Bearer
token và chỉ khả dụng sau khi trạng thái RQ là finished.

```http
GET /api/v1/jobs/65a9e989d81845398887eb72de104343/result
Authorization: Bearer <OCR_API_TOKEN>
```

- `200 OK`: body là nội dung `result.json`.
- `401 Unauthorized`: token thiếu hoặc sai.
- `404 Not Found`: job không tồn tại hoặc không tìm thấy file kết quả.
- `409 Conflict`: job chưa thành công (đang chờ, đang chạy hoặc thất bại).

Hình dạng tổng quát kết quả hiện tại:

```json
{
  "schema_version": 1,
  "source_pdf": {
    "name": "sample.pdf",
    "page_count": 11
  },
  "documents": [
    {
      "document_id": "0001",
      "source_page_start": 1,
      "source_page_end": 11,
      "page1": {},
      "page3_plus": []
    }
  ],
  "job_id": "65a9e989d81845398887eb72de104343"
}
```

Các trường `page1` và `page3_plus` phụ thuộc kết quả pipeline và có thể chứa
cấu trúc lồng nhau; client không nên giả định mọi document đều có dữ liệu cho
cả hai trường.

## Luồng gọi phía client

1. Gửi `POST /api/v1/jobs` với PDF và lưu `job_id` trong hệ thống chính.
2. Poll `GET /api/v1/jobs/{job_id}` theo chu kỳ do client chọn.
3. Khi status là `succeeded`, gọi `GET /api/v1/jobs/{job_id}/result`.
4. Khi status là `failed`, xử lý như job lỗi. API hiện không cung cấp retry tự
   động hay thông tin lỗi chi tiết trong response status.

Job metadata trong Redis và file job/kết quả trên đĩa được giữ theo cấu hình
retention; mặc định là 7 ngày. API không có endpoint liệt kê job hoặc callback.

## Cấu hình server liên quan

| Biến môi trường | Mặc định | Tác dụng |
|---|---|---|
| `OCR_API_TOKEN` | Không có | Bearer token bắt buộc cho endpoint job. |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis dùng bởi API và worker. |
| `RQ_QUEUE_NAME` | `ocr` | Tên queue RQ. |
| `OCR_JOB_ROOT` | `output/ocr_api/jobs` | Thư mục lưu PDF, output trung gian và result. |
| `MAX_UPLOAD_MB` | `100` | Dung lượng upload tối đa. |
| `MAX_PDF_PAGES` | `300` | Số trang tối đa. |
| `JOB_RETENTION_DAYS` | `7` | Thời hạn lưu job/result. |
| `JOB_TIMEOUT_SECONDS` | `86400` | Timeout job RQ. |

API và worker phải dùng chung Redis, `OCR_JOB_ROOT` và cấu hình OCR để worker
đọc được PDF do API lưu.
