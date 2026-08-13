import paddleocr
print(f"PaddleOCR version: {paddleocr.__version__}")

# Có thể dùng file này để kiểm tra thêm PaddlePaddle và GPU khi chạy engine paddle_static cục bộ.
import paddle
# Mục đích: kiểm tra nhanh việc cài đặt PaddleOCR và PaddlePaddle trong môi trường hiện tại.
# Chức năng: in phiên bản PaddleOCR/PaddlePaddle và kiểm tra khả năng dùng GPU.
# Luồng xử lý: import thư viện → đọc version → in trạng thái CUDA và số lượng GPU.
# Cấu hình/parameter: không có tham số chạy; kết quả phụ thuộc môi trường Python hiện tại.
# Quan hệ: script kiểm tra độc lập, không được pipeline OCR import.
# Input/output: không nhận input; chỉ in thông tin môi trường ra console.
# Contract: giữ các thuộc tính version, is_compiled_with_cuda() và device.cuda.device_count().
print(f"Paddle version: {paddle.__version__}")
print(f"GPU available: {paddle.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")
