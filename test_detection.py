from pathlib import Path
from paddleocr import TextDetection

# Mục đích: kiểm tra trực tiếp chức năng text detection của PaddleOCR trên toàn bộ ảnh đầu vào.
# Chức năng: khởi tạo TextDetection, chạy detection và lưu ảnh/json kết quả.
# Luồng xử lý: đọc data/image/ → predict từng ảnh → lưu vào output/image/ và output/json/.
# Cấu hình/parameter: model PP-OCRv6_medium_det, CPU, paddle_static, các ngưỡng limit/thresh/box_thresh/unclip_ratio.
# Quan hệ: script chạy độc lập, không import Detector hoặc OCRService.
# Input/output: input là các file trong data/image/; output là ảnh đã vẽ box và JSON tọa độ detection.
# Contract: giữ các tham số predict và cách lấy result.save_to_img()/save_to_json() để đối chiếu behavior detection.

detector = TextDetection(
    model_name="PP-OCRv6_medium_det",  # Model dùng để phát hiện vùng chứa text
    device="cpu",                      # Thiết bị chạy inference: cpu, gpu:0,...
    engine="paddle_static",            # Engine inference; static thường dùng cho deployment

    cpu_threads=2,                     # Số CPU thread dùng cho inference

    enable_mkldnn=False,              # Bật/tắt MKL-DNN để tăng tốc inference trên CPU
    # enable_hpi=True,                 # Bật High Performance Inference để tối ưu tốc độ
    # use_tensorrt=True,               # Dùng TensorRT để tăng tốc trên GPU NVIDIA
    # precision="fp32",                # Độ chính xác tính toán: fp32 hoặc fp16
    # mkldnn_cache_capacity=10,        # Số shape được cache khi dùng MKL-DNN
)

input_dir = Path("./data/image/")      # Thư mục chứa ảnh đầu vào
image_files = list(input_dir.glob("*"))  # Lấy toàn bộ file trong thư mục

print(f"Tìm thấy {len(image_files)} file")

for i, image_path in enumerate(image_files, start=1):
    print(f"[{i}/{len(image_files)}] Đang xử lý: {image_path.name}")

    results = detector.predict(
        str(image_path),               # Đường dẫn ảnh cần detection

        limit_side_len=1600,           # Giới hạn kích thước ảnh trước inference
        limit_type="max",              # Giới hạn cạnh dài nhất không vượt limit_side_len
        thresh=0.3,                    # Ngưỡng pixel để xác định pixel thuộc vùng text (0-1)
        box_thresh=0.6,                # Ngưỡng confidence để giữ lại bounding box (0-1)
        unclip_ratio=2.0,              # Mức mở rộng bounding box quanh vùng text (infinite, nhưng 2 là hợp lý)

        # input_shape=[3, 960, 960],    # Ép kích thước C,H,W đầu vào của detection model
    )

    for result in results:
        # result.print()                                # In kết quả detection ra console
        result.save_to_img("./output/image/")         # Lưu ảnh đã vẽ bounding box
        result.save_to_json("./output/json/")         # Lưu tọa độ/kết quả detection dạng JSON

    print(f"[{i}/{len(image_files)}] Hoàn thành: {image_path.name}")
