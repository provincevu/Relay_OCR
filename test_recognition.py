from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from paddleocr import TextDetection
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


detector = TextDetection(
    model_name="PP-OCRv6_medium_det",  # Mô hình dùng để phát hiện vùng chứa text
    device="cpu",                      # Thiết bị chạy inference: cpu, gpu:0,...
    engine="paddle_static",            # Engine inference; static thường dùng cho deployment

    cpu_threads=2,                     # Số CPU thread dùng cho inference

    enable_mkldnn=False,               # Bật/tắt MKL-DNN để tăng tốc inference trên CPU
)

config = Cfg.load_config_from_name("vgg_transformer")  # Mô hình VietOCR VGG Transformer
config["device"] = "cpu"                                # Thiết bị chạy recognition
recognitor = Predictor(config)

input_dir = Path("./data/image/")       # Thư mục chứa ảnh đầu vào
image_files = list(input_dir.glob("*")) # Lấy toàn bộ file trong thư mục

print(f"Tìm thấy {len(image_files)} file")

for i, image_path in enumerate(image_files, start=1):
    print(f"[{i}/{len(image_files)}] Đang xử lý: {image_path.name}")

    image = cv2.imread(str(image_path))
    results = detector.predict(
        str(image_path),               # Đường dẫn ảnh cần detection

        limit_side_len=1600,           # Giới hạn kích thước ảnh trước inference
        limit_type="max",              # Giới hạn cạnh dài nhất không vượt limit_side_len
        thresh=0.3,                    # Ngưỡng pixel để xác định pixel thuộc vùng text (0-1)
        box_thresh=0.6,                # Ngưỡng confidence để giữ lại bounding box (0-1)
        unclip_ratio=2.0,              # Mức mở rộng bounding box quanh vùng text
    )

    for result in results:
        for box in result["dt_polys"]:
            box = np.array(box, dtype=np.float32)

            # Cắt vùng chữ theo hình chữ nhật bao quanh polygon detection.
            x_min, y_min = box.min(axis=0).astype(int)
            x_max, y_max = box.max(axis=0).astype(int)
            crop = image[y_min:y_max, x_min:x_max]

            # VietOCR nhận ảnh PIL ở định dạng RGB.
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            text = recognitor.predict(Image.fromarray(crop))
            print(f"  - {text}")

    print(f"[{i}/{len(image_files)}] Hoàn thành: {image_path.name}")
