from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from paddleocr import TextDetection
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Mục đích: kiểm tra pipeline detection kết hợp recognition bằng VietOCR VGG Transformer.
# Chức năng: phát hiện polygon text, cắt từng vùng, nhận dạng và vẽ kết quả lên ảnh.
# Luồng xử lý: đọc data/image/ → PaddleOCR detection → cắt dt_polys → VietOCR → lưu output/recognition/.
# Cấu hình/parameter: detector dùng PP-OCRv6_medium_det; recognitor dùng vgg_transformer; các ngưỡng detection giữ như test_detection.py.
# Quan hệ: script chạy độc lập và là mẫu tham khảo cho Detector, Recognitor và OCRService.
# Input/output: input là ảnh trong data/image/; output là text trên console và ảnh có box/text trong output/recognition/.
# Contract: giữ dt_polys là polygon bốn điểm, crop chuyển BGR sang RGB trước khi đưa vào VietOCR.


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
output_dir = Path("./output/recognition/") # Thư mục lưu ảnh có kết quả recognition
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Tìm thấy {len(image_files)} file")

for i, image_path in enumerate(image_files, start=1):
    print(f"[{i}/{len(image_files)}] Đang xử lý: {image_path.name}")

    image = cv2.imread(str(image_path))
    output_image = image.copy()
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

            # Vẽ polygon và kết quả nhận dạng lên ảnh gốc.
            points = box.astype(np.int32)
            cv2.polylines(output_image, [points], True, (0, 255, 0), 2)
            cv2.putText(
                output_image,
                str(text),
                (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # Lưu ảnh gốc đã vẽ vùng detection và text recognition.
    cv2.imwrite(str(output_dir / image_path.name), output_image)

    print(f"[{i}/{len(image_files)}] Hoàn thành: {image_path.name}")
