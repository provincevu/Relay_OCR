from pathlib import Path
from paddleocr import TextDetection

detector = TextDetection(
    model_name="PP-OCRv6_medium_det",
    device="cpu",
    engine="paddle_static",
    enable_mkldnn=False,
    cpu_threads=2,
)

input_dir = Path("./data/image/")
image_files = list(input_dir.glob("*"))

print(f"Tìm thấy {len(image_files)} file")

for i, image_path in enumerate(image_files, start=1):
    print(f"[{i}/{len(image_files)}] Đang xử lý: {image_path.name}")

    results = detector.predict(
        str(image_path),
        limit_side_len=1600,
        limit_type="max",
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=2.0,
    )

    for result in results:
        result.print()
        result.save_to_img("./output/image/")
        result.save_to_json("./output/json/")

    print(f"[{i}/{len(image_files)}] Hoàn thành: {image_path.name}")