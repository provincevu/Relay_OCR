# Mục đích: so sánh nhiều cấu hình PP-OCR text detector trên một ảnh page 3+.
# Chức năng: chạy từng cấu hình, lưu polygon detection, ảnh overlay và HTML đánh giá.
# Luồng xử lý: ảnh đầu vào → detector theo từng cấu hình → JSON/PNG → index.html.
# Cấu hình/parameter: baseline và các giá trị thay thế cho limit_side_len, thresh,
# box_thresh và unclip_ratio.
# Quan hệ: dùng cùng model và tham số detector với test_table_key_value_page3_plus.py.
# Input/output: đọc một ảnh P_005 page 3; ghi kết quả vào thư mục output.
# Contract: mỗi thử nghiệm chỉ thay đổi một tham số so với baseline.

import argparse
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextDetection


BASE_CONFIG = {
    "limit_side_len": 1600,
    "limit_type": "max",
    "thresh": 0.3,
    "box_thresh": 0.6,
    "unclip_ratio": 1.5,
}


def build_experiments():
    """Tạo danh sách thử nghiệm, mỗi mẫu chỉ đổi một tham số."""
    experiments = [
        {
            "name": "baseline",
            "parameter": "baseline",
            "value": "-",
            "config": dict(BASE_CONFIG),
        }
    ]
    parameter_values = {
        "limit_side_len": [1200, 2000, 2400, 3200],
        "thresh": [0.2, 0.25, 0.35, 0.4],
        "box_thresh": [0.5, 0.7, 0.8],
        "unclip_ratio": [1.1, 1.25, 1.75, 2.0],
    }
    for parameter, values in parameter_values.items():
        for value in values:
            config = dict(BASE_CONFIG)
            config[parameter] = value
            experiments.append({
                "name": f"{parameter}_{str(value).replace('.', '_')}",
                "parameter": parameter,
                "value": value,
                "config": config,
            })
    return experiments


def detect_polygons(detector, image, config):
    """Chạy detector và chuẩn hóa polygon về danh sách tọa độ số thực."""
    polygons = []
    detection_results = detector.predict(input=image, **config)
    for detection_result in detection_results:
        for polygon in detection_result["dt_polys"]:
            polygons.append([
                [float(point[0]), float(point[1])]
                for point in polygon
            ])
    return polygons


def draw_polygons(image, polygons):
    """Vẽ polygon và số thứ tự detection lên ảnh overlay."""
    visualization = image.copy()
    for index, polygon in enumerate(polygons, start=1):
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(
            visualization,
            [points],
            isClosed=True,
            color=(0, 180, 0),
            thickness=3,
        )
        x_min = int(points[:, 0].min())
        y_min = int(points[:, 1].min())
        cv2.putText(
            visualization,
            str(index),
            (x_min, max(y_min - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return visualization


def write_json(output_path, image_path, experiment, polygons):
    """Lưu polygon và metadata của một thử nghiệm."""
    output_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "experiment": experiment,
                "polygons": polygons,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_html(image_path, experiment_results):
    """Tạo HTML mục lục và ảnh overlay của toàn bộ thử nghiệm."""
    sections = []
    navigation = []
    for result in experiment_results:
        experiment = result["experiment"]
        anchor = f"trial-{result['index']}"
        navigation.append(
            f'<li><a href="#{anchor}">{escape(experiment["name"])}'
            f' · {result["polygon_count"]} boxes</a></li>'
        )
        config_rows = "".join(
            f"<tr><th>{escape(str(key))}</th>"
            f"<td>{escape(str(value))}</td></tr>"
            for key, value in experiment["config"].items()
        )
        sections.append(
            f"""
<section id="{anchor}">
    <h2>{escape(experiment["name"])}</h2>
    <p class="summary">
        Thay đổi: <strong>{escape(str(experiment["parameter"]))}</strong>
        = <strong>{escape(str(experiment["value"]))}</strong>
        · Số polygon: <strong>{result["polygon_count"]}</strong>
        · <a href="{escape(result["json_name"])}">JSON</a>
    </p>
    <table><tbody>{config_rows}</tbody></table>
    <a href="{escape(result["image_name"])}" target="_blank">
        <img src="{escape(result["image_name"])}"
             alt="{escape(experiment["name"])}">
    </a>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Thử nghiệm PP-OCR detector</title>
    <style>
        body {{
            background: #f1f5f9;
            color: #172033;
            font-family: Arial, sans-serif;
            margin: 0;
        }}
        header {{
            background: #0f172a;
            color: white;
            padding: 20px 28px;
        }}
        main {{ margin: 0 auto; max-width: 1800px; padding: 20px; }}
        nav, section {{
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            margin-bottom: 18px;
            padding: 16px;
        }}
        nav {{ position: sticky; top: 0; z-index: 2; }}
        nav ul {{ columns: 3; margin: 8px 0 0; padding-left: 20px; }}
        li {{ margin: 4px 0; }}
        .summary {{ color: #475569; }}
        table {{ border-collapse: collapse; margin: 12px 0; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 5px 9px; text-align: left; }}
        th {{ background: #f8fafc; }}
        img {{
            border: 1px solid #94a3b8;
            display: block;
            height: auto;
            max-width: 100%;
        }}
        a {{ color: #1d4ed8; }}
        @media (max-width: 900px) {{ nav ul {{ columns: 1; }} }}
    </style>
</head>
<body>
<header>
    <h1>Thử nghiệm PP-OCR detector</h1>
    <p>{escape(str(image_path))}</p>
</header>
<main>
    <nav>
        <strong>Mục lục thử nghiệm</strong>
        <ul>{"".join(navigation)}</ul>
    </nav>
    {"".join(sections)}
</main>
</body>
</html>
"""


def parse_arguments():
    """Đọc đường dẫn ảnh và thư mục output."""
    parser = argparse.ArgumentParser(
        description="So sánh cấu hình PP-OCR detector trên một ảnh page 3.",
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/detector_experiments/P_005_page_3"),
    )
    return parser.parse_args()


def main():
    """Chạy toàn bộ thử nghiệm và tạo HTML trực quan."""
    arguments = parse_arguments()
    image_path = arguments.image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    output_directory = arguments.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    detector = TextDetection(
        model_name="PP-OCRv6_small_det",
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
        cpu_threads=4,
    )
    experiment_results = []
    experiments = build_experiments()
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"[{index}/{len(experiments)}] {experiment['name']}",
            flush=True,
        )
        polygons = detect_polygons(detector, image, experiment["config"])
        image_name = f"{index:02d}_{experiment['name']}.png"
        json_name = f"{index:02d}_{experiment['name']}.json"
        cv2.imwrite(
            str(output_directory / image_name),
            draw_polygons(image, polygons),
        )
        write_json(
            output_directory / json_name,
            image_path,
            experiment,
            polygons,
        )
        experiment_results.append({
            "index": index,
            "experiment": experiment,
            "polygon_count": len(polygons),
            "image_name": image_name,
            "json_name": json_name,
        })

    html_path = output_directory / "index.html"
    html_path.write_text(
        build_html(image_path, experiment_results),
        encoding="utf-8",
    )
    print(f"HTML đánh giá: {html_path}")


if __name__ == "__main__":
    main()
