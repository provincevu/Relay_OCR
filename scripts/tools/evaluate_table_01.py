# Mục đích: đánh giá dữ liệu TABLE 01 dự đoán so với ground truth.
# Chức năng: so sánh 16 field và tạo báo cáo HTML trực quan.
# Luồng xử lý: đọc JSON → lấy ảnh chung → chuẩn hóa giá trị → tính metric → ghi HTML.
# Cấu hình/parameter: đường dẫn input/output được khai báo ở cuối file.
# Quan hệ: dùng danh sách field trong config/table_01_fields.json.
# Input/output: đọc hai file JSON; ghi output/table_01_evaluation.html.
# Contract: chỉ so sánh ảnh có trong cả hai JSON; null chỉ khớp với null.

import json
import unicodedata
from html import escape
from pathlib import Path


class Table01Evaluator:
    """Đánh giá 16 field TABLE 01 và xuất báo cáo HTML."""

    def __init__(
        self,
        ground_truth_path,
        predictions_path,
        field_config_path,
        output_path,
    ):
        self.ground_truth_path = Path(ground_truth_path)
        self.predictions_path = Path(predictions_path)
        self.field_config_path = Path(field_config_path)
        self.output_path = Path(output_path)

    def evaluate(self):
        """So sánh các ảnh chung và trả kết quả đánh giá theo field."""
        ground_truth = json.loads(
            self.ground_truth_path.read_text(encoding="utf-8")
        )
        predictions = json.loads(
            self.predictions_path.read_text(encoding="utf-8")
        )
        field_config = json.loads(
            self.field_config_path.read_text(encoding="utf-8")
        )
        field_definitions = field_config["fields"]
        common_images = sorted(set(ground_truth) & set(predictions))

        image_results = []
        for image_name in common_images:
            field_results = []
            for field in field_definitions:
                field_key = field["key"]
                ground_truth_value = ground_truth[image_name][field_key]
                prediction_value = predictions[image_name][field_key]
                field_results.append(
                    self._compare_field(
                        field_key=field_key,
                        field_name=field["names"][0],
                        ground_truth_value=ground_truth_value,
                        prediction_value=prediction_value,
                    )
                )

            image_results.append(
                {
                    "image_name": image_name,
                    "field_results": field_results,
                    "normalized_matches": sum(
                        result["normalized_match"]
                        for result in field_results
                    ),
                }
            )

        return {
            "summary": self._build_summary(
                image_results,
                len(field_definitions),
            ),
            "field_metrics": self._build_field_metrics(
                image_results,
                field_definitions,
            ),
            "image_results": image_results,
        }

    def save_html(self, evaluation):
        """Ghi kết quả đánh giá thành một file HTML."""
        summary = evaluation["summary"]
        field_rows = "".join(
            self._build_field_metric_row(metric)
            for metric in evaluation["field_metrics"]
        )
        image_sections = "".join(
            self._build_image_section(
                image_result,
                summary["field_count"],
            )
            for image_result in evaluation["image_results"]
        )
        html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Đánh giá TABLE 01</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
.note {{ padding: 12px 16px; background: #eff6ff; border-left: 4px solid #2563eb; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }}
.card {{ min-width: 180px; padding: 14px; border: 1px solid #d1d5db; border-radius: 8px; }}
.value {{ display: block; margin-top: 6px; font-size: 24px; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; }}
th, td {{ padding: 8px; border: 1px solid #d1d5db; text-align: left; vertical-align: top; }}
th {{ background: #f3f4f6; }}
td.value-text {{ white-space: pre-wrap; word-break: break-word; }}
tr.match {{ background: #f0fdf4; }}
tr.mismatch {{ background: #fef2f2; }}
.ok {{ color: #15803d; font-weight: bold; }}
.error {{ color: #b91c1c; font-weight: bold; }}
.missing {{ color: #c2410c; font-weight: bold; }}
.extra {{ color: #7e22ce; font-weight: bold; }}
.null {{ color: #6b7280; font-style: italic; }}
details {{ margin: 12px 0; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; }}
summary {{ cursor: pointer; font-weight: bold; }}
</style>
</head>
<body>
<h1>Đánh giá kết quả TABLE 01</h1>
<div class="note">
Chỉ so sánh ảnh xuất hiện trong cả dữ liệu chuẩn (ground truth) và kết quả dự đoán
(prediction). Kết quả chuẩn hóa bỏ mọi ký tự không phải chữ hoặc số, không phân
biệt chữ hoa/thường nhưng vẫn giữ dấu tiếng Việt. Tỷ lệ lỗi ký tự (CER) chỉ tính
khi dữ liệu chuẩn có giá trị; CER càng thấp thì kết quả càng chính xác.
Riêng hai trường phiên bản, dấu chấm được giữ lại và tham gia so sánh.
Kết luận “Thiếu” nghĩa là kết quả dự đoán bị mất nội dung; “Thừa” nghĩa là kết
quả dự đoán có thêm nội dung. Trường hợp vừa thiếu vừa thừa được kết luận là “Sai”.
</div>

<div class="cards">
<div class="card">Ảnh được so sánh<span class="value">{summary["image_count"]}</span></div>
<div class="card">Tổng trường dữ liệu (field)<span class="value">{summary["comparison_count"]}</span></div>
<div class="card">Khớp sau chuẩn hóa (normalized exact)<span class="value">{self._format_rate(summary["normalized_match_rate"])}</span></div>
<div class="card">Khớp tuyệt đối (strict exact)<span class="value">{self._format_rate(summary["strict_match_rate"])}</span></div>
<div class="card">Tỷ lệ lỗi ký tự (CER)<span class="value">{self._format_rate(summary["cer"])}</span></div>
<div class="card">Ảnh đúng 16 trường dữ liệu<span class="value">{summary["fully_matched_images"]}/{summary["image_count"]}</span></div>
<div class="card">Kết quả thiếu<span class="value">{summary["missing_count"]}</span></div>
<div class="card">Kết quả thừa<span class="value">{summary["extra_count"]}</span></div>
</div>

<h2>Trạng thái không có giá trị (null)</h2>
<table>
<thead><tr><th>Cả hai cùng null</th><th>Chỉ dữ liệu chuẩn null</th><th>Chỉ kết quả dự đoán null</th></tr></thead>
<tbody><tr>
<td>{summary["both_null_count"]}</td>
<td>{summary["ground_truth_null_only_count"]}</td>
<td>{summary["prediction_null_only_count"]}</td>
</tr></tbody>
</table>

<h2>Kết quả theo trường dữ liệu (field)</h2>
<table>
<thead><tr>
<th>Trường dữ liệu (field)</th>
<th>Khớp sau chuẩn hóa (normalized exact)</th>
<th>Khớp tuyệt đối (strict exact)</th>
<th>Thiếu</th><th>Thừa</th>
<th>Tỷ lệ lỗi ký tự (CER)</th>
</tr></thead>
<tbody>{field_rows}</tbody>
</table>

<h2>Chi tiết theo ảnh</h2>
{image_sections}
</body>
</html>
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        print(f"Báo cáo đánh giá TABLE 01: {self.output_path}")

    def _compare_field(
        self,
        field_key,
        field_name,
        ground_truth_value,
        prediction_value,
    ):
        """So sánh một field theo giá trị gốc và giá trị đã chuẩn hóa."""
        keep_dot = field_key in {
            "phien_ban_ro_le",
            "phien_ban_phan_mem",
        }
        normalized_ground_truth = self._normalize_value(
            ground_truth_value,
            keep_dot,
        )
        normalized_prediction = self._normalize_value(
            prediction_value,
            keep_dot,
        )
        strict_match = ground_truth_value == prediction_value
        normalized_match = (
            normalized_ground_truth == normalized_prediction
            if ground_truth_value is not None
            and prediction_value is not None
            else ground_truth_value is None and prediction_value is None
        )
        conclusion = self._classify_result(
            normalized_ground_truth,
            normalized_prediction,
            normalized_match,
        )

        edit_distance = None
        reference_length = None
        cer = None
        if ground_truth_value is not None:
            edit_distance = self._levenshtein_distance(
                normalized_ground_truth,
                normalized_prediction or "",
            )
            reference_length = max(len(normalized_ground_truth), 1)
            cer = edit_distance / reference_length

        return {
            "field_key": field_key,
            "field_name": field_name,
            "ground_truth": ground_truth_value,
            "prediction": prediction_value,
            "normalized_ground_truth": normalized_ground_truth,
            "normalized_prediction": normalized_prediction,
            "strict_match": strict_match,
            "normalized_match": normalized_match,
            "conclusion": conclusion,
            "edit_distance": edit_distance,
            "reference_length": reference_length,
            "cer": cer,
        }

    def _build_summary(self, image_results, field_count):
        """Tổng hợp metric trên toàn bộ ảnh được so sánh."""
        field_results = [
            field_result
            for image_result in image_results
            for field_result in image_result["field_results"]
        ]
        comparison_count = len(field_results)
        strict_matches = sum(
            result["strict_match"] for result in field_results
        )
        normalized_matches = sum(
            result["normalized_match"] for result in field_results
        )
        total_edit_distance = sum(
            result["edit_distance"] or 0 for result in field_results
        )
        total_reference_length = sum(
            result["reference_length"] or 0 for result in field_results
        )

        return {
            "image_count": len(image_results),
            "field_count": field_count,
            "comparison_count": comparison_count,
            "strict_match_rate": self._rate(
                strict_matches,
                comparison_count,
            ),
            "normalized_match_rate": self._rate(
                normalized_matches,
                comparison_count,
            ),
            "cer": self._rate(
                total_edit_distance,
                total_reference_length,
            ),
            "fully_matched_images": sum(
                image_result["normalized_matches"] == field_count
                for image_result in image_results
            ),
            "missing_count": sum(
                result["conclusion"] == "Thiếu"
                for result in field_results
            ),
            "extra_count": sum(
                result["conclusion"] == "Thừa"
                for result in field_results
            ),
            "both_null_count": sum(
                result["ground_truth"] is None
                and result["prediction"] is None
                for result in field_results
            ),
            "ground_truth_null_only_count": sum(
                result["ground_truth"] is None
                and result["prediction"] is not None
                for result in field_results
            ),
            "prediction_null_only_count": sum(
                result["ground_truth"] is not None
                and result["prediction"] is None
                for result in field_results
            ),
        }

    def _build_field_metrics(self, image_results, field_definitions):
        """Tổng hợp exact match và CER riêng cho từng field."""
        metrics = []
        for field in field_definitions:
            field_key = field["key"]
            results = [
                field_result
                for image_result in image_results
                for field_result in image_result["field_results"]
                if field_result["field_key"] == field_key
            ]
            total_edit_distance = sum(
                result["edit_distance"] or 0 for result in results
            )
            total_reference_length = sum(
                result["reference_length"] or 0 for result in results
            )
            metrics.append(
                {
                    "field_key": field_key,
                    "field_name": field["names"][0],
                    "comparison_count": len(results),
                    "strict_matches": sum(
                        result["strict_match"] for result in results
                    ),
                    "normalized_matches": sum(
                        result["normalized_match"] for result in results
                    ),
                    "missing_count": sum(
                        result["conclusion"] == "Thiếu"
                        for result in results
                    ),
                    "extra_count": sum(
                        result["conclusion"] == "Thừa"
                        for result in results
                    ),
                    "cer": self._rate(
                        total_edit_distance,
                        total_reference_length,
                    ),
                }
            )

        return metrics

    def _build_field_metric_row(self, metric):
        """Tạo một hàng HTML cho metric của một field."""
        total = metric["comparison_count"]
        normalized_rate = self._rate(metric["normalized_matches"], total)
        strict_rate = self._rate(metric["strict_matches"], total)
        return (
            "<tr>"
            f"<td>{escape(metric['field_name'])}<br><small>{escape(metric['field_key'])}</small></td>"
            f"<td>{metric['normalized_matches']}/{total} "
            f"({self._format_rate(normalized_rate)})</td>"
            f"<td>{metric['strict_matches']}/{total} "
            f"({self._format_rate(strict_rate)})</td>"
            f"<td>{metric['missing_count']}</td>"
            f"<td>{metric['extra_count']}</td>"
            f"<td>{self._format_rate(metric['cer'])}</td>"
            "</tr>"
        )

    def _build_image_section(self, image_result, field_count):
        """Tạo bảng HTML chi tiết cho một ảnh."""
        rows = "".join(
            self._build_field_result_row(result)
            for result in image_result["field_results"]
        )
        return f"""
<details>
<summary>{escape(image_result["image_name"])} — {image_result["normalized_matches"]}/{field_count} trường dữ liệu đúng</summary>
<table>
<thead><tr>
<th>Trường dữ liệu (field)</th>
<th>Dữ liệu chuẩn (ground truth)</th>
<th>Kết quả dự đoán (prediction)</th>
<th>Dữ liệu chuẩn sau chuẩn hóa</th>
<th>Kết quả dự đoán sau chuẩn hóa</th>
<th>Kết quả</th><th>Tỷ lệ lỗi ký tự (CER)</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</details>
"""

    def _build_field_result_row(self, result):
        """Tạo một hàng HTML cho kết quả so sánh một field."""
        row_class = "match" if result["normalized_match"] else "mismatch"
        status_classes = {
            "Đúng": "ok",
            "Thiếu": "missing",
            "Thừa": "extra",
            "Sai": "error",
        }
        conclusion = result["conclusion"]
        return (
            f'<tr class="{row_class}">'
            f"<td>{escape(result['field_name'])}<br>"
            f"<small>{escape(result['field_key'])}</small></td>"
            f'<td class="value-text">{self._format_value(result["ground_truth"])}</td>'
            f'<td class="value-text">{self._format_value(result["prediction"])}</td>'
            f'<td class="value-text">{self._format_value(result["normalized_ground_truth"])}</td>'
            f'<td class="value-text">{self._format_value(result["normalized_prediction"])}</td>'
            f'<td class="{status_classes[conclusion]}">{conclusion}</td>'
            f"<td>{self._format_rate(result['cer'])}</td>"
            "</tr>"
        )

    def _normalize_value(self, value, keep_dot=False):
        """Loại ký tự đặc biệt và không phân biệt chữ hoa/thường."""
        if value is None:
            return None

        value = value.replace("\\n", " ")
        value = value.replace("\\r", " ")
        value = value.replace("\\t", " ")
        normalized = unicodedata.normalize("NFC", value).casefold()
        return "".join(
            character
            for character in normalized
            if character.isalnum() or keep_dot and character == "."
        )

    def _classify_result(
        self,
        ground_truth_value,
        prediction_value,
        normalized_match,
    ):
        """Kết luận đúng, thiếu, thừa hoặc sai cho một field."""
        if normalized_match:
            return "Đúng"

        ground_truth_text = ground_truth_value or ""
        prediction_text = prediction_value or ""
        if self._is_subsequence(prediction_text, ground_truth_text):
            return "Thiếu"
        if self._is_subsequence(ground_truth_text, prediction_text):
            return "Thừa"
        return "Sai"

    def _is_subsequence(self, candidate, reference):
        """Kiểm tra candidate có giữ đúng thứ tự ký tự trong reference không."""
        reference_characters = iter(reference)
        return all(
            character in reference_characters
            for character in candidate
        )

    def _levenshtein_distance(self, reference, prediction):
        """Tính khoảng cách chỉnh sửa giữa hai chuỗi đã chuẩn hóa."""
        previous_row = list(range(len(prediction) + 1))
        for reference_index, reference_character in enumerate(reference, start=1):
            current_row = [reference_index]
            for prediction_index, prediction_character in enumerate(
                prediction,
                start=1,
            ):
                insertion = current_row[prediction_index - 1] + 1
                deletion = previous_row[prediction_index] + 1
                substitution = previous_row[prediction_index - 1] + (
                    reference_character != prediction_character
                )
                current_row.append(min(insertion, deletion, substitution))
            previous_row = current_row

        return previous_row[-1]

    def _rate(self, numerator, denominator):
        """Tính tỷ lệ và trả 0 khi không có mẫu so sánh."""
        return numerator / denominator if denominator else 0.0

    def _format_rate(self, value):
        """Định dạng tỷ lệ để hiển thị trong HTML."""
        return "—" if value is None else f"{value * 100:.2f}%"

    def _format_value(self, value):
        """Escape giá trị và hiển thị rõ null trong HTML."""
        if value is None:
            return '<span class="null">không có giá trị (null)</span>'
        return escape(value)


if __name__ == "__main__":
    evaluator = Table01Evaluator(
        ground_truth_path=Path("./output/table_01_ground_truth.json"),
        predictions_path=Path("./output/layout/table_01_predictions.json"),
        field_config_path=Path("./config/table_01_fields.json"),
        output_path=Path("./output/table_01_evaluation.html"),
    )
    evaluation_result = evaluator.evaluate()
    evaluator.save_html(evaluation_result)
