# Mục đích: trích xuất field TABLE 01 từ text đã gán vào cell.
# Chức năng: khớp field name, sửa value bằng PP-OCR và trả metadata của cell.
# Luồng xử lý: cell theo hàng → field match → context → value → chuẩn hóa version.
# Cấu hình/parameter: nhận danh sách field từ table_01_fields.json.
# Quan hệ: được Table1Analysis gọi sau bước xử lý cell và text.
# Input/output: nhận danh sách cell; trả dict field key, value, coordinate và score.
# Contract: output luôn chứa đầy đủ field key trong cấu hình.

import re

from .table_cell_utils import group_boxes_into_rows
from .text_comparison import correct_value_with_ppocr


class TableFieldExtractor:
    """Trích xuất field TABLE 01 và metadata của cell chứa field."""

    # Khởi tạo extractor bằng danh sách field name được cấu hình.
    def __init__(self, field_definitions):
        """Lưu danh sách field dùng để khớp tên và context."""
        self.field_definitions = field_definitions

    # Tạo output rỗng nhưng vẫn giữ đầy đủ field key.
    def empty_fields(self):
        """Tạo field output rỗng theo đúng contract."""
        return {
            field["key"]: {
                "value": None,
                "coordinate": None,
                "score": None,
            }
            for field in self.field_definitions
        }

    # Kiểm tra text có chứa ít nhất một field name hợp lệ.
    def has_field_name(self, text):
        """Cho biết text có thể dùng để cứu cell score thấp hay không."""
        return bool(self._find_field_matches(text))

    # Trích xuất toàn bộ field từ danh sách cell đã OCR.
    def extract(self, cells):
        """Trả field value cùng coordinate và score của cell chứa field."""
        fields = self.empty_fields()
        rows = group_boxes_into_rows(cells)

        # Duyệt từng hàng để có context giữa các cell cùng hàng.
        for row in rows:
            parsed_cells = []
            row_context_keys = set()

            # Thu thập field match và context của toàn bộ hàng.
            for cell in row:
                matches = self._find_field_matches(cell["text"])
                parsed_cells.append((cell, matches))

                for match in matches:
                    candidate_keys = {
                        candidate["field"]["key"]
                        for candidate in match["candidates"]
                    }
                    if len(candidate_keys) == 1:
                        row_context_keys.update(candidate_keys)

            # Trích xuất value của từng field trong từng cell.
            for cell, matches in parsed_cells:
                cell_context_keys = set()
                for match in matches:
                    candidate_keys = {
                        candidate["field"]["key"]
                        for candidate in match["candidates"]
                    }
                    if len(candidate_keys) == 1:
                        cell_context_keys.update(candidate_keys)

                for match_index, match in enumerate(matches):
                    selected_candidate = self._resolve_field_candidate(
                        match,
                        cell_context_keys,
                        row_context_keys,
                    )

                    # Bỏ field chưa xác định được key duy nhất theo context.
                    if selected_candidate is None:
                        continue

                    value = self._extract_field_value(
                        cell["text"],
                        matches,
                        match_index,
                    )
                    # Chỉ sửa phần value, không sửa field name tiếng Việt.
                    if value:
                        value = correct_value_with_ppocr(
                            value,
                            cell["text_regions"],
                        )

                    field_key = selected_candidate["field"]["key"]
                    # Không để phần Phiên bản dính vào value của Tên rơ-le.
                    if field_key == "ten_ro_le" and value:
                        version_match = re.search(
                            r"phiên\s+bản",
                            value,
                            re.IGNORECASE,
                        )
                        if version_match:
                            value = self._clean_text(
                                value[:version_match.start()]
                            ) or None

                    fields[field_key] = {
                        "value": value,
                        "coordinate": list(cell["coordinate"]),
                        "score": float(cell["score"]),
                    }

        self._move_version_suffixes(fields)
        return fields

    # Tìm field name và loại các match ngắn bị chồng lấn.
    def _find_field_matches(self, text):
        """Tìm field name hợp lệ trong text của một cell."""
        candidates = []

        # Thử toàn bộ tên thay thế của từng field.
        for field in self.field_definitions:
            for name in field["names"]:
                name_pattern = r"(?<!\w)" + r"\s+".join(
                    rf"{re.escape(part)}\w*"
                    for part in name.split()
                )
                for match in re.finditer(
                    name_pattern,
                    text,
                    re.IGNORECASE,
                ):
                    is_first_field = not text[:match.start()].strip()
                    has_colon = re.match(r"\s*:", text[match.end():])
                    # Field nằm giữa cell chỉ hợp lệ khi có dấu hai chấm.
                    if not is_first_field and not has_colon:
                        continue
                    candidates.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "matched_text": match.group(0),
                            "name": name,
                            "field": field,
                        }
                    )

        longest_candidates = []
        # Giữ match dài nhất tại các vùng ký tự bị chồng lấn.
        for candidate in candidates:
            candidate_length = candidate["end"] - candidate["start"]
            has_longer_overlap = any(
                other["start"] < candidate["end"]
                and candidate["start"] < other["end"]
                and other["end"] - other["start"] > candidate_length
                for other in candidates
            )
            if not has_longer_overlap:
                longest_candidates.append(candidate)

        grouped_matches = {}
        # Gộp các field key dùng chung cùng một field name.
        for candidate in longest_candidates:
            match_key = (candidate["start"], candidate["end"])
            if match_key not in grouped_matches:
                grouped_matches[match_key] = {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "matched_text": candidate["matched_text"],
                    "candidates": [],
                }

            existing_keys = {
                item["field"]["key"]
                for item in grouped_matches[match_key]["candidates"]
            }
            if candidate["field"]["key"] not in existing_keys:
                grouped_matches[match_key]["candidates"].append(candidate)

        return sorted(
            grouped_matches.values(),
            key=lambda match: match["start"],
        )

    # Chọn field key duy nhất bằng context của cell rồi của hàng.
    def _resolve_field_candidate(
        self,
        match,
        cell_context_keys,
        row_context_keys,
    ):
        """Chọn field candidate phù hợp với context gần nhất."""
        if len(match["candidates"]) == 1:
            return match["candidates"][0]

        cell_candidates = [
            candidate
            for candidate in match["candidates"]
            if set(candidate["field"].get("context_keys", []))
            & cell_context_keys
        ]
        if len(cell_candidates) == 1:
            return cell_candidates[0]

        row_candidates = [
            candidate
            for candidate in match["candidates"]
            if set(candidate["field"].get("context_keys", []))
            & row_context_keys
        ]
        if len(row_candidates) == 1:
            return row_candidates[0]

        return None

    # Cắt value từ cuối field name đến field name kế tiếp.
    def _extract_field_value(self, text, matches, match_index):
        """Lấy value tương ứng với một field match."""
        match = matches[match_index]
        value_end = (
            matches[match_index + 1]["start"]
            if match_index + 1 < len(matches)
            else len(text)
        )
        value_text = text[match["end"]:value_end]
        if ":" in value_text:
            value_text = value_text.split(":", 1)[1]

        return self._clean_text(value_text) or None

    # Giữ hậu tố version hợp lệ và chuyển phần dư về field cha.
    def _move_version_suffixes(self, fields):
        """Chuẩn hóa suffix của version rơ-le và phần mềm."""
        version_parent_keys = {
            "phien_ban_ro_le": "ten_ro_le",
            "phien_ban_phan_mem": "phan_mem",
        }

        # Xử lý độc lập từng cặp field version và field cha.
        for version_key, parent_key in version_parent_keys.items():
            version_field = fields[version_key]
            version_value = version_field["value"]
            if not version_value:
                continue

            # A-D và X là hậu tố hợp lệ của field phiên bản.
            if re.search(r"\d[A-DX]$", version_value, re.IGNORECASE):
                continue

            integer_matches = list(re.finditer(r"\d+", version_value))
            version_coordinate = version_field["coordinate"]
            version_score = version_field["score"]

            # Giữ version đến số cuối và lấy phần còn lại làm suffix.
            if integer_matches:
                last_integer_end = integer_matches[-1].end()
                suffix = self._clean_text(
                    version_value[last_integer_end:]
                )
                if not suffix:
                    continue
                version_field["value"] = self._clean_text(
                    version_value[:last_integer_end]
                )
            else:
                suffix = version_value
                version_field["value"] = None
                version_field["coordinate"] = None
                version_field["score"] = None

            parent_field = fields[parent_key]
            parent_value = parent_field["value"] or ""
            parent_field["value"] = self._clean_text(
                f"{parent_value} {suffix}"
            )

            # Dùng metadata của version nếu field cha chưa có cell nguồn.
            if parent_field["coordinate"] is None:
                parent_field["coordinate"] = version_coordinate
                parent_field["score"] = version_score

    # Gom khoảng trắng thừa nhưng không thay đổi nội dung OCR.
    def _clean_text(self, text):
        """Chuẩn hóa khoảng trắng trong field name và value."""
        return " ".join(text.split())
