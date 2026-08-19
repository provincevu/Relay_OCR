# Mục đích: trích xuất field TABLE 01 từ text đã gán vào cell.
# Chức năng: khớp tên field, lấy value và ghi nhận field cần xem lại.
# Luồng xử lý: cell theo hàng → field match → context → data chuẩn hóa.
# Cấu hình/parameter: nhận danh sách field từ table_01_fields.json.
# Quan hệ: được Table1Analysis gọi sau bước nhận diện cell và OCR.
# Input/output: nhận danh sách cell; trả data, field_details và review.
# Contract: data luôn chứa đầy đủ key được khai báo trong field config.

import re

from .table_cell_utils import group_boxes_into_rows


class TableFieldExtractor:
    """Trích xuất field TABLE 01 từ danh sách cell."""

    def __init__(self, field_definitions):
        self.field_definitions = field_definitions

    def empty_result(self):
        """Tạo kết quả field rỗng theo đúng contract."""
        return {
            "data": {
                field["key"]: None for field in self.field_definitions
            },
            "field_details": {},
            "review": [],
        }

    def extract(self, cells):
        """Trích xuất field từ text của các cell."""
        result = self.empty_result()
        data = result["data"]
        field_details = result["field_details"]
        review = result["review"]

        for row in group_boxes_into_rows(cells):
            parsed_cells = [
                (cell, self._find_field_matches(cell["text"]))
                for cell in row
            ]
            row_context_keys = self._collect_context_keys(parsed_cells)

            for cell, matches in parsed_cells:
                cell_context_keys = self._collect_context_keys(
                    [(cell, matches)]
                )
                for match_index, match in enumerate(matches):
                    selected_candidate = self._resolve_field_candidate(
                        match,
                        cell_context_keys,
                        row_context_keys,
                    )
                    value = self._extract_field_value(
                        cell["text"],
                        matches,
                        match_index,
                    )
                    raw_text = self._extract_field_raw_text(
                        cell["text"],
                        matches,
                        match_index,
                    )
                    if selected_candidate is None:
                        review.append(
                            {
                                "cell_index": cell["index"],
                                "detected_field_name": match["matched_text"],
                                "value": value,
                                "candidate_keys": [
                                    candidate["field"]["key"]
                                    for candidate in match["candidates"]
                                ],
                                "cell_score": cell["score"],
                                "raw_text": raw_text,
                                "reason": "context_khong_xac_dinh",
                            }
                        )
                        continue

                    field_key = selected_candidate["field"]["key"]
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

                    data[field_key] = value
                    field_details[field_key] = {
                        "matched_name": selected_candidate["name"],
                        "detected_field_name": match["matched_text"],
                        "cell_index": cell["index"],
                        "cell_score": cell["score"],
                        "raw_text": raw_text,
                    }

                if not matches and ":" in cell["text"]:
                    review.append(self._build_unmatched_review(cell))

        self._move_version_suffixes(data, field_details)
        return result

    def _find_field_matches(self, text):
        """Tìm field name và giữ match dài nhất khi bị chồng lấn."""
        candidates = []
        for field in self.field_definitions:
            for name in field["names"]:
                name_pattern = r"\s+".join(
                    re.escape(part) for part in name.split()
                )
                for match in re.finditer(
                    name_pattern,
                    text,
                    re.IGNORECASE,
                ):
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

    def _collect_context_keys(self, parsed_cells):
        """Lấy các key được xác định duy nhất trong vùng context."""
        context_keys = set()
        for _, matches in parsed_cells:
            for match in matches:
                candidate_keys = {
                    candidate["field"]["key"]
                    for candidate in match["candidates"]
                }
                if len(candidate_keys) == 1:
                    context_keys.update(candidate_keys)
        return context_keys

    def _resolve_field_candidate(
        self,
        match,
        cell_context_keys,
        row_context_keys,
    ):
        """Chọn field duy nhất theo context của cell rồi của hàng."""
        if len(match["candidates"]) == 1:
            return match["candidates"][0]

        cell_candidates = self._filter_candidates(
            match["candidates"],
            cell_context_keys,
        )
        if len(cell_candidates) == 1:
            return cell_candidates[0]

        row_candidates = self._filter_candidates(
            match["candidates"],
            row_context_keys,
        )
        if len(row_candidates) == 1:
            return row_candidates[0]
        return None

    def _filter_candidates(self, candidates, context_keys):
        """Lọc candidate có context key phù hợp."""
        return [
            candidate
            for candidate in candidates
            if set(candidate["field"].get("context_keys", []))
            & context_keys
        ]

    def _extract_field_value(self, text, matches, match_index):
        """Lấy value từ cuối field name đến field name kế tiếp."""
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

    def _extract_field_raw_text(self, text, matches, match_index):
        """Lấy raw text của field hiện tại."""
        raw_text_end = (
            matches[match_index + 1]["start"]
            if match_index + 1 < len(matches)
            else len(text)
        )
        raw_text = text[matches[match_index]["start"]:raw_text_end]
        return self._clean_text(raw_text)

    def _build_unmatched_review(self, cell):
        """Tạo dữ liệu review cho cell không khớp field config."""
        detected_name, value = cell["text"].split(":", 1)
        return {
            "cell_index": cell["index"],
            "detected_field_name": self._clean_text(detected_name),
            "value": self._clean_text(value) or None,
            "cell_score": cell["score"],
            "raw_text": cell["text"],
            "reason": "khong_khop_names",
        }

    def _move_version_suffixes(self, data, field_details):
        """Chuyển text sau version về field cha tương ứng."""
        version_parent_keys = {
            "phien_ban_ro_le": "ten_ro_le",
            "phien_ban_phan_mem": "phan_mem",
        }
        for version_key, parent_key in version_parent_keys.items():
            version_value = data[version_key]
            if not version_value:
                continue

            integer_matches = list(re.finditer(r"\d+", version_value))
            if not integer_matches:
                continue

            last_integer_end = integer_matches[-1].end()
            suffix = self._clean_text(version_value[last_integer_end:])
            if not suffix:
                continue

            data[version_key] = self._clean_text(
                version_value[:last_integer_end]
            )
            parent_value = data[parent_key] or ""
            data[parent_key] = self._clean_text(
                f"{parent_value} {suffix}"
            )
            self._update_version_details(
                version_key,
                parent_key,
                suffix,
                field_details,
            )

    def _update_version_details(
        self,
        version_key,
        parent_key,
        suffix,
        field_details,
    ):
        """Cập nhật raw text sau khi chuyển suffix của version."""
        version_detail = field_details.get(version_key)
        if version_detail:
            integer_matches = list(
                re.finditer(r"\d+", version_detail["raw_text"])
            )
            if integer_matches:
                version_detail["raw_text"] = self._clean_text(
                    version_detail["raw_text"][:integer_matches[-1].end()]
                )

        parent_detail = field_details.get(parent_key)
        if parent_detail:
            parent_detail["raw_text"] = self._clean_text(
                f"{parent_detail['raw_text']} {suffix}"
            )

    def _clean_text(self, text):
        """Gom khoảng trắng thừa nhưng giữ nguyên nội dung OCR."""
        return " ".join(text.split())
