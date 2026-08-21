# Mục đích: tách logic trích xuất field cho script debug bảng page 1.
# Chức năng: khớp names, lấy value, xử lý context và hậu tố phiên bản.
# Luồng xử lý: cell đã OCR → field match → context → dữ liệu và review.
# Cấu hình/parameter: dùng field_definitions, field_review và extracted_tables của analyzer.
# Quan hệ: được Page1TableAnalyzer kế thừa để giữ nguyên contract hiện có.
# Input/output: nhận danh sách cell; cập nhật dữ liệu trích xuất và review.
# Contract: giữ nguyên các phương thức nội bộ mà luồng test đang gọi.

import re


class DebugTableFieldsMixin:
    """Cung cấp logic trích xuất field cho analyzer debug."""

    def _extract_table_fields(self, source_image, table_score, cells):
        """Trích xuất 16 field của TABLE 01 từ text trong các cell hợp lệ."""
        extracted_data = {
            field["key"]: None
            for field in self.field_definitions
        }
        field_details = {}
        rows = self._group_boxes_into_rows(cells)

        for row in rows:
            parsed_cells = []
            row_context_keys = set()

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
                    value = self._extract_field_value(
                        cell["text"],
                        matches,
                        match_index,
                    )
                    raw_field_text = self._extract_field_raw_text(
                        cell["text"],
                        matches,
                        match_index,
                    )

                    if selected_candidate is None:
                        self.field_review.append(
                            {
                                "source_image": source_image,
                                "cell_index": cell["index"],
                                "detected_field_name": match["matched_text"],
                                "value": value,
                                "candidate_keys": [
                                    candidate["field"]["key"]
                                    for candidate in match["candidates"]
                                ],
                                "muc_do_tin_cay": 1,
                                "cell_score": cell["score"],
                                "raw_text": raw_field_text,
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

                    extracted_data[field_key] = value
                    field_details[field_key] = {
                        "matched_name": selected_candidate["name"],
                        "detected_field_name": match["matched_text"],
                        "muc_do_tin_cay": 1,
                        "cell_index": cell["index"],
                        "cell_score": cell["score"],
                        "raw_text": raw_field_text,
                    }

                if not matches and ":" in cell["text"]:
                    detected_name, value = cell["text"].split(":", 1)
                    detected_name = self._clean_text(detected_name)
                    value = self._clean_text(value) or None
                    if detected_name:
                        self.field_review.append(
                            {
                                "source_image": source_image,
                                "cell_index": cell["index"],
                                "detected_field_name": detected_name,
                                "value": value,
                                "muc_do_tin_cay": 2,
                                "cell_score": cell["score"],
                                "raw_text": cell["text"],
                                "reason": "khong_khop_names",
                            }
                        )

        self._move_version_suffixes(extracted_data, field_details)
        self.extracted_tables.append(
            {
                "source_image": source_image,
                "table_score": table_score,
                "data": extracted_data,
                "field_details": field_details,
            }
        )

    def _find_field_matches(self, text):
        """Tìm field name và giữ match dài nhất tại vùng bị chồng lấn."""
        candidates = []

        for field in self.field_definitions:
            for name in field["names"]:
                name_pattern = r"(?<!\w)" + r"\s+".join(
                    rf"{re.escape(part)}\w*"
                    for part in name.split()
                )
                for match in re.finditer(name_pattern, text, re.IGNORECASE):
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

    def _resolve_field_candidate(
        self,
        match,
        cell_context_keys,
        row_context_keys,
    ):
        """Chọn key duy nhất, ưu tiên context trong cell rồi trong hàng."""
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

    def _move_version_suffixes(self, extracted_data, field_details):
        """Giữ version đến số cuối hoặc hậu tố A-D hợp lệ."""
        version_parent_keys = {
            "phien_ban_ro_le": "ten_ro_le",
            "phien_ban_phan_mem": "phan_mem",
        }

        for version_key, parent_key in version_parent_keys.items():
            version_value = extracted_data[version_key]
            if not version_value:
                continue

            if re.search(r"\d[A-D]$", version_value):
                continue

            integer_matches = list(re.finditer(r"\d+", version_value))
            if integer_matches:
                last_integer_end = integer_matches[-1].end()
                suffix = self._clean_text(version_value[last_integer_end:])
                if not suffix:
                    continue
                extracted_data[version_key] = self._clean_text(
                    version_value[:last_integer_end]
                )
            else:
                suffix = version_value
                extracted_data[version_key] = None

            parent_value = extracted_data[parent_key] or ""
            extracted_data[parent_key] = self._clean_text(
                f"{parent_value} {suffix}"
            )

            version_detail = field_details.get(version_key)
            if version_detail and integer_matches:
                raw_text = version_detail["raw_text"]
                raw_integer_matches = list(re.finditer(r"\d+", raw_text))
                if raw_integer_matches:
                    raw_integer_end = raw_integer_matches[-1].end()
                    version_detail["raw_text"] = self._clean_text(
                        raw_text[:raw_integer_end]
                    )
            elif version_detail:
                field_details.pop(version_key)

            parent_detail = field_details.get(parent_key)
            if parent_detail:
                parent_detail["raw_text"] = self._clean_text(
                    f"{parent_detail['raw_text']} {suffix}"
                )

    def _extract_field_raw_text(self, text, matches, match_index):
        """Lấy raw text từ field name hiện tại đến field name kế tiếp."""
        raw_text_end = (
            matches[match_index + 1]["start"]
            if match_index + 1 < len(matches)
            else len(text)
        )
        raw_text = text[matches[match_index]["start"]:raw_text_end]
        return self._clean_text(raw_text)

    def _clean_text(self, text):
        """Gom khoảng trắng thừa nhưng giữ nguyên nội dung OCR."""
        return " ".join(text.split())
