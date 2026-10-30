# Mục đích: lọc link KEY → VALUE và phục hồi region bị tách khỏi pair.
# Chức năng: lọc link, nối fragment cùng role với pair gần nhất và hỗ trợ output chỉ gồm region đã ghép cặp.
# Luồng xử lý: nhận prediction → lọc từng nhóm link → phục hồi fragment → lọc nhãn không có cặp.
# Cấu hình/parameter: so sánh hình học bbox; cùng region được xem là inline;
# tolerance vùng ngang FIELD_TABLE mặc định là 4% chiều rộng page.
# Quan hệ: dùng chung bởi Page3LayoutModel, viewer và script sửa predictions.json.
# Input/output: nhận một prediction dict và cập nhật trực tiếp dict đó.
# Contract: giữ nguyên region ID, text, bbox, score và role không liên quan đến link.

if __package__:
    from .role_change_metadata import (
        initialize_role_change_codes,
        record_role_change,
    )
else:
    from scripts.inference.role_change_metadata import (
        initialize_role_change_codes,
        record_role_change,
    )

OVERLAP_TOLERANCE = 10
SAME_ROW_GAP = 30
FIELD_TABLE_X_TOLERANCE = 0.04  # 4% chiều rộng page cho mỗi phía.
CROSS_ROW_MAX_VERTICAL_GAP = 2.5
CROSS_ROW_MIN_HORIZONTAL_OVERLAP = 0.2
CROSS_ROW_MIN_SCORE_MARGIN = 0.1


def value_is_right(key_region, value_region):
    """Kiểm tra VALUE nằm bên phải KEY hoặc cùng bbox dạng inline."""
    if key_region["id"] == value_region["id"]:
        return True

    key_box = key_region["bbox"]
    value_box = value_region["bbox"]
    key_center_x = (key_box[0] + key_box[2]) / 2
    value_center_x = (value_box[0] + value_box[2]) / 2
    return value_center_x > key_center_x


def force_field_table_headers(regions):
    """Ép header Item và Value thành role FIELD_TABLE."""
    for region in regions:
        if region.get("text", "").strip().lower() in {"item", "value"}:
            region["labels"] = ["FIELD_TABLE"]
    return regions


def filter_prediction_links(prediction):
    """Lọc link sai và cập nhật VALUE không còn link hợp lệ thành OTHER."""
    role_change_codes = prediction.setdefault("role_change_codes", {})
    force_field_table_headers(prediction["regions"])
    filter_distant_same_row_candidates(
        prediction["regions"],
        role_change_codes=role_change_codes,
    )
    region_by_id = {
        region["id"]: region
        for region in prediction["regions"]
    }
    link_groups = {}
    if "links_by_mode" in prediction:
        link_groups = prediction["links_by_mode"]
    else:
        link_groups = {"selected": prediction.get("links", [])}

    filtered_groups = {}
    rejected_value_ids = set()
    valid_value_ids = set()
    for mode, links in link_groups.items():
        filtered_links = []
        for link in links:
            key_region = region_by_id[link["key_region_id"]]
            value_region = region_by_id[link["value_region_id"]]
            if (
                "KEY" not in key_region.get("labels", [])
                or "VALUE" not in value_region.get("labels", [])
            ):
                continue
            if value_is_right(key_region, value_region):
                filtered_links.append(link)
                valid_value_ids.add(value_region["id"])
            else:
                rejected_value_ids.add(value_region["id"])
        filtered_groups[mode] = filtered_links

    selected_mode = None
    if "links_by_mode" in prediction:
        prediction["links_by_mode"] = filtered_groups
        selected_mode = (
            "learned"
            if prediction.get("link_mode") == "both"
            else prediction.get("link_mode", "row")
        )
        prediction["links"] = filtered_groups.get(selected_mode, [])
    else:
        prediction["links"] = filtered_groups["selected"]

    repair_split_links(prediction)
    if selected_mode is not None:
        filtered_groups[selected_mode] = prediction["links"]
        prediction["links_by_mode"] = filtered_groups
    valid_value_ids = {
        link["value_region_id"]
        for link in prediction.get("links", [])
    }
    for region in prediction["regions"]:
        if region["id"] in rejected_value_ids - valid_value_ids:
            region["labels"] = ["OTHER"]
            record_role_change(role_change_codes, region["id"], 5)
    return prediction


def _same_row(first, second):
    """Kiểm tra hai bbox có cùng một hàng hay không."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    overlap = min(first_box[3], second_box[3]) - max(
        first_box[1],
        second_box[1],
    )
    if overlap > 0:
        return True
    first_center = (first_box[1] + first_box[3]) / 2
    second_center = (second_box[1] + second_box[3]) / 2
    return abs(first_center - second_center) <= max(
        first_box[3] - first_box[1],
        second_box[3] - second_box[1],
    ) * 0.5


def _overlaps_field_table_y_mean(region, field_table):
    """Kiểm tra bbox chứa tâm dọc của FIELD_TABLE."""
    region_box = region["bbox"]
    table_box = field_table["bbox"]
    table_y_mean = table_box[1] + (table_box[3] - table_box[1]) / 2
    return region_box[1] <= table_y_mean <= region_box[3]


def _horizontal_gap(first, second):
    """Tính khoảng cách ngang giữa hai bbox."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    left, right = sorted((first_box, second_box), key=lambda box: box[0])
    return max(0, right[0] - left[2])


def _below_field_table(region, field_tables):
    """Kiểm tra region nằm dưới và cùng cột với một FIELD_TABLE."""
    region_box = region["bbox"]
    for field_table in field_tables:
        table_box = field_table["bbox"]
        x_overlap = min(region_box[2], table_box[2]) - max(
            region_box[0],
            table_box[0],
        )
        if x_overlap > 0 and region_box[1] >= table_box[3]:
            return True
    return False


def field_table_x_ranges(regions, image_width):
    """Lấy các vùng ngang của FIELD_TABLE theo tỷ lệ chiều rộng page."""
    return [
        (
            region["bbox"][0] / image_width,
            region["bbox"][2] / image_width,
        )
        for region in regions
        if "FIELD_TABLE" in region.get("labels", [])
    ]


def _inherited_field_table_scope(
    region,
    field_table_x_ranges,
    image_width,
    tolerance,
):
    """Kiểm tra bbox có nằm trong vùng ngang FIELD_TABLE kế thừa không."""
    region_box = region["bbox"]
    region_left = region_box[0] / image_width
    region_right = region_box[2] / image_width
    for table_left, table_right in field_table_x_ranges:
        expanded_left = max(0, table_left - tolerance)
        expanded_right = min(1, table_right + tolerance)
        if min(region_right, expanded_right) > max(
            region_left,
            expanded_left,
        ):
            return True
    return False


def _candidate_is_in_field_table_scope(
    region,
    field_tables,
    inherited_field_table_x_ranges,
    image_width,
    field_table_x_tolerance,
):
    """Xác định candidate theo FIELD_TABLE hiện tại hoặc page trước."""
    if field_tables:
        first_field_table_top = min(
            field_table["bbox"][1]
            for field_table in field_tables
        )
        if region["bbox"][1] < first_field_table_top:
            return bool(inherited_field_table_x_ranges) and (
                _inherited_field_table_scope(
                    region,
                    inherited_field_table_x_ranges,
                    image_width,
                    field_table_x_tolerance,
                )
            )
        return _below_field_table(region, field_tables)

    return bool(inherited_field_table_x_ranges) and (
        _inherited_field_table_scope(
            region,
            inherited_field_table_x_ranges,
            image_width,
            field_table_x_tolerance,
        )
    )


def filter_distant_same_row_candidates(
    regions,
    inherited_field_table_x_ranges=None,
    image_width=None,
    field_table_x_tolerance=FIELD_TABLE_X_TOLERANCE,
    role_change_codes=None,
):
    """Loại KEY/VALUE cùng hàng FIELD_TABLE hoặc ở xa candidate thật."""
    field_tables = [
        region
        for region in regions
        if "FIELD_TABLE" in region.get("labels", [])
    ]
    if role_change_codes is None:
        role_change_codes = {}
    if not field_tables and not inherited_field_table_x_ranges:
        return regions

    if field_tables:
        same_row_field_table_ids = {
            region["id"]
            for region in regions
            if set(region.get("labels", [])).intersection({"KEY", "VALUE"})
            and any(
                _overlaps_field_table_y_mean(region, field_table)
                for field_table in field_tables
            )
        }
        for region in regions:
            if region["id"] in same_row_field_table_ids:
                region["labels"] = ["OTHER"]
                record_role_change(role_change_codes, region["id"], 2)

    candidates_by_role = {
        role: [
            region
            for region in regions
            if role in region.get("labels", [])
        ]
        for role in ("KEY", "VALUE")
    }
    false_positive_ids = set()
    for candidates in candidates_by_role.values():
        for index, first in enumerate(candidates):
            for second in candidates[index + 1:]:
                if not _same_row(first, second):
                    continue
                if _horizontal_gap(first, second) <= SAME_ROW_GAP:
                    continue
                first_is_true = _candidate_is_in_field_table_scope(
                    first,
                    field_tables,
                    inherited_field_table_x_ranges,
                    image_width,
                    field_table_x_tolerance,
                )
                second_is_true = _candidate_is_in_field_table_scope(
                    second,
                    field_tables,
                    inherited_field_table_x_ranges,
                    image_width,
                    field_table_x_tolerance,
                )
                if first_is_true and not second_is_true:
                    false_positive_ids.add(second["id"])
                    reason_code = 4 if (
                        not field_tables
                        or second["bbox"][1] < min(
                            field_table["bbox"][1]
                            for field_table in field_tables
                        )
                    ) else 3
                    record_role_change(
                        role_change_codes,
                        second["id"],
                        reason_code,
                    )
                elif second_is_true and not first_is_true:
                    false_positive_ids.add(first["id"])
                    reason_code = 4 if (
                        not field_tables
                        or first["bbox"][1] < min(
                            field_table["bbox"][1]
                            for field_table in field_tables
                        )
                    ) else 3
                    record_role_change(
                        role_change_codes,
                        first["id"],
                        reason_code,
                    )

    for region in regions:
        if region["id"] in false_positive_ids:
            region["labels"] = ["OTHER"]
    return regions


def _has_layout_barrier(first, second, regions):
    """Kiểm tra header hoặc field table nằm giữa hai fragment."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    corridor = [
        min(first_box[0], second_box[0]),
        min(first_box[1], second_box[1]),
        max(first_box[2], second_box[2]),
        max(first_box[3], second_box[3]),
    ]
    for region in regions:
        if region["id"] in {first["id"], second["id"]}:
            continue
        labels = set(region.get("labels", []))
        if not labels.intersection({"HEADER", "FIELD_TABLE"}):
            continue
        box = region["bbox"]
        if (
            box[2] > corridor[0]
            and box[0] < corridor[2]
            and box[3] > corridor[1]
            and box[1] < corridor[3]
        ):
            return True
    return False


def _can_join_fragments(first, second, regions):
    """Kiểm tra overlap theo một trục với sai số cho phép."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    x_overlap = (
        min(first_box[2], second_box[2])
        - max(first_box[0], second_box[0])
    )
    y_overlap = (
        min(first_box[3], second_box[3])
        - max(first_box[1], second_box[1])
    )
    x_gap = max(
        0,
        max(first_box[0], second_box[0])
        - min(first_box[2], second_box[2]),
    )
    y_gap = max(
        0,
        max(first_box[1], second_box[1])
        - min(first_box[3], second_box[3]),
    )
    return (
        (
            (x_overlap > 0 and y_gap <= OVERLAP_TOLERANCE)
            or (y_overlap > 0 and x_gap <= OVERLAP_TOLERANCE)
        )
        and not _has_layout_barrier(
            first,
            second,
            regions,
        )
    )


def _fragment_distance(first, second):
    """Tính khoảng cách tâm để chọn fragment gần nhất."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    first_center_x = (first_box[0] + first_box[2]) / 2
    first_center_y = (first_box[1] + first_box[3]) / 2
    second_center_x = (second_box[0] + second_box[2]) / 2
    second_center_y = (second_box[1] + second_box[3]) / 2
    return abs(first_center_x - second_center_x) + abs(
        first_center_y - second_center_y
    )


def _paired_targets(prediction):
    """Lập ánh xạ region đã pair tới các region đối diện."""
    targets = {}
    for link in prediction.get("links", []):
        key_id = link["key_region_id"]
        value_id = link["value_region_id"]
        targets.setdefault(key_id, set()).add(value_id)
        targets.setdefault(value_id, set()).add(key_id)
    return targets


def repair_split_links(prediction):
    """Nối fragment KEY/VALUE mồ côi vào pair gần nhất và giữ pair hiện có."""
    regions = prediction["regions"]
    region_by_id = {region["id"]: region for region in regions}
    paired_targets = _paired_targets(prediction)
    original_paired_targets = {
        region_id: set(target_ids)
        for region_id, target_ids in paired_targets.items()
    }
    links = prediction.setdefault("links", [])

    while True:
        candidates = []
        for orphan in regions:
            orphan_id = orphan["id"]
            orphan_labels = set(orphan.get("labels", []))
            if not orphan_labels.intersection({"KEY", "VALUE"}):
                continue
            if orphan_id in paired_targets:
                continue

            role = "KEY" if "KEY" in orphan_labels else "VALUE"
            for anchor_id, target_ids in original_paired_targets.items():
                anchor = region_by_id[anchor_id]
                anchor_labels = set(anchor.get("labels", []))
                if role not in anchor_labels:
                    continue
                if len(target_ids) != 1:
                    continue
                if not _can_join_fragments(anchor, orphan, regions):
                    continue
                target_id = next(iter(target_ids))
                key_id, value_id = (
                    (orphan_id, target_id)
                    if role == "KEY"
                    else (target_id, orphan_id)
                )
                key_region = region_by_id[key_id]
                value_region = region_by_id[value_id]
                if not value_is_right(key_region, value_region):
                    continue
                distance = _fragment_distance(anchor, orphan)
                candidates.append((distance, orphan_id, anchor_id, target_id))

        if not candidates:
            return prediction

        candidates_by_orphan = {}
        for candidate in candidates:
            candidates_by_orphan.setdefault(candidate[1], []).append(candidate)
        unambiguous_candidates = []
        for orphan_candidates in candidates_by_orphan.values():
            orphan_candidates.sort(key=lambda item: item[0])
            if len(orphan_candidates) == 1 or (
                orphan_candidates[0][0] < orphan_candidates[1][0]
            ):
                unambiguous_candidates.append(orphan_candidates[0])
        if not unambiguous_candidates:
            return prediction

        unambiguous_candidates.sort(key=lambda item: item[0])
        distance, orphan_id, anchor_id, target_id = unambiguous_candidates[0]

        anchor = region_by_id[anchor_id]
        anchor_labels = set(anchor.get("labels", []))
        if "KEY" in anchor_labels:
            key_id, value_id = orphan_id, target_id
        else:
            key_id, value_id = target_id, orphan_id
        links.append({
            "key_region_id": key_id,
            "value_region_id": value_id,
            "score": next(
                link.get("score", 0.0)
                for link in links
                if link["key_region_id"] == target_id
                or link["value_region_id"] == target_id
            ),
        })
        paired_targets.setdefault(orphan_id, set()).add(
            target_id,
        )


def _horizontal_overlap_ratio(first, second):
    """Tính overlap theo trục X trên bbox nhỏ hơn."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    overlap = max(
        0,
        min(first_box[2], second_box[2])
        - max(first_box[0], second_box[0]),
    )
    first_width = max(first_box[2] - first_box[0], 1)
    second_width = max(second_box[2] - second_box[0], 1)
    return overlap / min(first_width, second_width)


def _cross_row_candidate_score(
    key_region,
    value_region,
    regions,
    max_vertical_gap,
    min_horizontal_overlap,
):
    """Tính score cho cặp orphan KEY/VALUE khác row."""
    if not value_is_right(key_region, value_region):
        return None
    if _same_row(key_region, value_region):
        return None

    key_box = key_region["bbox"]
    value_box = value_region["bbox"]
    vertical_gap = max(
        0,
        max(key_box[1], value_box[1])
        - min(key_box[3], value_box[3]),
    )
    max_height = max(
        key_box[3] - key_box[1],
        value_box[3] - value_box[1],
        1,
    )
    if vertical_gap > max_height * max_vertical_gap:
        return None

    horizontal_overlap = _horizontal_overlap_ratio(
        key_region,
        value_region,
    )
    if horizontal_overlap < min_horizontal_overlap:
        return None
    if _has_layout_barrier(key_region, value_region, regions):
        return None

    vertical_score = max(
        0,
        1 - vertical_gap / (max_height * max_vertical_gap),
    )
    return horizontal_overlap * 0.7 + vertical_score * 0.3


def _best_unambiguous_candidate(candidates, min_score_margin):
    """Chọn candidate tốt nhất nếu không bị candidate khác cạnh tranh."""
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and (
        candidates[0][0] - candidates[1][0] < min_score_margin
    ):
        return None
    return candidates[0]


def recover_cross_row_links(
    prediction,
    max_vertical_gap=CROSS_ROW_MAX_VERTICAL_GAP,
    min_horizontal_overlap=CROSS_ROW_MIN_HORIZONTAL_OVERLAP,
    min_score_margin=CROSS_ROW_MIN_SCORE_MARGIN,
):
    """Ghép orphan KEY/VALUE khác row khi hình học đủ rõ ràng."""
    regions = prediction["regions"]
    links = prediction.setdefault("links", [])
    linked_key_ids = {
        link["key_region_id"]
        for link in links
    }
    linked_value_ids = {
        link["value_region_id"]
        for link in links
    }
    orphan_keys = [
        region
        for region in regions
        if "KEY" in region.get("labels", [])
        and region["id"] not in linked_key_ids
    ]
    orphan_values = [
        region
        for region in regions
        if "VALUE" in region.get("labels", [])
        and region["id"] not in linked_value_ids
    ]
    if not orphan_keys or not orphan_values:
        return prediction

    key_candidates = {region["id"]: [] for region in orphan_keys}
    value_candidates = {region["id"]: [] for region in orphan_values}
    for key_region in orphan_keys:
        for value_region in orphan_values:
            score = _cross_row_candidate_score(
                key_region,
                value_region,
                regions,
                max_vertical_gap,
                min_horizontal_overlap,
            )
            if score is None:
                continue
            candidate = (
                score,
                key_region["id"],
                value_region["id"],
            )
            key_candidates[key_region["id"]].append(candidate)
            value_candidates[value_region["id"]].append(candidate)

    best_by_key = {
        key_id: _best_unambiguous_candidate(
            candidates,
            min_score_margin,
        )
        for key_id, candidates in key_candidates.items()
        if candidates
    }
    best_by_value = {
        value_id: _best_unambiguous_candidate(
            candidates,
            min_score_margin,
        )
        for value_id, candidates in value_candidates.items()
        if candidates
    }
    for key_id, candidate in best_by_key.items():
        if candidate is None:
            continue
        score, _, value_id = candidate
        if best_by_value.get(value_id) != candidate:
            continue
        links.append({
            "key_region_id": key_id,
            "value_region_id": value_id,
            "score": score,
            "link_mode": "cross_row_fallback",
        })
    return prediction


def filter_unlinked_regions(
    prediction,
    enable_cross_row_fallback=False,
    max_vertical_gap=CROSS_ROW_MAX_VERTICAL_GAP,
    min_horizontal_overlap=CROSS_ROW_MIN_HORIZONTAL_OVERLAP,
    min_score_margin=CROSS_ROW_MIN_SCORE_MARGIN,
):
    """Đổi KEY/VALUE không có link thành OTHER và giữ nguyên role khác."""
    role_change_codes = prediction.setdefault("role_change_codes", {})
    repair_split_links(prediction)
    if enable_cross_row_fallback:
        recover_cross_row_links(
            prediction,
            max_vertical_gap=max_vertical_gap,
            min_horizontal_overlap=min_horizontal_overlap,
            min_score_margin=min_score_margin,
        )
    linked_key_ids = {
        link["key_region_id"]
        for link in prediction.get("links", [])
    }
    linked_value_ids = {
        link["value_region_id"]
        for link in prediction.get("links", [])
    }
    for region in prediction["regions"]:
        original_labels = list(region.get("labels", []))
        labels = set(region.get("labels", []))
        if "KEY" in labels and region["id"] not in linked_key_ids:
            labels.discard("KEY")
            record_role_change(role_change_codes, region["id"], 6)
        if "VALUE" in labels and region["id"] not in linked_value_ids:
            labels.discard("VALUE")
            record_role_change(role_change_codes, region["id"], 7)
        region["labels"] = [
            role for role in original_labels
            if role in labels
        ] or ["OTHER"]
    return prediction


def filter_prediction_pages(
    pages,
    field_table_x_tolerance=FIELD_TABLE_X_TOLERANCE,
    enable_cross_row_fallback=False,
    max_vertical_gap=CROSS_ROW_MAX_VERTICAL_GAP,
    min_horizontal_overlap=CROSS_ROW_MIN_HORIZONTAL_OVERLAP,
    min_score_margin=CROSS_ROW_MIN_SCORE_MARGIN,
):
    """Lọc tuần tự các page và kế thừa vùng ngang FIELD_TABLE."""
    initialize_role_change_codes(pages)
    ordered_pages = sorted(
        pages,
        key=lambda page: (
            page.get("pdf_name", ""),
            page.get("physical_page", 0),
        ),
    )
    inherited_field_table_x_ranges = None
    previous_pdf_name = None
    previous_physical_page = None
    for page in ordered_pages:
        force_field_table_headers(page["regions"])
        pdf_name = page.get("pdf_name")
        physical_page = page.get("physical_page")
        if pdf_name != previous_pdf_name:
            inherited_field_table_x_ranges = None
        elif (
            previous_physical_page is not None
            and physical_page is not None
            and physical_page != previous_physical_page + 1
        ):
            inherited_field_table_x_ranges = None

        filter_distant_same_row_candidates(
            page["regions"],
            inherited_field_table_x_ranges,
            page["image_width"],
            field_table_x_tolerance,
            page["role_change_codes"],
        )
        filter_prediction_links(page)
        filter_unlinked_regions(
            page,
            enable_cross_row_fallback=enable_cross_row_fallback,
            max_vertical_gap=max_vertical_gap,
            min_horizontal_overlap=min_horizontal_overlap,
            min_score_margin=min_score_margin,
        )
        current_field_table_x_ranges = field_table_x_ranges(
            page["regions"],
            page["image_width"],
        )
        if current_field_table_x_ranges:
            inherited_field_table_x_ranges = current_field_table_x_ranges
        previous_pdf_name = pdf_name
        previous_physical_page = physical_page
    return pages
