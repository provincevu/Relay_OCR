# Mục đích: lọc link KEY → VALUE và phục hồi region bị tách khỏi pair.
# Chức năng: lọc link, nối fragment cùng role với pair gần nhất và hỗ trợ output chỉ gồm region đã ghép cặp.
# Luồng xử lý: nhận prediction → lọc từng nhóm link → phục hồi fragment → lọc nhãn không có cặp.
# Cấu hình/parameter: so sánh hình học bbox; cùng region được xem là inline.
# Quan hệ: dùng chung bởi Page3LayoutModel, viewer và script sửa predictions.json.
# Input/output: nhận một prediction dict và cập nhật trực tiếp dict đó.
# Contract: giữ nguyên region ID, text, bbox, score và role không liên quan đến link.


def value_is_right(key_region, value_region):
    """Kiểm tra VALUE nằm bên phải KEY hoặc cùng bbox dạng inline."""
    if key_region["id"] == value_region["id"]:
        return True

    key_box = key_region["bbox"]
    value_box = value_region["bbox"]
    key_center_x = (key_box[0] + key_box[2]) / 2
    value_center_x = (value_box[0] + value_box[2]) / 2
    return value_center_x > key_center_x


def filter_prediction_links(prediction):
    """Lọc link sai và cập nhật VALUE không còn link hợp lệ thành OTHER."""
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
    return prediction


def _box_width(region):
    """Trả chiều rộng bbox của region."""
    box = region["bbox"]
    return max(box[2] - box[0], 1)


def _box_height(region):
    """Trả chiều cao bbox của region."""
    box = region["bbox"]
    return max(box[3] - box[1], 1)


def _overlap_ratio(first_start, first_end, second_start, second_end):
    """Tính tỷ lệ phần giao trên đoạn ngắn hơn."""
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start))
    first_size = max(first_end - first_start, 1)
    second_size = max(second_end - second_start, 1)
    return overlap / min(first_size, second_size)


def _fragment_gap(first, second, horizontal):
    """Tính khoảng cách giữa hai fragment theo một trục."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    if horizontal:
        return max(
            0,
            max(first_box[0], second_box[0])
            - min(first_box[2], second_box[2]),
        )
    return max(
        0,
        max(first_box[1], second_box[1])
        - min(first_box[3], second_box[3]),
    )


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
    """Kiểm tra hai bbox cùng role có thể là hai mảnh liền nhau."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    y_overlap = _overlap_ratio(
        first_box[1],
        first_box[3],
        second_box[1],
        second_box[3],
    )
    x_overlap = _overlap_ratio(
        first_box[0],
        first_box[2],
        second_box[0],
        second_box[2],
    )
    horizontal = (
        y_overlap >= 0.5
        and _fragment_gap(first, second, True)
        <= max(_box_height(first), _box_height(second)) * 1.5
    )
    vertical = (
        (
            x_overlap >= 0.35
            or abs(first_box[0] - second_box[0])
            <= max(_box_width(first), _box_width(second)) * 0.35
        )
        and _fragment_gap(first, second, False)
        <= max(_box_height(first), _box_height(second)) * 1.5
    )
    return (horizontal or vertical) and not _has_layout_barrier(
        first,
        second,
        regions,
    )


def _fragment_distance(first, second):
    """Tính khoảng cách theo hướng nối của hai fragment."""
    first_box = first["bbox"]
    second_box = second["bbox"]
    y_overlap = _overlap_ratio(
        first_box[1],
        first_box[3],
        second_box[1],
        second_box[3],
    )
    if y_overlap >= 0.5:
        return _fragment_gap(first, second, True)
    return _fragment_gap(first, second, False)


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
            for anchor_id, target_ids in paired_targets.items():
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


def filter_unlinked_regions(prediction):
    """Đổi KEY/VALUE không có link thành OTHER và giữ nguyên role khác."""
    repair_split_links(prediction)
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
        if "VALUE" in labels and region["id"] not in linked_value_ids:
            labels.discard("VALUE")
        region["labels"] = [
            role for role in original_labels
            if role in labels
        ] or ["OTHER"]
    return prediction
