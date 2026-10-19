# Mục đích: lọc link KEY → VALUE theo quy tắc value nằm bên phải key hoặc inline.
# Chức năng: lọc link, giữ inline KEY/VALUE và hỗ trợ output chỉ gồm region đã ghép cặp.
# Luồng xử lý: nhận prediction → lọc từng nhóm link → tùy chọn lọc nhãn không có cặp.
# Cấu hình/parameter: so sánh tâm bbox theo trục X; cùng region được xem là inline.
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

    for region in prediction["regions"]:
        if region["id"] in rejected_value_ids - valid_value_ids:
            region["labels"] = ["OTHER"]
    return prediction


def filter_unlinked_regions(prediction):
    """Đổi KEY/VALUE không có link thành OTHER và giữ nguyên role khác."""
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
