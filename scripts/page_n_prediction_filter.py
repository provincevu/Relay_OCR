# Mục đích: lọc link KEY → VALUE theo quy tắc value nằm bên phải key.
# Chức năng: loại link sai và đổi value không còn link hợp lệ thành OTHER.
# Luồng xử lý: nhận prediction → lọc từng nhóm link → cập nhật nhãn region.
# Cấu hình/parameter: so sánh tâm bbox theo trục X, không dùng ngưỡng khoảng cách.
# Quan hệ: dùng chung bởi Page3LayoutModel, viewer và script sửa predictions.json.
# Input/output: nhận một prediction dict và cập nhật trực tiếp dict đó.
# Contract: giữ nguyên region ID, text, bbox, score và metadata; chỉ đổi links và labels.


def value_is_right(key_region, value_region):
    """Kiểm tra tâm bbox của value nằm bên phải tâm bbox của key."""
    key_box = key_region["bbox"]
    value_box = value_region["bbox"]
    key_center_x = (key_box[0] + key_box[2]) / 2
    value_center_x = (value_box[0] + value_box[2]) / 2
    return value_center_x > key_center_x


def filter_prediction_links(prediction):
    """Lọc link sai và cập nhật value không còn link hợp lệ thành OTHER."""
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
