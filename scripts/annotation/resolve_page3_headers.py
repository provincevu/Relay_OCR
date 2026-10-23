# Mục đích: gán header hiện tại hoặc header kế thừa cho quan hệ KEY → VALUE.
# Chức năng: đọc annotations.json, sao lưu bản gốc và ghi thêm metadata header.
# Luồng xử lý: annotation gốc → tìm header phía trên key → kế thừa page trước → ghi đè dataset.
# Cấu hình/parameter: đường dẫn annotations.json và tùy chọn đường dẫn bản sao lưu.
# Quan hệ: chỉ bổ sung metadata cho relation, không thay đổi role hoặc link KEY → VALUE.
# Input/output: đọc annotations.json; tạo annotations_before_header.json và cập nhật annotations.json.
# Contract: header_region_ids là danh sách ID hoặc null; header_source và header_page mô tả nguồn header.

import argparse
import json
import shutil
from pathlib import Path


def _header_regions(page):
    """Lấy các region có role HEADER theo thứ tự tọa độ trên page."""
    regions = [
        region
        for region in page["regions"]
        if "HEADER" in page["annotations"][region["id"]]
    ]
    return sorted(
        regions,
        key=lambda region: (
            region["box"][1],
            region["box"][0],
            region["id"],
        ),
    )


def _relation_header_ids(page, relation, headers):
    """Tìm header gần nhất nằm phía trên các region KEY của relation."""
    regions_by_id = {
        region["id"]: region
        for region in page["regions"]
    }
    key_regions = [
        regions_by_id[region_id]
        for region_id in relation["key_region_ids"]
    ]
    key_top = min(region["box"][1] for region in key_regions)
    candidates = [
        region
        for region in headers
        if region["box"][3] <= key_top
    ]
    if not candidates:
        return None
    nearest = max(
        candidates,
        key=lambda region: (
            region["box"][3],
            -region["box"][1],
            region["id"],
        ),
    )
    return [nearest["id"]]


def _set_relation_header(relation, header_ids, source, page_name):
    """Ghi metadata header vào relation mà không thay đổi link hiện có."""
    relation["header_region_ids"] = header_ids
    relation["header_source"] = source
    relation["header_page"] = page_name


def _ordered_pages(dataset):
    """Nhóm page theo PDF và sắp xếp theo số page vật lý."""
    pages_by_pdf = {}
    for image_name, page in dataset["pages"].items():
        pages_by_pdf.setdefault(page["pdf_name"], []).append(
            (image_name, page),
        )
    for pages in pages_by_pdf.values():
        pages.sort(
            key=lambda item: (
                item[1]["physical_page"],
                item[0],
            ),
        )
    return pages_by_pdf


def resolve_headers(dataset):
    """Gán header cục bộ và header kế thừa cho toàn bộ relation."""
    relation_count = 0
    inherited_count = 0
    for pages in _ordered_pages(dataset).values():
        previous_header = None
        previous_physical_page = None
        for image_name, page in pages:
            headers = _header_regions(page)
            header_ids = [region["id"] for region in headers]
            physical_page = page["physical_page"]
            is_contiguous_page = (
                previous_physical_page is not None
                and physical_page == previous_physical_page + 1
            )
            for relation in page["relations"]:
                local_header_ids = _relation_header_ids(
                    page,
                    relation,
                    headers,
                )
                if local_header_ids is not None:
                    _set_relation_header(
                        relation,
                        local_header_ids,
                        "current_page",
                        image_name,
                    )
                elif (
                    previous_header is not None
                    and is_contiguous_page
                ):
                    _set_relation_header(
                        relation,
                        list(previous_header["region_ids"]),
                        "previous_page",
                        previous_header["page"],
                    )
                    inherited_count += 1
                else:
                    _set_relation_header(relation, None, None, None)
                relation_count += 1

            if header_ids:
                previous_header = {
                    "region_ids": header_ids,
                    "page": image_name,
                }
            elif not is_contiguous_page:
                previous_header = None
            previous_physical_page = physical_page
    return relation_count, inherited_count


def _backup_path(annotations_path):
    """Tạo đường dẫn mặc định cho bản sao annotation gốc."""
    return annotations_path.with_name(
        f"{annotations_path.stem}_before_header{annotations_path.suffix}"
    )


def _parse_arguments():
    """Đọc đường dẫn dataset và bản sao lưu tùy chọn."""
    parser = argparse.ArgumentParser(
        description="Gán header hiện tại và header kế thừa cho annotations.json."
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--backup", type=Path)
    return parser.parse_args()


def main():
    """Sao lưu rồi cập nhật annotations.json tại chỗ."""
    arguments = _parse_arguments()
    annotations_path = arguments.annotations.resolve()
    backup_path = (
        arguments.backup.resolve()
        if arguments.backup is not None
        else _backup_path(annotations_path)
    )
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation: {annotations_path}")
    if backup_path.exists():
        raise FileExistsError(
            f"Bản sao đã tồn tại: {backup_path}. "
            "Không ghi đè để bảo toàn annotation gốc."
        )
    dataset = json.loads(annotations_path.read_text(encoding="utf-8"))
    relation_count, inherited_count = resolve_headers(dataset)
    shutil.copy2(annotations_path, backup_path)
    temporary_path = annotations_path.with_suffix(
        f"{annotations_path.suffix}.tmp"
    )
    temporary_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(annotations_path)
    print(f"Đã sao lưu annotation gốc: {backup_path}")
    print(f"Đã cập nhật annotation: {annotations_path}")
    print(f"Đã xử lý {relation_count} relation")
    print(f"Relation kế thừa header từ page trước: {inherited_count}")


if __name__ == "__main__":
    main()
