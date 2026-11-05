"""Image-based classifier for distinguishing page-2 comment sheets."""

import cv2


def _count_row_groups(row_indices, maximum_gap=4):
    """Count clusters of nearby rows."""
    group_count = 0
    previous_row = None

    for row_index in row_indices:
        if previous_row is None or row_index - previous_row > maximum_gap:
            group_count += 1
        previous_row = row_index

    return group_count


def is_comment_page(image_path):
    """Classify a comment page from dotted rows and horizontal rules."""
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise ValueError(f"Could not read page image: {image_path}")

    height, width = gray_image.shape
    normalized_height = round(height * 1000 / width)
    normalized_image = cv2.resize(
        gray_image,
        (1000, normalized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    binary_image = cv2.adaptiveThreshold(
        normalized_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    table_area = binary_image[
        int(normalized_height * 0.15):int(normalized_height * 0.90),
        50:950,
    ]

    dotted_runs = (
        (table_area[:, 1:] > 0) & (table_area[:, :-1] == 0)
    ).sum(axis=1)
    dotted_rows = (dotted_runs >= 25).nonzero()[0]
    dotted_group_count = _count_row_groups(dotted_rows)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (300, 1),
    )
    horizontal_lines = cv2.morphologyEx(
        table_area,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    solid_rows = ((horizontal_lines > 0).sum(axis=1) >= 300).nonzero()[0]
    solid_group_count = _count_row_groups(solid_rows)

    return 20 <= dotted_group_count <= 35 and solid_group_count <= 5
