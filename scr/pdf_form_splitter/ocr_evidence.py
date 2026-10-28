"""Use production detection and recognition to build splitter evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.detection import DocumentTextDetectionService
from src.image_io import read_image
from src.layout_analysis.pagination import TICKET_PATTERN, detect_page_reference
from src.recognition import VietnameseRecognitionService

from .evidence import PageEvidence, build_page_evidence


def _top_detections(detections: list[Any], image_height: int, scan_ratio: float) -> list[Any]:
    selected = []
    limit = image_height * scan_ratio
    for detection in detections:
        polygon = detection.polygon if hasattr(detection, "polygon") else detection["polygon"]
        centre_y = sum(float(point[1]) for point in polygon) / len(polygon)
        if centre_y <= limit:
            selected.append(detection)
    return selected


def _blocks(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for region in regions:
        polygon = region["polygon"]
        xs, ys = zip(*[(float(point[0]), float(point[1])) for point in polygon])
        blocks.append({
            "block_id": f"ocr_{region['index']}",
            "text": region["text"],
            "bbox_pixel": [min(xs), min(ys), max(xs), max(ys)],
            "polygon": polygon,
            "detection_score": region.get("detection_score"),
            "recognition_score": region.get("recognition_score"),
        })
    return blocks


def _recover_complete_pagination(
    image_path: Path,
    image_width: int,
    image_height: int,
    blocks: list[dict[str, Any]],
    recognizer: VietnameseRecognitionService,
) -> list[dict[str, Any]]:
    """Re-OCR the whole pagination row when detection kept only current page."""

    pagination = detect_page_reference(blocks, page_width=image_width, page_height=image_height)
    if pagination is not None and pagination["total_pages"] is not None:
        return blocks
    if pagination is not None:
        boxes = [block["bbox_pixel"] for block in pagination["source_blocks"]]
        x1 = max(0.0, min(box[0] for box in boxes) - image_width * 0.01)
        y1 = max(0.0, min(box[1] for box in boxes) - image_height * 0.004)
        x2 = min(float(image_width - 1), max(box[2] for box in boxes) + image_width * 0.12)
        y2 = min(float(image_height - 1), max(box[3] for box in boxes) + image_height * 0.004)
    else:
        ticket_blocks = [
            block for block in blocks
            if TICKET_PATTERN.search(str(block.get("text", "")))
            and float(block["bbox_pixel"][1]) <= image_height * 0.20
        ]
        if not ticket_blocks:
            return blocks
        ticket = min(ticket_blocks, key=lambda block: float(block["bbox_pixel"][1]))
        tx1, _, tx2, ty2 = [float(value) for value in ticket["bbox_pixel"]]
        x1 = max(0.0, tx1 - image_width * 0.16)
        y1 = max(0.0, ty2 - image_height * 0.003)
        x2 = min(float(image_width - 1), tx2 + image_width * 0.03)
        y2 = min(float(image_height - 1), ty2 + image_height * 0.04)
    detection = {"polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "score": None}
    region = recognizer.recognise_page(image_path, [detection]).as_dict()["regions"][0]
    recovered = {
        "block_id": "pagination_row_reocr",
        "text": region["text"],
        "bbox_pixel": [x1, y1, x2, y2],
        "polygon": region["polygon"],
        "detection_score": None,
        "recognition_score": region["recognition_score"],
    }
    candidate_blocks = [*blocks, recovered]
    candidate = detect_page_reference(candidate_blocks, page_width=image_width, page_height=image_height)
    same_current = pagination is None or (candidate and candidate["page_number"] == pagination["page_number"])
    if candidate and candidate["total_pages"] is not None and same_current:
        return candidate_blocks
    return blocks


def analyse_rendered_pages(
    pages: list[Path],
    *,
    use_gpu: bool = False,
    scan_ratio: float = 0.45,
    detector: DocumentTextDetectionService | None = None,
    recognizer: VietnameseRecognitionService | None = None,
) -> tuple[list[PageEvidence], list[dict[str, Any]]]:
    """Load OCR models once and analyse the upper cover-signature region."""

    import cv2

    detector = detector or DocumentTextDetectionService(use_gpu=use_gpu)
    recognizer = recognizer or VietnameseRecognitionService(use_gpu=use_gpu)
    evidence: list[PageEvidence] = []
    cache_pages: list[dict[str, Any]] = []
    for page_index, image_path in enumerate(pages, start=1):
        image = read_image(image_path, cv2.IMREAD_COLOR)
        detection_result = detector.detect_page(image_path)
        selected = _top_detections(detection_result.detections, image.shape[0], scan_ratio)
        recognition = recognizer.recognise_page(image_path, selected)
        blocks = _blocks(recognition.as_dict()["regions"])
        blocks = _recover_complete_pagination(image_path, image.shape[1], image.shape[0], blocks, recognizer)
        page_evidence = build_page_evidence(page_index, blocks, page_width=image.shape[1], page_height=image.shape[0])
        evidence.append(page_evidence)
        cache_pages.append({
            "page_index": page_index,
            "image_path": str(image_path),
            "evidence": page_evidence.as_dict(),
            "ocr_blocks": blocks,
            "detection_count": len(detection_result.detections),
            "recognised_top_region_count": len(blocks),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
        })
    return evidence, cache_pages
