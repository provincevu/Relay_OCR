# Mục đích: thử nghiệm Rule Induction tương tác cho key-value trên một page.
# Chức năng: OCR ảnh, tạo HTML cho phép chọn mẫu và xem preview tức thời.
# Luồng xử lý: ảnh → OCR regions → HTML tương tác → chọn mẫu → suy luận rule.
# Cấu hình/parameter: nhận một đường dẫn ảnh page 3+.
# Quan hệ: sử dụng OCR từ test_table_key_value_page3_plus.py.
# Input/output: đọc một ảnh; ghi ảnh nguồn và rule_induction.html.
# Contract: rule chỉ tồn tại trong HTML của đúng page và không được lưu lại.

import argparse
import json
import shutil
from html import escape
from pathlib import Path

import cv2

from test_table_key_value_page3_plus import Page3PlusKeyValueExtractor


def parse_arguments():
    """Đọc đường dẫn một ảnh cần tạo giao diện Rule Induction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    return parser.parse_args()


def build_html(image_name, image_width, image_height, regions):
    """Tạo HTML tương tác để chọn mẫu và suy luận key-value."""
    regions_json = json.dumps(regions, ensure_ascii=False).replace(
        "</",
        "<\\/",
    )
    template = r"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rule Induction key-value</title>
    <style>
        :root {
            --xam: #64748b;
            --xanh-la: #15803d;
            --xanh-duong: #0369a1;
            --tim: #7c3aed;
            --hong: #c026d3;
            --cam: #ea580c;
            --nen: #f1f5f9;
            --chu: #172033;
        }
        * { box-sizing: border-box; }
        body {
            background: var(--nen);
            color: var(--chu);
            font-family: Arial, sans-serif;
            margin: 0;
        }
        header {
            background: linear-gradient(135deg, #0f172a, #1e3a5f);
            color: white;
            padding: 22px 28px;
        }
        header h1 { font-size: 24px; margin: 0 0 8px; }
        header p { color: #dbeafe; margin: 0; }
        .app { margin: 0 auto; max-width: 1800px; padding: 20px; }
        .control-panel {
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
            padding: 18px;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .steps { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
        .step {
            background: #e2e8f0;
            border-radius: 999px;
            color: #475569;
            font-size: 14px;
            font-weight: 700;
            padding: 8px 13px;
        }
        .step.active { background: #dbeafe; color: #1d4ed8; }
        .step.done { background: #dcfce7; color: #166534; }
        .instruction {
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            font-size: 17px;
            font-weight: 700;
            line-height: 1.5;
            margin-bottom: 12px;
            padding: 13px 15px;
        }
        .instruction.wait-value {
            background: #ecfdf5;
            border-color: var(--xanh-la);
            color: #14532d;
        }
        .instruction.confirmed {
            background: #f0fdf4;
            border-color: #16a34a;
            color: #14532d;
        }
        .hover-info {
            color: #475569;
            font-size: 14px;
            min-height: 22px;
        }
        .actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
        button {
            border: 0;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 700;
            padding: 10px 14px;
        }
        .actions button:disabled { cursor: not-allowed; opacity: 0.45; }
        .secondary { background: #e2e8f0; color: #334155; }
        .danger { background: #fee2e2; color: #991b1b; }
        .confirm { background: #16a34a; color: white; margin-left: auto; }
        .workspace {
            align-items: start;
            display: grid;
            gap: 18px;
            grid-template-columns: minmax(0, 1fr) 360px;
        }
        .card {
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06);
            padding: 15px;
        }
        .card h2 { font-size: 18px; margin: 0 0 12px; }
        .image-toolbar {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; }
        .legend-item { align-items: center; display: flex; gap: 5px; }
        .swatch { border: 2px solid; height: 13px; width: 22px; }
        .swatch.detected {
            background: rgba(100, 116, 139, 0.14);
            border-color: var(--xam);
        }
        .swatch.manual-key {
            background: rgba(21, 128, 61, 0.18);
            border-color: var(--xanh-la);
        }
        .swatch.manual-value {
            background: rgba(3, 105, 161, 0.18);
            border-color: var(--xanh-duong);
        }
        .swatch.predicted {
            background: rgba(124, 58, 237, 0.16);
            border-color: var(--tim);
        }
        .image-stage {
            background: #e5e7eb;
            border: 1px solid #94a3b8;
            line-height: 0;
            position: relative;
            user-select: none;
            width: 100%;
        }
        .image-stage img { display: block; height: auto; width: 100%; }
        .box-layer, .link-layer {
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 100%;
        }
        .link-layer { pointer-events: none; z-index: 5; }
        .box-layer { z-index: 10; }
        .ocr-box {
            background: rgba(100, 116, 139, 0.10);
            border: 2px solid rgba(71, 85, 105, 0.82);
            cursor: pointer;
            min-height: 5px;
            min-width: 5px;
            position: absolute;
            transition: background 100ms, border-color 100ms, box-shadow 100ms;
        }
        .ocr-box:hover {
            background: rgba(250, 204, 21, 0.22);
            border-color: #ca8a04;
            box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.40);
        }
        .box-label {
            background: rgba(51, 65, 85, 0.92);
            color: white;
            font-size: 10px;
            left: -2px;
            line-height: 1.2;
            padding: 2px 4px;
            position: absolute;
            top: -17px;
            white-space: nowrap;
        }
        .ocr-box.ocr-box.current-key {
            animation: pulse 1.1s infinite;
            background: rgba(21, 128, 61, 0.25);
            border-color: var(--xanh-la);
            border-style: solid;
            border-width: 3px;
            box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.30);
        }
        .ocr-box.manual-key {
            background: rgba(21, 128, 61, 0.22);
            border-color: var(--xanh-la);
            border-width: 3px;
        }
        .ocr-box.manual-value {
            background: rgba(3, 105, 161, 0.22);
            border-color: var(--xanh-duong);
            border-width: 3px;
        }
        .ocr-box.predicted-key {
            background: rgba(124, 58, 237, 0.17);
            border-color: var(--tim);
            border-style: dashed;
        }
        .ocr-box.predicted-value {
            background: rgba(192, 38, 211, 0.17);
            border-color: var(--hong);
            border-style: dashed;
        }
        @keyframes pulse {
            50% { box-shadow: 0 0 0 6px rgba(21, 128, 61, 0.10); }
        }
        .side-panel { display: grid; gap: 18px; position: sticky; top: 190px; }
        .empty {
            background: #f8fafc;
            border: 1px dashed #94a3b8;
            border-radius: 8px;
            color: #64748b;
            line-height: 1.45;
            padding: 12px;
        }
        .list { display: grid; gap: 8px; max-height: 330px; overflow: auto; }
        .pair-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.45;
            padding: 9px;
        }
        .pair-item strong { color: #0f172a; }
        .pair-arrow { color: var(--cam); font-weight: 700; padding: 0 5px; }
        .rule-summary {
            background: #faf5ff;
            border: 1px solid #ddd6fe;
            border-radius: 8px;
            color: #4c1d95;
            font-size: 13px;
            line-height: 1.5;
            margin-top: 10px;
            padding: 10px;
        }
        .result-section { margin-top: 20px; }
        .result-section[hidden] { display: none; }
        table { border-collapse: collapse; width: 100%; }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }
        th { background: #e2e8f0; }
        .source-manual { color: var(--xanh-la); font-weight: 700; }
        .source-rule { color: var(--tim); font-weight: 700; }
        @media (max-width: 1100px) {
            .workspace { grid-template-columns: 1fr; }
            .side-panel { position: static; }
            .confirm { margin-left: 0; }
        }
    </style>
</head>
<body>
<header>
    <h1>Rule Induction key-value</h1>
    <p>
        Chọn ví dụ trực tiếp trên ảnh và xem chương trình suy luận lại
        ngay sau mỗi cặp.
    </p>
</header>
<main class="app">
    <section class="control-panel">
        <div class="steps">
            <span class="step active" id="stepKey">1. Chọn key</span>
            <span class="step" id="stepValue">2. Chọn value</span>
            <span class="step" id="stepPreview">3. Kiểm tra preview</span>
            <span class="step" id="stepConfirm">4. Xác nhận</span>
        </div>
        <div class="instruction" id="instruction" role="status" aria-live="polite">
            Bước 1: Hãy bấm vào bbox chứa key đầu tiên trên ảnh.
        </div>
        <div class="hover-info" id="hoverInfo">
            Di chuột lên bbox để xem ID, nội dung OCR và độ tin cậy.
        </div>
        <div class="actions">
            <button class="secondary" id="cancelKey" disabled>
                Hủy key đang chọn
            </button>
            <button class="secondary" id="undoExample" disabled>
                Xóa mẫu gần nhất
            </button>
            <button class="danger" id="resetAll" disabled>
                Làm lại từ đầu
            </button>
            <button class="confirm" id="confirmResult" disabled>
                Xác nhận kết quả
            </button>
        </div>
    </section>

    <section class="workspace">
        <div class="card">
            <div class="image-toolbar">
                <h2>Ảnh và các bbox OCR</h2>
                <div class="legend">
                    <span class="legend-item">
                        <span class="swatch detected"></span>Đã OCR
                    </span>
                    <span class="legend-item">
                        <span class="swatch manual-key"></span>Key đã chọn
                    </span>
                    <span class="legend-item">
                        <span class="swatch manual-value"></span>Value đã chọn
                    </span>
                    <span class="legend-item">
                        <span class="swatch predicted"></span>Preview suy luận
                    </span>
                </div>
            </div>
            <div class="image-stage" id="imageStage">
                <img id="pageImage" src="__IMAGE_NAME__" alt="Ảnh page cần phân tích">
                <svg class="link-layer" id="linkLayer"
                     viewBox="0 0 __IMAGE_WIDTH__ __IMAGE_HEIGHT__"
                     preserveAspectRatio="none"></svg>
                <div class="box-layer" id="boxLayer"></div>
            </div>
        </div>

        <aside class="side-panel">
            <section class="card">
                <h2>Mẫu người dùng đã chọn (<span id="exampleCount">0</span>)</h2>
                <div class="empty" id="exampleEmpty">
                    Chưa có mẫu. Bấm một bbox key, sau đó bấm bbox value tương ứng.
                </div>
                <div class="list" id="exampleList"></div>
            </section>

            <section class="card">
                <h2>Preview suy luận (<span id="previewCount">0</span>)</h2>
                <div class="empty" id="previewEmpty">
                    Preview sẽ xuất hiện ngay sau khi bạn chọn xong cặp đầu tiên.
                </div>
                <div class="list" id="previewList"></div>
                <div class="rule-summary" id="ruleSummary">
                    Chưa có rule. Số mẫu không bị giới hạn.
                </div>
            </section>
        </aside>
    </section>

    <section class="card result-section" id="resultSection" hidden>
        <h2>Kết quả key-value đã xác nhận</h2>
        <p id="resultSummary"></p>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Key</th><th>Value</th>
                    <th>Nguồn</th><th>Độ tin cậy</th>
                </tr>
            </thead>
            <tbody id="resultBody"></tbody>
        </table>
    </section>
</main>

<script>
    const PAGE_WIDTH = __IMAGE_WIDTH__;
    const PAGE_HEIGHT = __IMAGE_HEIGHT__;
    const regions = __REGIONS_JSON__;
    const regionById = new Map(regions.map(region => [region.id, region]));

    const state = {
        currentKeyId: null,
        examples: [],
        predictions: [],
        rule: null,
        confirmed: false,
    };

    const boxLayer = document.getElementById("boxLayer");
    const linkLayer = document.getElementById("linkLayer");
    const instruction = document.getElementById("instruction");
    const hoverInfo = document.getElementById("hoverInfo");
    const cancelKeyButton = document.getElementById("cancelKey");
    const undoExampleButton = document.getElementById("undoExample");
    const resetAllButton = document.getElementById("resetAll");
    const confirmResultButton = document.getElementById("confirmResult");
    const resultSection = document.getElementById("resultSection");

    function centerX(region) {
        return (region.box[0] + region.box[2]) / 2;
    }

    function centerY(region) {
        return (region.box[1] + region.box[3]) / 2;
    }

    function regionHeight(region) {
        return Math.max(region.box[3] - region.box[1], 1);
    }

    function verticalOverlap(keyRegion, valueRegion) {
        const overlap = Math.max(
            0,
            Math.min(keyRegion.box[3], valueRegion.box[3])
                - Math.max(keyRegion.box[1], valueRegion.box[1]),
        );
        return overlap / Math.min(
            regionHeight(keyRegion),
            regionHeight(valueRegion),
        );
    }

    function median(values) {
        const ordered = [...values].sort((left, right) => left - right);
        const middle = Math.floor(ordered.length / 2);
        if (ordered.length % 2 === 1) {
            return ordered[middle];
        }
        return (ordered[middle - 1] + ordered[middle]) / 2;
    }

    function maxDeviation(values, expected) {
        return Math.max(...values.map(value => Math.abs(value - expected)));
    }

    function textType(text) {
        const normalized = text.trim();
        if (/^(Y|N|ON|OFF|YES|NO|ENABLED|DISABLED|ENABLE|DISABLE)$/i.test(normalized)) {
            return "lựa chọn";
        }
        if (/^[+-]?[0-9.,]+\s*[A-Za-z%°Ω/'-]+$/u.test(normalized)) {
            return "số kèm đơn vị";
        }
        if (/^[+-]?[0-9.,]+$/u.test(normalized)) {
            return "số";
        }
        if (/^(?=.*[0-9])[A-Z0-9_.<>/+-]{2,16}$/u.test(normalized)) {
            return "mã";
        }
        return "văn bản";
    }

    function pairFeatures(keyRegion, valueRegion) {
        const averageHeight = (
            regionHeight(keyRegion) + regionHeight(valueRegion)
        ) / 2;
        return {
            keyX: centerX(keyRegion) / PAGE_WIDTH,
            valueX: centerX(valueRegion) / PAGE_WIDTH,
            deltaX: (centerX(valueRegion) - centerX(keyRegion)) / PAGE_WIDTH,
            rowOffset: Math.abs(
                centerY(valueRegion) - centerY(keyRegion),
            ) / Math.max(averageHeight, 1),
            overlap: verticalOverlap(keyRegion, valueRegion),
            keyType: textType(keyRegion.text),
            valueType: textType(valueRegion.text),
        };
    }

    function inferRule() {
        if (state.examples.length === 0) {
            return null;
        }
        const features = state.examples.map(example => pairFeatures(
            regionById.get(example.keyId),
            regionById.get(example.valueId),
        ));
        const keyXs = features.map(feature => feature.keyX);
        const valueXs = features.map(feature => feature.valueX);
        const deltaXs = features.map(feature => feature.deltaX);
        const keyX = median(keyXs);
        const valueX = median(valueXs);
        const deltaX = median(deltaXs);
        const sampleCount = features.length;
        const baseColumnTolerance = sampleCount === 1
            ? 0.12
            : sampleCount === 2
                ? 0.075
                : 0.05;
        const keyTolerance = Math.max(
            baseColumnTolerance,
            maxDeviation(keyXs, keyX) * 1.5 + 0.015,
        );
        const valueTolerance = Math.max(
            baseColumnTolerance,
            maxDeviation(valueXs, valueX) * 1.5 + 0.015,
        );
        const deltaTolerance = Math.max(
            baseColumnTolerance * 1.5,
            maxDeviation(deltaXs, deltaX) * 1.5 + 0.025,
        );
        return {
            keyX,
            valueX,
            deltaX,
            keyTolerance,
            valueTolerance,
            deltaTolerance,
            direction: deltaX >= 0 ? 1 : -1,
            rowTolerance: Math.max(
                1.15,
                Math.max(...features.map(feature => feature.rowOffset)) + 0.4,
            ),
            minimumOverlap: Math.max(
                0,
                Math.min(...features.map(feature => feature.overlap)) - 0.25,
            ),
            keyTypes: new Set(features.map(feature => feature.keyType)),
            valueTypes: new Set(features.map(feature => feature.valueType)),
            sampleCount,
        };
    }

    function similarity(value, expected, tolerance) {
        return Math.max(0, 1 - Math.abs(value - expected) / tolerance);
    }

    function predictPairs(rule) {
        const candidates = [];
        const manualKeyIds = new Set(state.examples.map(example => example.keyId));
        const manualValueIds = new Set(state.examples.map(example => example.valueId));

        for (const keyRegion of regions) {
            if (manualKeyIds.has(keyRegion.id) || manualValueIds.has(keyRegion.id)) {
                continue;
            }
            for (const valueRegion of regions) {
                if (
                    keyRegion.id === valueRegion.id
                    || manualKeyIds.has(valueRegion.id)
                    || manualValueIds.has(valueRegion.id)
                ) {
                    continue;
                }
                const features = pairFeatures(keyRegion, valueRegion);
                if (rule.direction * features.deltaX <= 0) {
                    continue;
                }
                if (Math.abs(features.keyX - rule.keyX) > rule.keyTolerance) {
                    continue;
                }
                if (Math.abs(features.valueX - rule.valueX) > rule.valueTolerance) {
                    continue;
                }
                if (Math.abs(features.deltaX - rule.deltaX) > rule.deltaTolerance) {
                    continue;
                }
                if (features.rowOffset > rule.rowTolerance) {
                    continue;
                }
                if (features.overlap < rule.minimumOverlap) {
                    continue;
                }

                const keyColumnScore = similarity(
                    features.keyX,
                    rule.keyX,
                    rule.keyTolerance,
                );
                const valueColumnScore = similarity(
                    features.valueX,
                    rule.valueX,
                    rule.valueTolerance,
                );
                const distanceScore = similarity(
                    features.deltaX,
                    rule.deltaX,
                    rule.deltaTolerance,
                );
                const rowScore = Math.max(
                    0,
                    1 - features.rowOffset / rule.rowTolerance,
                );
                const typeScore = (
                    (rule.keyTypes.has(features.keyType) ? 0.5 : 0.25)
                    + (rule.valueTypes.has(features.valueType) ? 0.5 : 0.25)
                );
                const score = (
                    keyColumnScore * 0.24
                    + valueColumnScore * 0.24
                    + distanceScore * 0.20
                    + rowScore * 0.22
                    + typeScore * 0.10
                );
                if (score >= 0.58) {
                    candidates.push({
                        keyId: keyRegion.id,
                        valueId: valueRegion.id,
                        score,
                    });
                }
            }
        }

        candidates.sort((left, right) => right.score - left.score);
        const usedKeys = new Set(manualKeyIds);
        const usedValues = new Set(manualValueIds);
        const predictions = [];
        for (const candidate of candidates) {
            if (
                usedKeys.has(candidate.keyId)
                || usedValues.has(candidate.valueId)
                || usedValues.has(candidate.keyId)
                || usedKeys.has(candidate.valueId)
            ) {
                continue;
            }
            predictions.push(candidate);
            usedKeys.add(candidate.keyId);
            usedValues.add(candidate.valueId);
        }
        return predictions;
    }

    function renderBoxes() {
        boxLayer.replaceChildren();
        for (const region of regions) {
            const box = document.createElement("button");
            const width = region.box[2] - region.box[0];
            const height = region.box[3] - region.box[1];
            box.type = "button";
            box.className = "ocr-box";
            box.dataset.regionId = region.id;
            box.style.left = `${region.box[0] / PAGE_WIDTH * 100}%`;
            box.style.top = `${region.box[1] / PAGE_HEIGHT * 100}%`;
            box.style.width = `${width / PAGE_WIDTH * 100}%`;
            box.style.height = `${height / PAGE_HEIGHT * 100}%`;
            box.style.zIndex = String(
                Math.round(1000000 / Math.max(width * height, 1))
            );
            box.title = `${region.id}: ${region.text} (${region.score.toFixed(4)})`;

            const label = document.createElement("span");
            label.className = "box-label";
            label.textContent = region.id;
            box.appendChild(label);
            box.addEventListener("click", () => selectRegion(region.id));
            box.addEventListener("mouseenter", () => {
                hoverInfo.textContent = (
                    `${region.id} · OCR: “${region.text}” · `
                    + `Độ tin cậy: ${region.score.toFixed(4)}`
                );
            });
            box.addEventListener("mouseleave", () => {
                hoverInfo.textContent = (
                    "Di chuột lên bbox để xem ID, nội dung OCR và độ tin cậy."
                );
            });
            boxLayer.appendChild(box);
        }
        refreshBoxStyles();
    }

    function refreshBoxStyles() {
        const manualKeys = new Set(state.examples.map(example => example.keyId));
        const manualValues = new Set(state.examples.map(example => example.valueId));
        const predictedKeys = new Set(state.predictions.map(pair => pair.keyId));
        const predictedValues = new Set(state.predictions.map(pair => pair.valueId));

        for (const box of boxLayer.querySelectorAll(".ocr-box")) {
            const regionId = box.dataset.regionId;
            box.classList.toggle("current-key", state.currentKeyId === regionId);
            box.classList.toggle("manual-key", manualKeys.has(regionId));
            box.classList.toggle("manual-value", manualValues.has(regionId));
            box.classList.toggle("predicted-key", predictedKeys.has(regionId));
            box.classList.toggle("predicted-value", predictedValues.has(regionId));
            box.disabled = state.confirmed;
            const label = box.querySelector(".box-label");
            if (state.currentKeyId === regionId) {
                label.textContent = `${regionId} · KEY`;
            } else if (manualKeys.has(regionId)) {
                label.textContent = `${regionId} · KEY mẫu`;
            } else if (manualValues.has(regionId)) {
                label.textContent = `${regionId} · VALUE mẫu`;
            } else if (predictedKeys.has(regionId)) {
                label.textContent = `${regionId} · key?`;
            } else if (predictedValues.has(regionId)) {
                label.textContent = `${regionId} · value?`;
            } else {
                label.textContent = regionId;
            }
        }
    }

    function svgLine(pair, cssClass, color, dashArray) {
        const keyRegion = regionById.get(pair.keyId);
        const valueRegion = regionById.get(pair.valueId);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", centerX(keyRegion));
        line.setAttribute("y1", centerY(keyRegion));
        line.setAttribute("x2", centerX(valueRegion));
        line.setAttribute("y2", centerY(valueRegion));
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", "5");
        line.setAttribute("stroke-linecap", "round");
        line.setAttribute("class", cssClass);
        if (dashArray) {
            line.setAttribute("stroke-dasharray", dashArray);
        }
        linkLayer.appendChild(line);
    }

    function renderLinks() {
        linkLayer.replaceChildren();
        for (const example of state.examples) {
            svgLine(example, "manual-link", "#ea580c", "");
        }
        for (const prediction of state.predictions) {
            svgLine(prediction, "predicted-link", "#7c3aed", "16 12");
        }
    }

    function pairHtml(pair, includeScore) {
        const keyRegion = regionById.get(pair.keyId);
        const valueRegion = regionById.get(pair.valueId);
        const item = document.createElement("div");
        item.className = "pair-item";

        const key = document.createElement("strong");
        key.textContent = `${pair.keyId}: ${keyRegion.text}`;
        const arrow = document.createElement("span");
        arrow.className = "pair-arrow";
        arrow.textContent = "→";
        const value = document.createElement("strong");
        value.textContent = `${pair.valueId}: ${valueRegion.text}`;
        item.append(key, arrow, value);
        if (includeScore) {
            const score = document.createElement("div");
            score.textContent = `Độ tin cậy: ${(pair.score * 100).toFixed(1)}%`;
            item.appendChild(score);
        }
        return item;
    }

    function renderLists() {
        const exampleList = document.getElementById("exampleList");
        const previewList = document.getElementById("previewList");
        const exampleEmpty = document.getElementById("exampleEmpty");
        const previewEmpty = document.getElementById("previewEmpty");
        exampleList.replaceChildren();
        previewList.replaceChildren();
        for (const example of state.examples) {
            exampleList.appendChild(pairHtml(example, false));
        }
        for (const prediction of state.predictions) {
            previewList.appendChild(pairHtml(prediction, true));
        }
        document.getElementById("exampleCount").textContent = state.examples.length;
        document.getElementById("previewCount").textContent = state.predictions.length;
        exampleEmpty.hidden = state.examples.length > 0;
        previewEmpty.hidden = state.predictions.length > 0;
        if (state.examples.length > 0 && state.predictions.length === 0) {
            previewEmpty.textContent = (
                "Chưa tìm thấy cặp tương tự. Hãy chọn thêm một cặp mẫu khác "
                + "hoặc kiểm tra lại mẫu vừa chọn."
            );
        }
    }

    function renderRuleSummary() {
        const target = document.getElementById("ruleSummary");
        if (!state.rule) {
            target.textContent = "Chưa có rule. Số mẫu không bị giới hạn.";
            return;
        }
        target.textContent = (
            `Rule từ ${state.rule.sampleCount} mẫu: key quanh `
            + `${(state.rule.keyX * 100).toFixed(1)}% `
            + `chiều rộng page; value quanh ${(state.rule.valueX * 100).toFixed(1)}%; `
            + `value nằm ${state.rule.direction > 0 ? "bên phải" : "bên trái"} key; `
            + `preview ${state.predictions.length} cặp.`
        );
    }

    function updateSteps() {
        const stepKey = document.getElementById("stepKey");
        const stepValue = document.getElementById("stepValue");
        const stepPreview = document.getElementById("stepPreview");
        const stepConfirm = document.getElementById("stepConfirm");
        for (const step of [stepKey, stepValue, stepPreview, stepConfirm]) {
            step.classList.remove("active", "done");
        }
        if (state.confirmed) {
            for (const step of [stepKey, stepValue, stepPreview, stepConfirm]) {
                step.classList.add("done");
            }
            return;
        }
        if (state.currentKeyId) {
            stepKey.classList.add("done");
            stepValue.classList.add("active");
            return;
        }
        stepKey.classList.add("active");
        if (state.examples.length > 0) {
            stepPreview.classList.add("done");
            stepConfirm.classList.add("active");
        }
    }

    function updateButtons() {
        cancelKeyButton.disabled = !state.currentKeyId || state.confirmed;
        undoExampleButton.disabled = state.examples.length === 0 || state.confirmed;
        resetAllButton.disabled = (
            state.examples.length === 0 && !state.currentKeyId
        ) || state.confirmed;
        confirmResultButton.disabled = state.examples.length === 0 || state.confirmed;
    }

    function recomputePreview() {
        state.rule = inferRule();
        state.predictions = state.rule ? predictPairs(state.rule) : [];
        refreshBoxStyles();
        renderLinks();
        renderLists();
        renderRuleSummary();
        updateSteps();
        updateButtons();
    }

    function selectRegion(regionId) {
        if (state.confirmed) {
            return;
        }
        const region = regionById.get(regionId);
        const regionWasUsed = state.examples.some(example => (
            example.keyId === regionId || example.valueId === regionId
        ));
        if (!state.currentKeyId) {
            if (regionWasUsed) {
                instruction.className = "instruction";
                instruction.textContent = (
                    `${regionId} đã thuộc một cặp mẫu. `
                    + "Hãy chọn một bbox chưa được dùng làm key mới."
                );
                return;
            }
            state.currentKeyId = regionId;
            instruction.className = "instruction wait-value";
            instruction.textContent = (
                `Đã chọn key ${regionId}: “${region.text}”. `
                + "Bước tiếp theo: hãy bấm vào bbox chứa value tương ứng."
            );
            refreshBoxStyles();
            updateSteps();
            updateButtons();
            return;
        }
        if (state.currentKeyId === regionId) {
            instruction.textContent = (
                "Key và value đang là cùng một bbox. Phiên bản này yêu cầu "
                + "hai bbox khác nhau; "
                + "hãy bấm một bbox value khác hoặc hủy key đang chọn."
            );
            return;
        }
        if (regionWasUsed) {
            instruction.textContent = (
                `${regionId} đã thuộc một cặp mẫu. `
                + "Hãy bấm một bbox chưa được dùng làm value."
            );
            return;
        }

        const keyId = state.currentKeyId;
        state.examples.push({keyId, valueId: regionId});
        state.currentKeyId = null;
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = (
            `Đã thêm mẫu ${keyId} → ${regionId} và cập nhật preview. `
            + "Nếu preview chưa phù hợp, hãy bấm một key khác rồi chọn value của nó. "
            + "Nếu đã đúng, bấm “Xác nhận kết quả”."
        );
    }

    function cancelCurrentKey() {
        state.currentKeyId = null;
        instruction.className = "instruction";
        instruction.textContent = "Đã hủy key. Hãy bấm vào bbox chứa key muốn chọn.";
        refreshBoxStyles();
        updateSteps();
        updateButtons();
    }

    function undoLastExample() {
        state.examples.pop();
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = state.examples.length > 0
            ? "Đã xóa mẫu gần nhất và cập nhật preview. Bạn có thể chọn thêm key-value."
            : "Đã xóa hết mẫu. Hãy bấm vào bbox chứa key đầu tiên.";
    }

    function resetAll() {
        state.currentKeyId = null;
        state.examples = [];
        state.predictions = [];
        state.rule = null;
        state.confirmed = false;
        resultSection.hidden = true;
        instruction.className = "instruction";
        instruction.textContent = (
            "Bước 1: Hãy bấm vào bbox chứa key đầu tiên trên ảnh."
        );
        recomputePreview();
    }

    function appendResultRow(index, pair, source, score) {
        const keyRegion = regionById.get(pair.keyId);
        const valueRegion = regionById.get(pair.valueId);
        const row = document.createElement("tr");
        const values = [
            String(index),
            keyRegion.text,
            valueRegion.text,
            source,
            score,
        ];
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        }
        row.children[3].className = source === "Người dùng chọn"
            ? "source-manual"
            : "source-rule";
        document.getElementById("resultBody").appendChild(row);
    }

    function confirmResults() {
        state.confirmed = true;
        state.currentKeyId = null;
        const resultBody = document.getElementById("resultBody");
        resultBody.replaceChildren();
        let index = 1;
        for (const example of state.examples) {
            appendResultRow(index, example, "Người dùng chọn", "100.0%");
            index += 1;
        }
        for (const prediction of state.predictions) {
            appendResultRow(
                index,
                prediction,
                "Rule suy luận",
                `${(prediction.score * 100).toFixed(1)}%`,
            );
            index += 1;
        }
        document.getElementById("resultSummary").textContent = (
            `Đã xác nhận ${index - 1} cặp: ${state.examples.length} cặp mẫu và `
            + `${state.predictions.length} cặp do Rule Induction suy luận.`
        );
        resultSection.hidden = false;
        instruction.className = "instruction confirmed";
        instruction.textContent = (
            "Đã xác nhận kết quả. Bảng key-value đã được hiển thị phía dưới ảnh; "
            + "dữ liệu chưa được lưu ra file."
        );
        refreshBoxStyles();
        updateSteps();
        updateButtons();
        resultSection.scrollIntoView({behavior: "smooth", block: "start"});
    }

    cancelKeyButton.addEventListener("click", cancelCurrentKey);
    undoExampleButton.addEventListener("click", undoLastExample);
    resetAllButton.addEventListener("click", resetAll);
    confirmResultButton.addEventListener("click", confirmResults);

    renderBoxes();
    renderLinks();
    renderLists();
    renderRuleSummary();
    updateSteps();
    updateButtons();
</script>
</body>
</html>
"""
    html = template.replace("__IMAGE_NAME__", escape(image_name))
    html = html.replace("__IMAGE_WIDTH__", str(image_width))
    html = html.replace("__IMAGE_HEIGHT__", str(image_height))
    return html.replace("__REGIONS_JSON__", regions_json)


def main():
    """Chạy OCR cho một ảnh và tạo giao diện Rule Induction."""
    arguments = parse_arguments()
    image = cv2.imread(str(arguments.image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {arguments.image_path}")

    extractor = Page3PlusKeyValueExtractor()
    text_regions, _ = extractor._recognize_text(image)
    text_regions.sort(
        key=lambda region: (
            region["center"][1],
            region["center"][0],
        )
    )
    regions = [
        {
            "id": f"O{index}",
            "text": region["text"],
            "box": [int(coordinate) for coordinate in region["box"]],
            "score": float(region["score"]),
        }
        for index, region in enumerate(text_regions, start=1)
    ]

    output_directory = (
        Path("./output/rule_induction") / arguments.image_path.parent.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_image_path = output_directory / arguments.image_path.name
    shutil.copyfile(arguments.image_path, output_image_path)

    image_height, image_width = image.shape[:2]
    html_path = output_directory / "rule_induction.html"
    html_path.write_text(
        build_html(
            output_image_path.name,
            image_width,
            image_height,
            regions,
        ),
        encoding="utf-8",
    )
    print(f"HTML Rule Induction: {html_path}")


if __name__ == "__main__":
    main()
