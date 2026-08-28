# Mục đích: thử nghiệm Rule Induction tương tác cho key-value trên một page.
# Chức năng: OCR ảnh, tạo HTML cho phép chọn mẫu và xem preview tức thời.
# Luồng xử lý: ảnh → OCR regions → HTML tương tác → chọn mẫu → suy luận rule.
# Cấu hình/parameter: nhận một đường dẫn ảnh page 3+.
# Quan hệ: sử dụng OCR từ test_table_key_value_page3_plus.py.
# Input/output: đọc một ảnh; ghi ảnh nguồn và rule_induction.html.
# Contract: rule chỉ áp dụng từ mẫu đầu tiên trở xuống trong đúng page.

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
        .box-layer, .link-layer, .logical-layer {
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 100%;
        }
        .link-layer { pointer-events: none; z-index: 5; }
        .box-layer { z-index: 10; }
        .logical-layer { pointer-events: none; z-index: 20; }
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
        .logical-box {
            border: 4px solid;
            position: absolute;
        }
        .logical-box span {
            color: white;
            font-size: 12px;
            font-weight: 700;
            left: -4px;
            line-height: 1.2;
            padding: 3px 6px;
            position: absolute;
            top: -22px;
            white-space: nowrap;
        }
        .logical-box.manual-key { border-color: var(--xanh-la); }
        .logical-box.manual-key span { background: var(--xanh-la); }
        .logical-box.manual-value { border-color: var(--xanh-duong); }
        .logical-box.manual-value span { background: var(--xanh-duong); }
        .logical-box.predicted-key {
            border-color: var(--tim);
            border-style: dashed;
        }
        .logical-box.predicted-key span { background: var(--tim); }
        .logical-box.predicted-value {
            border-color: var(--hong);
            border-style: dashed;
        }
        .logical-box.predicted-value span { background: var(--hong); }
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
            Di chuột lên bbox để xem nội dung OCR và độ tin cậy.
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
                <div class="logical-layer" id="logicalLayer"></div>
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
    const logicalLayer = document.getElementById("logicalLayer");
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

    function combineRegions(parts) {
        const ordered = [...parts].sort((left, right) => (
            centerY(left) - centerY(right) || centerX(left) - centerX(right)
        ));
        const regionIds = ordered.map(region => region.id);
        return {
            id: regionIds.join("+"),
            regionIds,
            text: ordered.map(region => region.text).join(" "),
            box: [
                Math.min(...ordered.map(region => region.box[0])),
                Math.min(...ordered.map(region => region.box[1])),
                Math.max(...ordered.map(region => region.box[2])),
                Math.max(...ordered.map(region => region.box[3])),
            ],
            score: ordered.reduce((sum, region) => sum + region.score, 0)
                / ordered.length,
        };
    }

    function generateBlocks(columnRegions) {
        const ordered = [...columnRegions].sort((left, right) => (
            centerY(left) - centerY(right) || centerX(left) - centerX(right)
        ));
        const centerGaps = [];
        for (let index = 1; index < ordered.length; index += 1) {
            const gap = centerY(ordered[index]) - centerY(ordered[index - 1]);
            if (gap > 0) {
                centerGaps.push(gap);
            }
        }
        const typicalRowPitch = centerGaps.length >= 3
            ? median(centerGaps)
            : Number.POSITIVE_INFINITY;
        const blocks = [];
        for (let start = 0; start < ordered.length; start += 1) {
            const parts = [];
            for (let end = start; end < ordered.length; end += 1) {
                if (end > start) {
                    const previous = ordered[end - 1];
                    const current = ordered[end];
                    const verticalGap = current.box[1] - previous.box[3];
                    const centerGap = centerY(current) - centerY(previous);
                    const allowedGap = Math.max(
                        regionHeight(previous),
                        regionHeight(current),
                    ) * 0.55;
                    if (
                        verticalGap > allowedGap
                        || centerGap >= typicalRowPitch * 0.85
                    ) {
                        break;
                    }
                }
                parts.push(ordered[end]);
                blocks.push(combineRegions(parts));
            }
        }
        return blocks;
    }

    function blockContains(block, regionId) {
        return block.regionIds.includes(regionId);
    }

    function blockUsesAny(block, regionIds) {
        return block.regionIds.some(regionId => regionIds.has(regionId));
    }

    function pairFeatures(keyRegion, valueRegion) {
        const averageHeight = (
            regionHeight(keyRegion) + regionHeight(valueRegion)
        ) / 2;
        return {
            keyX: keyRegion.box[0] / PAGE_WIDTH,
            valueX: valueRegion.box[0] / PAGE_WIDTH,
            deltaX: (valueRegion.box[0] - keyRegion.box[0]) / PAGE_WIDTH,
            rowOffset: Math.abs(
                centerY(valueRegion) - centerY(keyRegion),
            ) / Math.max(averageHeight, 1),
            overlap: verticalOverlap(keyRegion, valueRegion),
            keyType: textType(keyRegion.text),
            valueType: textType(valueRegion.text),
        };
    }

    function buildExample(keyId, valueId) {
        const keyAnchor = regionById.get(keyId);
        const valueAnchor = regionById.get(valueId);
        const usedRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        const keyColumn = regions.filter(region => (
            !usedRegionIds.has(region.id)
            && Math.abs(region.box[0] - keyAnchor.box[0]) / PAGE_WIDTH <= 0.055
        ));
        const valueColumn = regions.filter(region => (
            !usedRegionIds.has(region.id)
            && Math.abs(region.box[0] - valueAnchor.box[0]) / PAGE_WIDTH <= 0.055
        ));
        const keyBlocks = generateBlocks(keyColumn).filter(block => (
            blockContains(block, keyId)
        ));
        const valueBlocks = generateBlocks(valueColumn).filter(block => (
            blockContains(block, valueId)
        ));
        const direction = valueAnchor.box[0] >= keyAnchor.box[0] ? 1 : -1;
        let bestExample = null;
        let bestCost = Number.POSITIVE_INFINITY;

        for (const key of keyBlocks) {
            for (const value of valueBlocks) {
                const features = pairFeatures(key, value);
                if (direction * features.deltaX <= 0) {
                    continue;
                }
                const extraLines = (
                    key.regionIds.length + value.regionIds.length - 2
                );
                const cost = (
                    features.rowOffset * 0.55
                    + (1 - features.overlap) * 0.35
                    + extraLines * 0.05
                );
                if (cost < bestCost) {
                    bestCost = cost;
                    bestExample = {key, value};
                }
            }
        }
        return bestExample;
    }

    function inferRule() {
        if (state.examples.length === 0) {
            return null;
        }
        const features = state.examples.map(example => pairFeatures(
            example.key,
            example.value,
        ));
        const keyXs = features.map(feature => feature.keyX);
        const valueXs = features.map(feature => feature.valueX);
        const deltaXs = features.map(feature => feature.deltaX);
        const keyX = median(keyXs);
        const valueX = median(valueXs);
        const deltaX = median(deltaXs);
        const firstExample = state.examples[0];
        const startY = (
            centerY(firstExample.key) + centerY(firstExample.value)
        ) / 2;
        const sampleCount = features.length;
        const baseColumnTolerance = sampleCount === 1
            ? 0.06
            : sampleCount === 2
                ? 0.04
                : 0.03;
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
            startY,
            sampleCount,
        };
    }

    function similarity(value, expected, tolerance) {
        return Math.max(0, 1 - Math.abs(value - expected) / tolerance);
    }

    function predictPairs(rule) {
        const candidates = [];
        const manualRegionIds = new Set(state.examples.flatMap(example => (
            [...example.key.regionIds, ...example.value.regionIds]
        )));
        const keyRegions = regions.filter(region => (
            !manualRegionIds.has(region.id)
            && Math.abs(region.box[0] / PAGE_WIDTH - rule.keyX)
                <= rule.keyTolerance
        ));
        const valueRegions = regions.filter(region => (
            !manualRegionIds.has(region.id)
            && Math.abs(region.box[0] / PAGE_WIDTH - rule.valueX)
                <= rule.valueTolerance
        ));
        const keyBlocks = generateBlocks(keyRegions);
        const valueBlocks = generateBlocks(valueRegions);

        for (const keyRegion of keyBlocks) {
            for (const valueRegion of valueBlocks) {
                const keyIds = new Set(keyRegion.regionIds);
                if (valueRegion.regionIds.some(regionId => keyIds.has(regionId))) {
                    continue;
                }
                const valueIds = new Set(valueRegion.regionIds);
                const crossesOtherKey = keyRegions.some(region => (
                    !keyIds.has(region.id)
                    && centerY(region) >= valueRegion.box[1]
                    && centerY(region) <= valueRegion.box[3]
                ));
                const crossesOtherValue = valueRegions.some(region => (
                    !valueIds.has(region.id)
                    && centerY(region) >= keyRegion.box[1]
                    && centerY(region) <= keyRegion.box[3]
                ));
                if (crossesOtherKey || crossesOtherValue) {
                    continue;
                }
                const features = pairFeatures(keyRegion, valueRegion);
                const candidateY = (
                    centerY(keyRegion) + centerY(valueRegion)
                ) / 2;
                if (candidateY < rule.startY) {
                    continue;
                }
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
                const groupPenalty = (
                    keyRegion.regionIds.length
                    + valueRegion.regionIds.length
                    - 2
                ) * 0.015;
                const score = (
                    keyColumnScore * 0.20
                    + valueColumnScore * 0.20
                    + distanceScore * 0.16
                    + rowScore * 0.19
                    + features.overlap * 0.15
                    + typeScore * 0.10
                    - groupPenalty
                );
                if (score >= 0.58) {
                    candidates.push({
                        key: keyRegion,
                        value: valueRegion,
                        score,
                    });
                }
            }
        }

        candidates.sort((left, right) => right.score - left.score);
        const usedRegionIds = new Set(manualRegionIds);
        const predictions = [];
        for (const candidate of candidates) {
            if (
                blockUsesAny(candidate.key, usedRegionIds)
                || blockUsesAny(candidate.value, usedRegionIds)
            ) {
                continue;
            }
            predictions.push(candidate);
            for (const regionId of candidate.key.regionIds) {
                usedRegionIds.add(regionId);
            }
            for (const regionId of candidate.value.regionIds) {
                usedRegionIds.add(regionId);
            }
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
            box.title = `${region.text} (${region.score.toFixed(4)})`;
            box.addEventListener("click", () => selectRegion(region.id));
            box.addEventListener("mouseenter", () => {
                hoverInfo.textContent = (
                    `OCR: “${region.text}” · `
                    + `Độ tin cậy: ${region.score.toFixed(4)}`
                );
            });
            box.addEventListener("mouseleave", () => {
                hoverInfo.textContent = (
                    "Di chuột lên bbox để xem nội dung OCR và độ tin cậy."
                );
            });
            boxLayer.appendChild(box);
        }
        refreshBoxStyles();
    }

    function appendLogicalBox(block, cssClass, labelText) {
        if (block.regionIds.length === 1) {
            return;
        }
        const box = document.createElement("div");
        box.className = `logical-box ${cssClass}`;
        box.style.left = `${block.box[0] / PAGE_WIDTH * 100}%`;
        box.style.top = `${block.box[1] / PAGE_HEIGHT * 100}%`;
        box.style.width = `${(block.box[2] - block.box[0]) / PAGE_WIDTH * 100}%`;
        box.style.height = `${(block.box[3] - block.box[1]) / PAGE_HEIGHT * 100}%`;
        const label = document.createElement("span");
        label.textContent = `${labelText} · ${block.regionIds.length} bbox`;
        box.appendChild(label);
        logicalLayer.appendChild(box);
    }

    function renderLogicalBlocks() {
        logicalLayer.replaceChildren();
        for (const example of state.examples) {
            appendLogicalBox(example.key, "manual-key", "KEY gộp");
            appendLogicalBox(example.value, "manual-value", "VALUE gộp");
        }
        for (const prediction of state.predictions) {
            appendLogicalBox(prediction.key, "predicted-key", "key? gộp");
            appendLogicalBox(
                prediction.value,
                "predicted-value",
                "value? gộp",
            );
        }
    }

    function refreshBoxStyles() {
        const manualKeys = new Set(state.examples.flatMap(example => (
            example.key.regionIds
        )));
        const manualValues = new Set(state.examples.flatMap(example => (
            example.value.regionIds
        )));
        const predictedKeys = new Set(state.predictions.flatMap(pair => (
            pair.key.regionIds
        )));
        const predictedValues = new Set(state.predictions.flatMap(pair => (
            pair.value.regionIds
        )));

        for (const box of boxLayer.querySelectorAll(".ocr-box")) {
            const regionId = box.dataset.regionId;
            box.classList.toggle("current-key", state.currentKeyId === regionId);
            box.classList.toggle("manual-key", manualKeys.has(regionId));
            box.classList.toggle("manual-value", manualValues.has(regionId));
            box.classList.toggle("predicted-key", predictedKeys.has(regionId));
            box.classList.toggle("predicted-value", predictedValues.has(regionId));
            box.disabled = state.confirmed;
        }
    }

    function svgLine(pair, cssClass, color, dashArray) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", centerX(pair.key));
        line.setAttribute("y1", centerY(pair.key));
        line.setAttribute("x2", centerX(pair.value));
        line.setAttribute("y2", centerY(pair.value));
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
        const item = document.createElement("div");
        item.className = "pair-item";

        const key = document.createElement("strong");
        key.textContent = pair.key.text;
        const arrow = document.createElement("span");
        arrow.className = "pair-arrow";
        arrow.textContent = "→";
        const value = document.createElement("strong");
        value.textContent = pair.value.text;
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
            + "chỉ suy luận từ hàng của mẫu đầu tiên trở xuống; "
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
        renderLogicalBlocks();
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
            example.key.regionIds.includes(regionId)
            || example.value.regionIds.includes(regionId)
        ));
        if (!state.currentKeyId) {
            if (regionWasUsed) {
                instruction.className = "instruction";
                instruction.textContent = (
                    `“${region.text}” đã thuộc một cặp mẫu. `
                    + "Hãy chọn một bbox chưa được dùng làm key mới."
                );
                return;
            }
            state.currentKeyId = regionId;
            instruction.className = "instruction wait-value";
            instruction.textContent = (
                `Đã chọn key: “${region.text}”. `
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
                `“${region.text}” đã thuộc một cặp mẫu. `
                + "Hãy bấm một bbox chưa được dùng làm value."
            );
            return;
        }

        const keyId = state.currentKeyId;
        const example = buildExample(keyId, regionId);
        if (!example) {
            instruction.textContent = (
                "Không tạo được cặp logical block từ hai bbox này. "
                + "Hãy bấm một bbox value ở cột đối diện key."
            );
            return;
        }
        state.examples.push(example);
        state.currentKeyId = null;
        recomputePreview();
        instruction.className = "instruction";
        instruction.textContent = (
            `Đã thêm mẫu “${example.key.text}” → “${example.value.text}” `
            + "và cập nhật preview. "
            + "Chương trình đã tự gộp các bbox liền kề khi chúng thuộc cùng hàng. "
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
        const row = document.createElement("tr");
        const values = [
            String(index),
            pair.key.text,
            pair.value.text,
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
