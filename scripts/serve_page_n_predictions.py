# Mục đích: hiển thị kết quả suy luận page 3+ bằng giao diện chỉ đọc.
# Chức năng: xem ảnh, bbox OCR, nhãn Layout model và link KEY → VALUE.
# Luồng xử lý: đọc predictions.json → tạo HTML → phục vụ qua HTTP localhost.
# Cấu hình/parameter: thư mục output và port HTTP.
# Quan hệ: phục vụ output page 3+ của infer_page_n_text_layout.py.
# Input/output: đọc predictions.json và ảnh; ghi predictions.html tạm thời.
# Contract: không có API POST và không sửa predictions.json hoặc annotations.json.

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__:
    from .page_n_prediction_filter import filter_prediction_links
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from page_n_prediction_filter import filter_prediction_links


HOST = "127.0.0.1"
PORT = 8002


def build_html(dataset):
    """Tạo giao diện chỉ đọc từ predictions.json."""
    pages = [
        filter_prediction_links(page)
        for page in dataset["pages"]
    ]
    pages_json = json.dumps(pages, ensure_ascii=False).replace(
        "</",
        "<\\/",
    )
    template = r"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Layout prediction page 3+</title>
    <style>
        :root {
            --background: #f1f5f9;
            --text: #172033;
            --muted: #64748b;
            --border: #cbd5e1;
            --panel: #ffffff;
            --key: #2563eb;
            --value: #16a34a;
            --other: #64748b;
        }
        * { box-sizing: border-box; }
        body {
            background: var(--background);
            color: var(--text);
            font-family: Arial, sans-serif;
            margin: 0;
        }
        header {
            background: linear-gradient(135deg, #0f172a, #1e3a5f);
            color: white;
            padding: 18px 24px;
        }
        header h1 { font-size: 22px; margin: 0 0 7px; }
        header p { color: #dbeafe; margin: 0; }
        .app { margin: 0 auto; max-width: 1900px; padding: 18px; }
        .toolbar, .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            box-shadow: 0 3px 14px rgba(15, 23, 42, .07);
        }
        .toolbar {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            padding: 13px 15px;
        }
        .toolbar label { font-weight: 700; }
        select { font: inherit; padding: 7px 9px; }
        button { cursor: pointer; font: inherit; padding: 7px 9px; }
        button:disabled { cursor: not-allowed; opacity: .5; }
        .toolbar span { color: var(--muted); font-size: 14px; }
        .toolbar input { margin-right: 4px; }
        .layout {
            align-items: start;
            display: grid;
            gap: 16px;
            grid-template-columns: minmax(0, 1fr) 390px;
        }
        .image-panel { padding: 14px; }
        #stage {
            background: #e2e8f0;
            max-width: 100%;
            overflow: hidden;
            position: relative;
        }
        #pageImage {
            display: block;
            height: auto;
            user-select: none;
            width: 100%;
        }
        #linkLayer { height: 100%; left: 0; pointer-events: none; position: absolute; top: 0; width: 100%; }
        .region-box {
            border: 2px solid var(--other);
            cursor: pointer;
            overflow: visible;
            position: absolute;
        }
        .region-box:hover, .region-box.selected { box-shadow: 0 0 0 3px #facc15; z-index: 10; }
        .region-box.key { border-color: var(--key); background: #2563eb2b; }
        .region-box.value { border-color: var(--value); background: #16a34a2b; }
        .region-box.other { border-color: var(--other); background: #64748b1a; }
        .region-label {
            background: currentColor;
            color: white;
            font-size: 11px;
            font-weight: 700;
            left: -2px;
            line-height: 1;
            padding: 3px 4px;
            position: absolute;
            top: -18px;
            white-space: nowrap;
        }
        .key .region-label { color: var(--key); }
        .value .region-label { color: var(--value); }
        .other .region-label { color: var(--other); }
        .panel { padding: 14px; }
        .panel h2 { font-size: 17px; margin: 0 0 10px; }
        .summary { color: var(--muted); font-size: 13px; line-height: 1.5; margin-bottom: 12px; }
        .legend { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
        .legend span { border-radius: 5px; color: white; font-size: 12px; padding: 4px 6px; }
        .legend .key { background: var(--key); }
        .legend .value { background: var(--value); }
        .legend .other { background: var(--other); }
        .detail {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 7px;
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 13px;
            min-height: 45px;
            padding: 8px;
            white-space: pre-wrap;
        }
        .link-section { margin-top: 14px; }
        .link-section h3 { font-size: 14px; margin: 0 0 7px; }
        .link-list { display: grid; gap: 6px; }
        .link-item {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            line-height: 1.4;
            padding: 7px;
        }
        .link-item:hover { background: #eff6ff; border-color: #60a5fa; }
        .link-score { color: var(--muted); float: right; }
        .empty { color: var(--muted); font-size: 13px; }
        @media (max-width: 1100px) {
            .layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<header>
    <h1>Layout prediction · KEY / VALUE · page 3+</h1>
    <p>PP-OCR + VietOCR hiệu chỉnh → Layout model → row-link; value ở bên phải key · Chế độ chỉ đọc</p>
</header>
<main class="app">
    <div class="toolbar">
        <label for="pageSelect">Page:</label>
        <select id="pageSelect"></select>
        <button id="nextPageButton" type="button">Trang kế tiếp →</button>
        <span id="pagePosition"></span>
        <label><input id="showBoxes" type="checkbox" checked> Bbox</label>
        <label><input id="showLinks" type="checkbox" checked> Link</label>
    </div>
    <div class="layout">
        <section class="panel image-panel">
            <div id="stage">
                <img id="pageImage" alt="Trang PDF">
                <svg id="linkLayer"></svg>
            </div>
        </section>
        <aside class="panel">
            <h2 id="pageTitle">Kết quả</h2>
            <div class="summary" id="summary"></div>
            <div class="legend">
                <span class="key">KEY</span>
                <span class="value">VALUE</span>
                <span class="other">OTHER</span>
            </div>
            <div class="detail" id="regionDetail">Chọn một bbox để xem chi tiết.</div>
            <div class="link-section">
                <h3>KEY → VALUE</h3>
                <div class="link-list" id="keyValueLinks"></div>
            </div>
        </aside>
    </div>
</main>
<script>
const pages = __PAGES_JSON__;
const pageSelect = document.getElementById("pageSelect");
const nextPageButton = document.getElementById("nextPageButton");
const pagePosition = document.getElementById("pagePosition");
const pageTitle = document.getElementById("pageTitle");
const summary = document.getElementById("summary");
const stage = document.getElementById("stage");
const pageImage = document.getElementById("pageImage");
const linkLayer = document.getElementById("linkLayer");
const regionDetail = document.getElementById("regionDetail");
const showBoxes = document.getElementById("showBoxes");
const showLinks = document.getElementById("showLinks");
const regionMap = new Map();
let currentPage = 0;

function primaryLabel(region) {
    for (const label of ["KEY", "VALUE", "OTHER"]) {
        if (region.labels.includes(label)) return label;
    }
    return "OTHER";
}

function center(region) {
    return {
        x: (region.bbox[0] + region.bbox[2]) / 2,
        y: (region.bbox[1] + region.bbox[3]) / 2,
    };
}

function percent(value, total) {
    return `${(100 * value / total).toFixed(4)}%`;
}

function selectRegion(region) {
    document.querySelectorAll(".region-box.selected").forEach(box => box.classList.remove("selected"));
    const box = document.querySelector(`[data-region-id="${CSS.escape(region.id)}"]`);
    if (box) box.classList.add("selected");
    const scores = Object.entries(region.scores || {})
        .map(([label, score]) => `${label}=${score.toFixed(3)}`).join(", ");
    regionDetail.textContent = [
        `${region.id} · ${region.labels.join("/")}`,
        `Text: ${region.text}`,
        `BBox: [${region.bbox.join(", ")}]`,
        `OCR score: ${(region.ocr_score ?? 0).toFixed(3)}`,
        `Role scores: ${scores}`,
    ].join("\n");
}

function renderRegions(page) {
    document.querySelectorAll(".region-box").forEach(box => box.remove());
    regionMap.clear();
    for (const region of page.regions) {
        regionMap.set(region.id, region);
        const label = primaryLabel(region);
        const box = document.createElement("div");
        box.className = `region-box ${label.toLowerCase()}`;
        box.dataset.regionId = region.id;
        box.style.left = percent(region.bbox[0], page.image_width);
        box.style.top = percent(region.bbox[1], page.image_height);
        box.style.width = percent(region.bbox[2] - region.bbox[0], page.image_width);
        box.style.height = percent(region.bbox[3] - region.bbox[1], page.image_height);
        box.title = `${region.id} · ${region.text}`;
        const badge = document.createElement("span");
        badge.className = "region-label";
        badge.textContent = `${region.id} · ${label}`;
        box.appendChild(badge);
        box.addEventListener("click", () => selectRegion(region));
        stage.appendChild(box);
    }
    showBoxes.checked = true;
    showBoxes.dispatchEvent(new Event("change"));
}

function clearLinks() {
    while (linkLayer.firstChild) linkLayer.removeChild(linkLayer.firstChild);
}

function drawLine(firstId, secondId, color) {
    const first = regionMap.get(firstId);
    const second = regionMap.get(secondId);
    if (!first || !second) return;
    const firstCenter = center(first);
    const secondCenter = center(second);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", firstCenter.x);
    line.setAttribute("y1", firstCenter.y);
    line.setAttribute("x2", secondCenter.x);
    line.setAttribute("y2", secondCenter.y);
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "4");
    line.setAttribute("stroke-opacity", ".72");
    linkLayer.appendChild(line);
}

function drawLinks(page) {
    clearLinks();
    linkLayer.setAttribute("viewBox", `0 0 ${page.image_width} ${page.image_height}`);
    if (!showLinks.checked) return;
    for (const link of page.links || []) {
        drawLine(link.key_region_id, link.value_region_id, "#2563eb");
    }
}

function appendLinkItem(container, text, ids, score) {
    const item = document.createElement("div");
    item.className = "link-item";
    const scoreLabel = document.createElement("span");
    scoreLabel.className = "link-score";
    scoreLabel.textContent = score.toFixed(3);
    item.appendChild(scoreLabel);
    item.appendChild(document.createTextNode(text));
    item.addEventListener("click", () => {
        document.querySelectorAll(".region-box.selected").forEach(box => box.classList.remove("selected"));
        for (const id of ids) {
            const box = document.querySelector(`[data-region-id="${CSS.escape(id)}"]`);
            if (box) box.classList.add("selected");
        }
    });
    container.appendChild(item);
}

function renderLinks(page) {
    const keyValueContainer = document.getElementById("keyValueLinks");
    keyValueContainer.replaceChildren();
    for (const link of page.links || []) {
        appendLinkItem(
            keyValueContainer,
            `${link.key_text} [${link.key_region_id}] → ${link.value_text} [${link.value_region_id}]`,
            [link.key_region_id, link.value_region_id],
            link.score,
        );
    }
    if (!keyValueContainer.children.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "Không có kết quả.";
        keyValueContainer.appendChild(empty);
    }
}

function renderPage(index) {
    currentPage = index;
    const page = pages[index];
    pageSelect.value = String(index);
    nextPageButton.disabled = index >= pages.length - 1;
    pagePosition.textContent = `${index + 1}/${pages.length}`;
    pageTitle.textContent = `${page.pdf_name} · trang ${page.physical_page}`;
    summary.textContent = [
        `${page.regions.length} region · ${page.links.length} KEY → VALUE`,
        `OCR: ${page.recognition_source}`,
    ].join("\n");
    pageImage.src = page.image_path || page.image_name;
    stage.style.aspectRatio = `${page.image_width} / ${page.image_height}`;
    renderRegions(page);
    renderLinks(page);
    drawLinks(page);
    regionDetail.textContent = "Chọn một bbox để xem chi tiết.";
}

for (const [index, page] of pages.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${page.pdf_name} · trang ${page.physical_page}`;
    pageSelect.appendChild(option);
}
pageSelect.addEventListener("change", () => renderPage(Number(pageSelect.value)));
nextPageButton.addEventListener("click", () => {
    if (currentPage < pages.length - 1) renderPage(currentPage + 1);
});
showBoxes.addEventListener("change", () => {
    document.querySelectorAll(".region-box").forEach(box => {
        box.style.display = showBoxes.checked ? "block" : "none";
    });
});
showLinks.addEventListener("change", () => drawLinks(pages[currentPage]));
renderPage(0);
</script>
</body>
</html>
"""
    return template.replace("__PAGES_JSON__", pages_json)


def main():
    """Phục vụ predictions.html và ảnh bằng HTTP chỉ đọc."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--port", type=int, default=PORT)
    arguments = parser.parse_args()
    directory = arguments.output_directory.resolve()
    predictions_path = directory / "predictions.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {predictions_path}")
    dataset = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not dataset["pages"]:
        raise ValueError("predictions.json không có page để hiển thị.")
    html_path = directory / "predictions.html"
    html_path.write_text(build_html(dataset), encoding="utf-8")
    request_handler = partial(
        SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    server = ThreadingHTTPServer((HOST, arguments.port), request_handler)
    print(f"Giao diện: http://{HOST}:{arguments.port}/predictions.html")
    print(f"Dữ liệu: {predictions_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
