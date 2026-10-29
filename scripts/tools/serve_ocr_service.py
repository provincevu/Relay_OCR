# Mục đích: cung cấp giao diện web đơn giản cho OCRService.
# Chức năng: nhận PDF upload, chạy OCRService.process_pdf() và hiển thị từng phiếu.
# Luồng xử lý: HTTP upload → OCRService → metadata/ảnh → giao diện HTML.
# Cấu hình/parameter: file JSON config, thư mục output, host và port HTTP.
# Quan hệ: chỉ điều phối HTTP; không chứa logic OCR hoặc logic phân tách phiếu.
# Input/output: nhận multipart PDF; trả JSON kết quả và phục vụ ảnh đã render.
# Contract: kết quả hiển thị lấy từ OCRService.process_pdf().

import argparse
from email import policy
from email.parser import BytesParser
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from scr.ocr_service import OCRService


HOST = "127.0.0.1"
PORT = 8003


HTML = r"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>OCRService PDF Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #f1f5f9; color: #172033; font-family: Arial, sans-serif; margin: 0; }
        header { background: #0f172a; color: white; padding: 18px 24px; }
        header h1 { font-size: 22px; margin: 0 0 6px; }
        header p { color: #cbd5e1; margin: 0; }
        .app { margin: auto; max-width: 1900px; padding: 18px; }
        .toolbar, .panel { background: white; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; }
        .toolbar { align-items: center; display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
        .toolbar label { font-weight: 700; }
        .toolbar span { color: #64748b; font-size: 13px; }
        button, select, input { font: inherit; padding: 7px 9px; }
        button { cursor: pointer; }
        button:disabled { cursor: not-allowed; opacity: .5; }
        #status { color: #475569; }
        .layout { align-items: start; display: grid; gap: 16px; grid-template-columns: minmax(0, 1fr) 390px; }
        .stage { background: #e2e8f0; min-height: 260px; overflow: hidden; position: relative; }
        #pageImage { display: block; height: auto; user-select: none; width: 100%; }
        #linkLayer { height: 100%; left: 0; pointer-events: none; position: absolute; top: 0; width: 100%; }
        .region-box { border: 2px solid #64748b; cursor: pointer; overflow: visible; position: absolute; }
        .region-box:hover, .region-box.selected { box-shadow: 0 0 0 3px #facc15; z-index: 10; }
        .region-box.key { border-color: #2563eb; background: #2563eb2b; }
        .region-box.value { border-color: #16a34a; background: #16a34a2b; }
        .region-box.header { border-color: #ea580c; background: #ea580c2b; }
        .region-box.field-table { border-color: #0f766e; background: #0f766e2b; }
        .region-box.note { border-color: #a16207; background: #a162072b; }
        .region-box.page-number { border-color: #db2777; background: #db27772b; }
        .region-box.form-number { border-color: #0891b2; background: #0891b22b; }
        .region-box.inline { border-color: #7c3aed; background: #7c3aed2b; }
        .region-label { background: #64748b; color: white; font-size: 11px; font-weight: 700; left: -2px; line-height: 1; padding: 3px 4px; position: absolute; top: -18px; white-space: nowrap; }
        .key .region-label { background: #2563eb; }
        .value .region-label { background: #16a34a; }
        .header .region-label { background: #ea580c; }
        .field-table .region-label { background: #0f766e; }
        .note .region-label { background: #a16207; }
        .page-number .region-label { background: #db2777; }
        .form-number .region-label { background: #0891b2; }
        .inline .region-label { background: #7c3aed; }
        .page-title { font-size: 18px; margin: 0 0 8px; }
        .muted { color: #64748b; font-size: 13px; }
        .legend { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
        .legend span { border-radius: 5px; color: white; font-size: 11px; padding: 4px 6px; }
        .legend .key { background: #2563eb; }
        .legend .value { background: #16a34a; }
        .legend .header { background: #ea580c; }
        .legend .field-table { background: #0f766e; }
        .legend .note { background: #a16207; }
        .legend .page-number { background: #db2777; }
        .legend .form-number { background: #0891b2; }
        .legend .inline { background: #7c3aed; }
        .legend .other { background: #64748b; }
        .role-summary { color: #64748b; font-size: 12px; line-height: 1.5; margin-bottom: 12px; }
        .detail { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 7px; font-size: 12px; line-height: 1.5; margin-bottom: 12px; min-height: 45px; padding: 8px; white-space: pre-wrap; }
        .section { border-top: 1px solid #e2e8f0; margin-top: 14px; padding-top: 12px; }
        .section h3 { font-size: 15px; margin: 0 0 8px; }
        .field { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 6px; padding: 7px; }
        .region-item, .link-item { cursor: pointer; font-size: 12px; line-height: 1.4; }
        .region-item:hover, .link-item:hover { background: #eff6ff; border-color: #60a5fa; }
        .link-score { color: #64748b; float: right; }
        pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; max-height: 340px; overflow: auto; padding: 9px; white-space: pre-wrap; }
        .empty { color: #64748b; font-size: 13px; }
        @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<header>
    <h1>OCRService PDF Viewer</h1>
    <p>Upload PDF → tách phiếu → hiển thị page 1 và page 3+.</p>
</header>
<main class="app">
    <form class="toolbar" id="uploadForm">
        <input id="pdfInput" name="pdf" type="file" accept="application/pdf" required>
        <button type="submit">Xử lý PDF</button>
        <button id="previousDocument" type="button" disabled>← Phiếu trước</button>
        <button id="nextDocument" type="button" disabled>Phiếu sau →</button>
        <span id="documentPosition"></span>
        <span id="status">Chưa có PDF.</span>
    </form>
    <div class="toolbar">
        <label for="pageSelect">Ảnh trong phiếu:</label>
        <select id="pageSelect" disabled></select>
        <button id="previousPage" type="button" disabled>← Page trước</button>
        <button id="nextPage" type="button" disabled>Page sau →</button>
        <span id="pagePosition"></span>
        <label><input id="showBoxes" type="checkbox" checked> Bbox</label>
        <label><input id="showLinks" type="checkbox" checked> Link</label>
    </div>
    <div class="layout">
        <section class="panel">
            <h2 class="page-title" id="pageTitle">Chưa có kết quả</h2>
            <div class="stage" id="stage">
                <img id="pageImage" alt="Ảnh page PDF">
                <svg id="linkLayer"></svg>
            </div>
        </section>
        <aside class="panel">
            <div class="muted" id="pageSummary"></div>
            <div class="legend">
                <span class="key">KEY</span>
                <span class="value">VALUE</span>
                <span class="header">HEADER</span>
                <span class="field-table">FIELD_TABLE</span>
                <span class="note">NOTE</span>
                <span class="page-number">PAGE_NUMBER</span>
                <span class="form-number">FORM_NUMBER</span>
                <span class="inline">INLINE_KEY_VALUE</span>
                <span class="other">OTHER</span>
            </div>
            <div class="role-summary" id="roleSummary"></div>
            <div class="detail" id="regionDetail">Chọn một bbox để xem chi tiết.</div>
            <div class="section">
                <h3>Thông tin trích xuất</h3>
                <div id="fields"></div>
            </div>
            <div class="section">
                <h3>Region OCR / prediction</h3>
                <div id="regions"></div>
            </div>
            <div class="section">
                <h3>Link KEY → VALUE</h3>
                <div id="links"></div>
            </div>
            <div class="section">
                <h3>JSON kết quả page</h3>
                <pre id="pageJson">Chưa có dữ liệu.</pre>
            </div>
        </aside>
    </div>
</main>
<script>
const form = document.getElementById("uploadForm");
const input = document.getElementById("pdfInput");
const statusText = document.getElementById("status");
const previousDocument = document.getElementById("previousDocument");
const nextDocument = document.getElementById("nextDocument");
const documentPosition = document.getElementById("documentPosition");
const pageSelect = document.getElementById("pageSelect");
const previousPage = document.getElementById("previousPage");
const nextPage = document.getElementById("nextPage");
const pagePosition = document.getElementById("pagePosition");
const pageTitle = document.getElementById("pageTitle");
const pageImage = document.getElementById("pageImage");
const stage = document.getElementById("stage");
const linkLayer = document.getElementById("linkLayer");
const pageSummary = document.getElementById("pageSummary");
const roleSummary = document.getElementById("roleSummary");
const regionDetail = document.getElementById("regionDetail");
const fields = document.getElementById("fields");
const regions = document.getElementById("regions");
const links = document.getElementById("links");
const pageJson = document.getElementById("pageJson");
const showBoxes = document.getElementById("showBoxes");
const showLinks = document.getElementById("showLinks");
const regionMap = new Map();
const regionBoxMap = new Map();
let result = null;
let documentIndex = 0;
let pageIndex = 0;

function clear(element) {
    element.replaceChildren();
}

function showValue(value) {
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function renderDocument() {
    const formDocument = result.documents[documentIndex];
    documentPosition.textContent = `Phiếu ${documentIndex + 1}/${result.documents.length}`;
    previousDocument.disabled = documentIndex === 0;
    nextDocument.disabled = documentIndex === result.documents.length - 1;
    pageIndex = 0;
    pageSelect.replaceChildren();
    formDocument.pages.forEach((page, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = `${page.document_page_number} · ${page.page_type}`;
        pageSelect.appendChild(option);
    });
    pageSelect.disabled = formDocument.pages.length === 0;
    renderPage(pageIndex);
}

function renderPage(index) {
    const formDocument = result.documents[documentIndex];
    if (!formDocument.pages.length) return;
    pageIndex = Math.max(0, Math.min(index, formDocument.pages.length - 1));
    const page = formDocument.pages[pageIndex];
    if (!page) return;
    pageSelect.value = pageIndex;
    pagePosition.textContent = `${pageIndex + 1}/${formDocument.pages.length}`;
    previousPage.disabled = pageIndex === 0;
    nextPage.disabled = pageIndex === formDocument.pages.length - 1;
    pageTitle.textContent = `${formDocument.document_id} · ${page.page_type}`;
    const pageResult = page.result;
    const regionCount = pageResult?.regions?.length || 0;
    const linkCount = pageResult?.links?.length || 0;
    pageSummary.textContent = [
        `Page trong phiếu: ${page.document_page_number} · Page PDF gốc: ${page.source_pdf_page_number}`,
        `${regionCount} region · ${linkCount} link KEY → VALUE`,
    ].join("\n");
    if (pageResult?.image_width && pageResult?.image_height) {
        stage.style.aspectRatio = `${pageResult.image_width} / ${pageResult.image_height}`;
    } else {
        stage.style.removeProperty("aspect-ratio");
    }
    pageImage.src = page.image_url;
    renderRegions(pageResult);
    drawLinks(pageResult);
    renderRoleSummary(pageResult);
    regionDetail.textContent = "Chọn một bbox để xem chi tiết.";
    clear(fields);
    clear(regions);
    clear(links);
    pageJson.textContent = JSON.stringify(pageResult || {}, null, 2);
    renderFields(pageResult);
    renderRegionList(pageResult);
    renderLinks(pageResult);
}

function renderFields(pageResult) {
    if (!pageResult) {
        fields.innerHTML = '<div class="empty">Page này chưa có kết quả inference.</div>';
        return;
    }
    if (pageResult.fields) {
        Object.entries(pageResult.fields).forEach(([name, value]) => {
            const item = document.createElement("div");
            item.className = "field";
            item.textContent = `${name}: ${showValue(value)}`;
            fields.appendChild(item);
        });
        return;
    }
    fields.innerHTML = '<div class="empty">Page này không có field TABLE 01.</div>';
}

function primaryLabel(region) {
    const labels = region.labels || [];
    if (labels.includes("INLINE_KEY_VALUE")) return "INLINE_KEY_VALUE";
    if (labels.includes("KEY") && labels.includes("VALUE")) {
        return "INLINE_KEY_VALUE";
    }
    return labels[0] || "OTHER";
}

function roleClass(role) {
    if (role === "INLINE_KEY_VALUE") return "inline";
    return role.toLowerCase().replaceAll("_", "-");
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
    regionBoxMap.forEach((box) => box.classList.remove("selected"));
    const box = regionBoxMap.get(region.id);
    if (box) box.classList.add("selected");
    const scores = Object.entries(region.scores || {})
        .map(([label, score]) => `${label}=${Number(score).toFixed(3)}`)
        .join(", ");
    const bestRole = Object.entries(region.scores || {}).reduce(
        (best, current) => current[1] > best[1] ? current : best,
        ["Không có score", -1],
    )[0];
    regionDetail.textContent = [
        `${region.id}`,
        `Role: ${primaryLabel(region)}`,
        `Role model cao nhất: ${bestRole}`,
        `Text: ${region.text || ""}`,
        `BBox: [${(region.bbox || []).join(", ")}]`,
        `OCR score: ${Number(region.ocr_score ?? 0).toFixed(3)}`,
        `Role scores: ${scores || "Không có"}`,
    ].join("\n");
}

function renderRegions(pageResult) {
    stage.querySelectorAll(".region-box").forEach((box) => box.remove());
    regionMap.clear();
    regionBoxMap.clear();
    if (!pageResult || !pageResult.regions?.length) {
        return;
    }
    const width = pageResult.image_width;
    const height = pageResult.image_height;
    if (!width || !height) return;
    pageResult.regions.forEach((region) => {
        const bbox = region.bbox;
        if (!bbox || bbox.length < 4) return;
        const label = primaryLabel(region);
        // Không vẽ bbox cho các region bị phân loại là OTHER.
        if (label === "OTHER") return;
        const box = document.createElement("div");
        box.className = `region-box ${roleClass(label)}`;
        box.style.left = percent(bbox[0], width);
        box.style.top = percent(bbox[1], height);
        box.style.width = percent(bbox[2] - bbox[0], width);
        box.style.height = percent(bbox[3] - bbox[1], height);
        const roleScore = region.scores?.[label];
        const scoreText = typeof roleScore === "number"
            ? ` · ${(roleScore * 100).toFixed(1)}%`
            : "";
        const labelElement = document.createElement("span");
        labelElement.className = "region-label";
        labelElement.textContent = `${label}${scoreText}`;
        box.appendChild(labelElement);
        box.title = `${region.id} · ${label}${scoreText} · ${region.text || ""}`;
        box.addEventListener("click", () => selectRegion(region));
        stage.appendChild(box);
        regionMap.set(region.id, region);
        regionBoxMap.set(region.id, box);
    });
    updateBoxVisibility();
}

function renderRoleSummary(pageResult) {
    if (!pageResult?.regions?.length) {
        roleSummary.textContent = "Page này không có prediction role.";
        return;
    }
    const counts = {};
    pageResult.regions.forEach((region) => {
        const role = primaryLabel(region);
        counts[role] = (counts[role] || 0) + 1;
    });
    roleSummary.textContent = Object.entries(counts)
        .sort(([first], [second]) => first.localeCompare(second))
        .map(([role, count]) => `${role}: ${count}`)
        .join(" · ");
}

function updateBoxVisibility() {
    regionBoxMap.forEach((box) => {
        box.style.display = showBoxes.checked ? "block" : "none";
    });
}

function clearLinks() {
    while (linkLayer.firstChild) linkLayer.removeChild(linkLayer.firstChild);
}

function drawLinks(pageResult) {
    clearLinks();
    if (!pageResult?.image_width || !pageResult?.image_height) return;
    linkLayer.setAttribute(
        "viewBox",
        `0 0 ${pageResult.image_width} ${pageResult.image_height}`,
    );
    if (!showLinks.checked) return;
    for (const link of pageResult.links || []) {
        const first = regionMap.get(link.key_region_id);
        const second = regionMap.get(link.value_region_id);
        if (!first || !second || first.id === second.id) continue;
        const firstCenter = center(first);
        const secondCenter = center(second);
        const line = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "line",
        );
        line.setAttribute("x1", firstCenter.x);
        line.setAttribute("y1", firstCenter.y);
        line.setAttribute("x2", secondCenter.x);
        line.setAttribute("y2", secondCenter.y);
        line.setAttribute("stroke", "#2563eb");
        line.setAttribute("stroke-width", "4");
        line.setAttribute("stroke-opacity", ".72");
        linkLayer.appendChild(line);
    }
}

function renderRegionList(pageResult) {
    if (!pageResult || !pageResult.regions?.length) {
        regions.innerHTML = '<div class="empty">Không có region prediction.</div>';
        return;
    }
    pageResult.regions.forEach((region) => {
        const item = document.createElement("div");
        item.className = "field region-item";
        item.textContent = `${region.id} · ${primaryLabel(region)}: ${region.text || ""}`;
        item.addEventListener("click", () => selectRegion(region));
        regions.appendChild(item);
    });
}

function renderLinks(pageResult) {
    if (!pageResult || !pageResult.links?.length) {
        links.innerHTML = '<div class="empty">Không có link KEY → VALUE.</div>';
        return;
    }
    pageResult.links.forEach((link) => {
        const item = document.createElement("div");
        item.className = "field link-item";
        const score = document.createElement("span");
        score.className = "link-score";
        score.textContent = typeof link.score === "number"
            ? link.score.toFixed(3)
            : "";
        item.appendChild(score);
        const isInline = link.key_region_id === link.value_region_id;
        item.appendChild(document.createTextNode(
            isInline
                ? `KEY/VALUE inline: ${link.key_text || link.key_region_id}`
                : `${link.key_text || link.key_region_id} → ${link.value_text || link.value_region_id}`,
        ));
        item.addEventListener("click", () => {
            regionBoxMap.forEach((box) => box.classList.remove("selected"));
            [link.key_region_id, link.value_region_id].forEach((id) => {
                const box = regionBoxMap.get(id);
                if (box) box.classList.add("selected");
            });
        });
        links.appendChild(item);
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.files.length) return;
    statusText.textContent = "Đang xử lý PDF...";
    const response = await fetch("/api/process", {
        method: "POST",
        body: new FormData(form),
    });
    result = await response.json();
    if (!response.ok) {
        statusText.textContent = result.error || "Xử lý PDF thất bại.";
        return;
    }
    documentIndex = 0;
    statusText.textContent = `Đã xử lý ${result.documents.length} phiếu.`;
    renderDocument();
});

previousDocument.addEventListener("click", () => {
    if (documentIndex > 0) {
        documentIndex -= 1;
        renderDocument();
    }
});

nextDocument.addEventListener("click", () => {
    if (documentIndex < result.documents.length - 1) {
        documentIndex += 1;
        renderDocument();
    }
});

pageSelect.addEventListener("change", () => renderPage(Number(pageSelect.value)));
previousPage.addEventListener("click", () => {
    if (pageIndex > 0) renderPage(pageIndex - 1);
});
nextPage.addEventListener("click", () => {
    const pages = result?.documents?.[documentIndex]?.pages || [];
    if (pageIndex < pages.length - 1) renderPage(pageIndex + 1);
});
showBoxes.addEventListener("change", updateBoxVisibility);
showLinks.addEventListener("change", () => {
    const page = result?.documents?.[documentIndex]?.pages?.[pageIndex];
    drawLinks(page?.result);
});
document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea, button")) return;
    if (event.key === "ArrowLeft" && pageIndex > 0) {
        renderPage(pageIndex - 1);
    }
    if (event.key === "ArrowRight") {
        const pages = result?.documents?.[documentIndex]?.pages || [];
        if (pageIndex < pages.length - 1) renderPage(pageIndex + 1);
    }
});
</script>
</body>
</html>"""


def build_result(job_id, output_directory, process_result):
    """Gắn ảnh và metadata page vào kết quả OCRService."""
    # image_path trong metadata được tính tương đối từ output_directory.
    # output_directory của web server nằm tại <output_root>/<job_id>/splitted_pdf,
    # nên URL assets phải giữ cả thư mục splitted_pdf.
    asset_directory = Path(job_id) / output_directory.name
    result_documents = {
        document["document_id"]: document
        for document in process_result["documents"]
    }
    documents = []

    for document_id, process_document in result_documents.items():
        metadata_path = output_directory / document_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        page3_by_physical_page = {
            page["physical_page"]: page
            for page in process_document["page3_plus"]
        }
        pages = []
        for page in metadata["pages"]:
            page_result = None
            if page["page_type"] == "PAGE_1":
                page_result = process_document["page1"]
            elif page["page_type"] == "PAGE_3_PLUS":
                page_result = page3_by_physical_page.get(
                    page["source_pdf_page_number"],
                )
            pages.append(
                {
                    **page,
                    "image_url": (
                        f"/assets/{asset_directory.as_posix()}/{page['image_path']}"
                    ),
                    "result": page_result,
                },
            )
        documents.append(
            {
                "document_id": document_id,
                "source_page_start": process_document["source_page_start"],
                "source_page_end": process_document["source_page_end"],
                "pages": pages,
            },
        )

    return {
        "source_pdf": process_result["source_pdf"],
        "documents": documents,
    }


class OCRWebServer(ThreadingHTTPServer):
    """HTTP server giữ config OCR và thư mục ảnh của các job."""

    def __init__(self, address, handler, output_root, config):
        super().__init__(address, handler)
        self.output_root = Path(output_root).resolve()
        self.ocr_config = config


class OCRRequestHandler(SimpleHTTPRequestHandler):
    """Phục vụ HTML, upload PDF và ảnh kết quả OCR."""

    def do_GET(self):
        """Trả giao diện hoặc ảnh đã render."""
        if urlparse(self.path).path == "/":
            self._send_html()
            return
        if urlparse(self.path).path.startswith("/assets/"):
            super().do_GET()
            return
        self.send_error(404)

    def do_POST(self):
        """Lưu PDF upload và gọi OCRService.process_pdf()."""
        if urlparse(self.path).path != "/api/process":
            self.send_error(404)
            return

        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + self.headers["Content-Type"].encode()
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body,
        )
        file_part = next(
            part for part in message.walk()
            if part.get_filename()
        )
        job_id = uuid4().hex
        job_directory = self.server.output_root / job_id
        job_directory.mkdir(parents=True, exist_ok=True)
        source_pdf = job_directory / "source.pdf"
        source_pdf.write_bytes(file_part.get_payload(decode=True))

        output_directory = job_directory / "splitted_pdf"
        config = dict(self.server.ocr_config)
        config["input_loader_output"] = str(output_directory)
        process_result = OCRService(config).process_pdf(str(source_pdf))
        result = build_result(job_id, output_directory, process_result)
        self._send_json(result)

    def translate_path(self, path):
        """Ánh xạ URL assets vào thư mục output của server."""
        if path.startswith("/assets/"):
            relative_path = unquote(urlparse(path).path[len("/assets/"):])
            return str(self.server.output_root / Path(relative_path))
        return super().translate_path(path)

    def _send_html(self):
        """Trả HTML giao diện OCR."""
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload):
        """Trả JSON kết quả OCR."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_arguments():
    """Đọc tham số chạy web server."""
    parser = argparse.ArgumentParser(
        description="Phục vụ giao diện OCRService cho file PDF.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/ocr_web"))
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def main():
    """Khởi động web server OCRService."""
    arguments = parse_arguments()
    config = (
        json.loads(arguments.config.read_text(encoding="utf-8"))
        if arguments.config
        else {}
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    server = OCRWebServer(
        (arguments.host, arguments.port),
        OCRRequestHandler,
        arguments.output,
        config,
    )
    print(f"Giao diện: http://{arguments.host}:{arguments.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
