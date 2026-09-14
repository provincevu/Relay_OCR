// Mở rộng giao diện gốc trước khi gắn listener; giữ các hàm suy luận hiện có.
const original = {
    reset: resetAll, load: loadPage, styles: refreshBoxStyles,
    logical: renderLogicalBlocks, links: renderLinks, pair: pairHtml,
    buttons: updateButtons, inline: confirmInlineSelection,
};
const annotation = {
    selected: new Set(), groups: [], labels: {}, history: [],
    mode: "sample", sessions: new Map(), activePage: -1,
    saved: {}, editing: null, dirty: false, frozenPairs: [],
};
const annotationStyle = document.createElement("style");
annotationStyle.textContent = `
    .annotation-toolbar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:10px}
    .annotation-toolbar select{padding:8px;font:inherit;border-radius:7px}
    .annotation-toolbar button{padding:8px 10px}
    .annotation-toolbar button:disabled{opacity:.45;cursor:not-allowed}
    #groupSummary{width:100%;font-size:14px;color:#334155;line-height:1.5}
    .ocr-box.annotation-selected{outline:4px solid #eab308;outline-offset:3px}
    .ocr-box.annotation-code{border-color:#dc2626;background:#dc26262b}
    .ocr-box.annotation-header{border-color:#a16207;background:#a162072b}
    .ocr-box.annotation-unlabeled{border-color:#475569;border-style:dashed;background:#47556945}
    .ocr-box.annotation-other{border-color:#94a3b8;background:#94a3b81a}
    .annotation-drag{position:absolute;border:3px dashed #eab308;background:#facc1526;
        pointer-events:none;z-index:2147483647;display:none}
    .logical-box.annotation-code{border-color:#dc2626}
    .logical-box.annotation-code span{background:#dc2626}
    .steps{display:none}.instruction{font-size:14px;padding:8px;margin-bottom:6px}
    .control-panel{padding:12px}.side-panel{top:320px}
    header{padding:14px 24px}.actions{margin-top:6px}
`;
document.head.appendChild(annotationStyle);
document.title = "Annotation CODE · KEY · VALUE";
document.querySelector("header h1").textContent = "Annotation page 3 · CODE / KEY / VALUE";
document.querySelector("header p").textContent = "Rule Induction từ mẫu; lưu annotation theo ID region.";
const annotationToolbar = document.createElement("div");
annotationToolbar.className = "annotation-toolbar";
annotationToolbar.innerHTML = `
    <label>Chế độ <select id="annotationMode">
        <option value="sample">Chọn nhóm cho mẫu</option>
        <option value="inspect">Kiểm tra / sửa cặp</option>
        <option value="UNLABELED">Đánh dấu UNLABELED</option>
        <option value="HEADER">Đánh dấu HEADER</option>
        <option value="OTHER">Đánh dấu OTHER</option>
    </select></label>
    <button id="commitGroup" class="secondary">Chốt nhóm 1</button>
    <button id="inlineGroup" class="secondary">Key-value trong cùng bbox</button>
    <button id="finishPair" class="confirm">Xác nhận mẫu key-value</button>
    <button id="breakRule" class="secondary">Ngắt rule / Bảng mới</button>
    <button id="applyAnnotation" class="secondary">Gán nhãn</button>
    <button id="clearAnnotation" class="secondary">Gỡ nhãn đã gán</button>
    <button id="clearSelection" class="secondary">Hủy chọn</button>
    <button id="undoAnnotation" class="secondary">Hoàn tác annotation</button>
    <button id="editConfirmed" class="secondary">Sửa phiếu đã xác nhận</button>
    <div id="groupSummary" role="status" aria-live="polite"></div>
`;
document.querySelector(".control-panel").appendChild(annotationToolbar);
const modeSelect = document.getElementById("annotationMode");
const commitButton = document.getElementById("commitGroup");
const inlineGroupButton = document.getElementById("inlineGroup");
const finishPairButton = document.getElementById("finishPair");
const breakRuleButton = document.getElementById("breakRule");
const applyButton = document.getElementById("applyAnnotation");
const clearAnnotationButton = document.getElementById("clearAnnotation");
const undoAnnotationButton = document.getElementById("undoAnnotation");
cancelKeyButton.textContent = "Hủy mẫu đang chọn";
const dragBox = document.createElement("div");
dragBox.className = "annotation-drag";
document.getElementById("imageStage").appendChild(dragBox);

function allPairIds(pair) {
    return [...(pair.code?.regionIds || []), ...pair.key.regionIds, ...pair.value.regionIds];
}

function allPairs() {
    return [...annotation.frozenPairs, ...state.examples, ...state.predictions];
}

function pairSource(pair) {
    return pair.source || (state.examples.includes(pair) ? "manual" : "rule");
}

function snapshotAnnotation() {
    return structuredClone({
        examples: state.examples, predictions: state.predictions,
        frozenPairs: annotation.frozenPairs,
        labels: annotation.labels, groups: annotation.groups,
        selected: [...annotation.selected], suppressed: [...state.suppressedPairIds],
        suppressionHistory: state.suppressionHistory, confirmed: state.confirmed,
        autoHeaders: state.autoHeaders, headerProbe: state.headerProbe,
        editing: annotation.editing, dirty: annotation.dirty,
    });
}

function restoreAnnotation(snapshot) {
    snapshot = structuredClone(snapshot);
    state.examples = snapshot.examples;
    state.predictions = snapshot.predictions;
    state.suppressedPairIds = new Set(snapshot.suppressed);
    state.suppressionHistory = snapshot.suppressionHistory;
    state.confirmed = snapshot.confirmed;
    state.autoHeaders = snapshot.autoHeaders;
    state.headerProbe = snapshot.headerProbe;
    annotation.labels = snapshot.labels;
    annotation.frozenPairs = snapshot.frozenPairs || [];
    annotation.groups = snapshot.groups;
    annotation.selected = new Set(snapshot.selected);
    annotation.editing = snapshot.editing;
    annotation.dirty = snapshot.dirty;
    recomputePreview();
}

function rememberAnnotation() {
    annotation.history.push(snapshotAnnotation());
    annotation.dirty = true;
}

function visibleAnnotations(finalize = false) {
    const labels = structuredClone(annotation.labels);
    for (const pair of allPairs()) {
        for (const role of ["CODE", "KEY", "VALUE"]) {
            for (const id of pair[role.toLowerCase()]?.regionIds || []) {
                labels[id] = [...new Set([...(labels[id] || []), role])];
            }
        }
    }
    for (const pair of [...state.autoHeaders, ...(state.headerProbe ? [state.headerProbe] : [])]) {
        for (const id of allPairIds(pair)) {
            if (!labels[id]) labels[id] = ["HEADER"];
        }
    }
    if (finalize) {
        for (const region of regions) labels[region.id] ||= ["OTHER"];
    }
    return labels;
}

updateButtons = function () {
    original.buttons();
    const selectedCount = annotation.selected.size;
    const groupCount = annotation.groups.length;
    const isSample = annotation.mode === "sample";
    commitButton.disabled = state.confirmed || !isSample || !selectedCount;
    commitButton.textContent = `Chốt nhóm ${groupCount + 1} (hoặc click lại bbox)`;
    inlineGroupButton.disabled = state.confirmed || !isSample || groupCount > 0 || selectedCount !== 1;
    finishPairButton.disabled = state.confirmed || !isSample || groupCount !== 2 || selectedCount > 0;
    breakRuleButton.disabled = state.confirmed || !state.examples.length || groupCount > 0
        || selectedCount > 0 || Boolean(annotation.editing);
    applyButton.disabled = state.confirmed || isSample || annotation.mode === "inspect" || !selectedCount;
    applyButton.textContent = `Gán ${annotation.mode === "sample" || annotation.mode === "inspect" ? "nhãn" : annotation.mode}`;
    clearAnnotationButton.disabled = state.confirmed || !selectedCount;
    undoAnnotationButton.disabled = !annotation.history.length || state.confirmed;
    document.getElementById("clearSelection").disabled = state.confirmed || !selectedCount;
    document.getElementById("editConfirmed").disabled = !state.confirmed;
    cancelKeyButton.disabled = state.confirmed || (!groupCount && !selectedCount && !annotation.editing);
    inlineCurrentKeyButton.disabled = state.confirmed || !isSample || groupCount > 0 || selectedCount !== 1;
    state.currentKeyId = selectedCount === 1 ? [...annotation.selected][0] : null;
    confirmResultButton.disabled = state.confirmed || groupCount > 0 || selectedCount > 0 || Boolean(annotation.editing);
    resetAllButton.disabled = state.confirmed;
    const groups = annotation.groups.map((group, index) => `Nhóm ${index + 1}: ${group.regionIds.join(", ")}`).join(" · ");
    const unlabeledCount = Object.values(annotation.labels).filter(labels => labels.includes("UNLABELED")).length;
    document.getElementById("groupSummary").textContent =
        `${selectedCount} region đang chọn · ${unlabeledCount} UNLABELED chủ động · `
        + `${annotation.frozenPairs.length} cặp đã ngắt rule. ${groups}`;
};

updateSteps = function () {
    selectionStatus.textContent = state.confirmed
        ? "Đã xác nhận và lưu phiếu"
        : `${annotation.groups.length} nhóm đã chốt · ${state.examples.length} mẫu · `
        + `${state.predictions.length} suy luận · ${annotation.frozenPairs.length} cặp đã ngắt rule`;
};

refreshBoxStyles = function () {
    original.styles();
    const labels = visibleAnnotations();
    const groupIds = new Set(annotation.groups.flatMap(group => group.regionIds));
    for (const box of boxLayer.children) {
        const id = box.dataset.regionId;
        const roles = labels[id] || [];
        for (const role of ["CODE", "KEY", "VALUE", "HEADER", "UNLABELED", "OTHER"]) {
            box.classList.toggle(`annotation-${role.toLowerCase()}`, roles.includes(role));
        }
        box.classList.toggle("annotation-selected", annotation.selected.has(id) || groupIds.has(id));
        if (roles.length || annotation.selected.has(id) || groupIds.has(id)) {
            box.classList.remove("hidden-unused");
        }
        box.title = `${id} · ${roles.join("/") || "Chưa gán"} · ${regionById.get(id)?.text || ""}`;
    }
};

renderLogicalBlocks = function () {
    original.logical();
    for (const pair of annotation.frozenPairs) {
        appendLogicalBox(pair.key, "manual-key", "KEY đã ngắt rule");
        appendLogicalBox(pair.value, "manual-value", "VALUE đã ngắt rule");
    }
    for (const pair of allPairs()) {
        if (pair.code) appendLogicalBox(pair.code, "annotation-code", "CODE gộp");
    }
};

renderLinks = function () {
    original.links();
    for (const pair of annotation.frozenPairs) {
        svgLine(pair, "manual-link", "#15803d", "");
    }
    for (const pair of allPairs()) {
        if (pair.code) svgLine({key: pair.code, value: pair.key}, "code-link", "#dc2626", "");
    }
};

pairHtml = function (pair, includeScore, allowRemoval = false) {
    const item = original.pair(pair, includeScore, allowRemoval);
    if (pair.code) {
        const code = document.createElement("div");
        code.textContent = `CODE: ${pair.code.text} (${pair.code.regionIds.join(", ")})`;
        code.style.color = "#dc2626";
        item.prepend(code);
    }
    const ids = document.createElement("div");
    ids.textContent = `${pair.key.regionIds.join(", ")} → ${pair.value.regionIds.join(", ")}`;
    item.appendChild(ids);
    const edit = document.createElement("button");
    edit.className = "secondary";
    edit.textContent = "Sửa nhóm";
    edit.disabled = state.confirmed;
    edit.addEventListener("click", () => editPrediction(pair));
    item.appendChild(edit);
    if (!includeScore) {
        const remove = document.createElement("button");
        remove.className = "remove-pair";
        remove.textContent = "Xóa mẫu";
        remove.disabled = state.confirmed;
        remove.addEventListener("click", () => {
            rememberAnnotation();
            state.examples = state.examples.filter(example => example !== pair);
            recomputePreview();
        });
        item.appendChild(remove);
    }
    return item;
};

// Chạy đúng các hàm suy luận gốc cho từng họ mẫu 2 hoặc 3 nhóm.
recomputePreview = function () {
    if (!state.confirmed) {
        const allRegions = pages[currentPageIndex].regions;
        const examples = state.examples;
        const frozenIds = new Set(annotation.frozenPairs.flatMap(allPairIds));
        const predictions = [];
        const headers = [];
        const suspected = new Set();
        let probe = null;
        state.rule = null;
        state.inlineRule = null;
        for (const hasCode of [false, true]) {
            const family = examples.filter(pair => Boolean(pair.code) === hasCode);
            if (!family.length) continue;
            const outside = new Set(examples.filter(pair => !family.includes(pair)).flatMap(allPairIds));
            const codes = new Set(examples.flatMap(pair => pair.code?.regionIds || []));
            regions = allRegions.filter(region => !annotation.labels[region.id] && !frozenIds.has(region.id)
                && !outside.has(region.id) && !codes.has(region.id));
            state.examples = family;
            state.rule = inferRule();
            state.inlineRule = inferInlineRule();
            state.predictions = [];
            state.autoHeaders = [];
            state.headerProbe = null;
            state.suspectedHeaderIds = new Set();
            if (state.rule) classifyPredictions(state.rule);
            const familyHeaders = [...state.autoHeaders];
            const familyProbe = state.headerProbe;
            const occupied = new Set(examples.flatMap(allPairIds));
            for (const pair of state.predictions) allPairIds(pair).forEach(id => occupied.add(id));
            state.predictions.push(...predictInlinePairs(state.inlineRule, occupied));
            const familyPredictions = state.predictions;
            if (hasCode) {
                regions = allRegions.filter(region => !annotation.labels[region.id]
                    && !frozenIds.has(region.id) && !outside.has(region.id));
                state.examples = family.map(pair => ({key: pair.code, value: pair.key, mode: "spatial"}));
                const codeRule = inferRule();
                const codeCandidates = predictPairs(codeRule).filter(pair =>
                    !examples.some(example => allPairIds(example).some(id => pair.key.regionIds.includes(id)))
                );
                for (const pair of familyPredictions) {
                    const candidate = codeCandidates.find(codePair =>
                        codePair.value.regionIds.join("+") === pair.key.regionIds.join("+")
                        && !codePair.key.regionIds.some(id => pair.value.regionIds.includes(id))
                    );
                    if (candidate) {
                        pair.code = candidate.key;
                        pair.score = Math.min(pair.score, candidate.score);
                        predictions.push(pair);
                    }
                }
            } else {
                predictions.push(...familyPredictions.map(pair => ({...pair, code: null})));
            }
            headers.push(...familyHeaders);
            probe ||= familyProbe;
            state.suspectedHeaderIds.forEach(id => suspected.add(id));
        }
        regions = allRegions;
        state.examples = examples;
        state.autoHeaders = headers;
        state.headerProbe = probe;
        state.suspectedHeaderIds = suspected;
        const occupied = new Set([
            ...annotation.frozenPairs,
            ...examples,
        ].flatMap(allPairIds));
        const codedPairs = new Set(predictions.filter(pair => pair.code).map(pairInstanceId));
        state.predictions = predictions.filter(pair => pair.code || !codedPairs.has(pairInstanceId(pair)))
            .sort((a, b) => b.score - a.score).filter(pair => {
            const ids = allPairIds(pair);
            if (ids.some(id => occupied.has(id) || annotation.labels[id])) return false;
            ids.forEach(id => occupied.add(id));
            return true;
        }).sort((a, b) => pairY(a) - pairY(b));
    }
    refreshBoxStyles(); renderLogicalBlocks(); renderLinks(); renderLists();
    renderRuleSummary(); updateSteps(); updateButtons();
    resultSection.hidden = !state.confirmed;
};

function cancelGroupSelection() {
    annotation.selected.clear();
    state.currentKeyId = null;
    lastSampleTapId = null;
    refreshBoxStyles(); updateButtons();
}

let lastSampleTapId = null;
selectRegion = function (id) {
    if (state.confirmed || suppressClick) return;
    if (annotation.mode === "inspect") {
        const pair = [...state.examples, ...state.predictions].find(pair => allPairIds(pair).includes(id));
        if (pair) openPredictionPopover(pair);
        return;
    }
    if (annotation.mode === "sample" && lastSampleTapId === id
        && annotation.selected.has(id)) {
        lastSampleTapId = null;
        commitButton.click();
        return;
    }
    lastSampleTapId = id;
    if (annotation.selected.has(id)) annotation.selected.delete(id);
    else annotation.selected.add(id);
    refreshBoxStyles(); updateButtons();
};

function finishGroups() {
    const groups = annotation.groups;
    const hasCode = groups.length === 3;
    const key = groups[hasCode ? 1 : 0];
    const value = groups[hasCode ? 2 : 1];
    let pair = {code: hasCode ? groups[0] : null, key, value, mode: "spatial"};
    if (key.regionIds.length === 1 && value.regionIds.length === 1) {
        const allRegions = regions;
        const excluded = new Set([
            ...(pair.code?.regionIds || []),
            ...state.examples.flatMap(example => example.code?.regionIds || []),
        ]);
        regions = regions.filter(region => !annotation.labels[region.id] && !excluded.has(region.id));
        const grouped = buildExample(key.regionIds[0], value.regionIds[0]);
        regions = allRegions;
        if (grouped) pair = {...grouped, code: pair.code};
    }
    if (annotation.editing) {
        state.suppressedPairIds.delete(pairInstanceId(annotation.editing.pair));
        annotation.editing = null;
    }
    state.examples.push(pair);
    annotation.groups = [];
    annotation.selected.clear();
    recomputePreview();
    instruction.textContent = hasCode
        ? "Đã tự xác nhận mẫu CODE → KEY → VALUE và cập nhật preview."
        : "Đã xác nhận mẫu KEY → VALUE; CODE = null. Preview đã cập nhật.";
}

commitButton.addEventListener("click", () => {
    const ids = [...annotation.selected];
    const used = new Set([...allPairs().flatMap(allPairIds), ...annotation.groups.flatMap(group => group.regionIds)]);
    if (ids.some(id => used.has(id) || annotation.labels[id])) {
        instruction.textContent = "Vùng đã thuộc mẫu hoặc có nhãn. Hãy sửa mẫu/gỡ nhãn trước khi dùng lại.";
        return;
    }
    rememberAnnotation();
    annotation.groups.push(combineRegions(ids.map(id => regionById.get(id))));
    annotation.selected.clear();
    if (annotation.groups.length === 3) finishGroups();
    else {
        instruction.textContent = annotation.groups.length === 2
            ? "Bấm Xác nhận mẫu key-value để dùng 2 nhóm; hoặc chọn nhóm 3 để tạo CODE → KEY → VALUE."
            : "Đã chốt nhóm 1. Chọn các region của nhóm 2 rồi bấm Chốt nhóm.";
        refreshBoxStyles(); updateSteps(); updateButtons();
    }
});
finishPairButton.addEventListener("click", () => { rememberAnnotation(); finishGroups(); });
breakRuleButton.addEventListener("click", () => {
    rememberAnnotation();
    const frozenIds = new Set([...state.examples, ...state.predictions].flatMap(allPairIds));
    for (const pair of [...state.autoHeaders, ...(state.headerProbe ? [state.headerProbe] : [])]) {
        for (const id of allPairIds(pair)) {
            if (!frozenIds.has(id) && !annotation.labels[id]) annotation.labels[id] = ["HEADER"];
        }
    }
    annotation.frozenPairs.push(
        ...state.examples.map(pair => ({...structuredClone(pair), source: "manual"})),
        ...state.predictions.map(pair => ({...structuredClone(pair), source: "rule"})),
    );
    state.examples = [];
    state.predictions = [];
    state.suppressedPairIds = new Set();
    state.suppressionHistory = [];
    annotation.groups = [];
    cancelGroupSelection();
    recomputePreview();
    instruction.textContent = "Đã ngắt rule hiện tại. Chọn mẫu của bảng mới để suy luận độc lập.";
});
inlineGroupButton.addEventListener("click", () => {
    if (inlineGroupButton.disabled) return;
    const id = [...annotation.selected][0];
    state.currentKeyId = id;
    openInlineEditor(id);
});

editPrediction = function (pair) {
    if (annotation.groups.length || annotation.selected.size || annotation.editing) {
        instruction.textContent = "Hãy hoàn tất hoặc hủy mẫu đang chọn trước khi sửa cặp khác.";
        return;
    }
    rememberAnnotation();
    const manual = state.examples.includes(pair);
    annotation.editing = {pair: structuredClone(pair), manual};
    state.examples = state.examples.filter(example => example !== pair);
    state.suppressedPairIds.add(pairInstanceId(pair));
    annotation.mode = "sample";
    modeSelect.value = "sample";
    closePredictionPopover(); recomputePreview();
    instruction.textContent = "Đang sửa: chọn lại đủ 2 hoặc 3 nhóm. Hủy mẫu đang chọn sẽ khôi phục cặp cũ.";
};

cancelCurrentKey = function () {
    if (annotation.editing) {
        const {pair, manual} = annotation.editing;
        state.suppressedPairIds.delete(pairInstanceId(pair));
        if (manual) state.examples.push(pair);
    }
    annotation.editing = null;
    annotation.groups = [];
    cancelGroupSelection(); recomputePreview();
    instruction.textContent = "Đã hủy mẫu đang chọn. Click hoặc kéo khung để chọn nhóm mới.";
};

const originalSuppress = suppressPrediction;
suppressPrediction = function (pair) {
    rememberAnnotation();
    state.examples = state.examples.filter(example => example !== pair);
    originalSuppress(pair);
};
const originalUndoExample = undoLastExample;
undoLastExample = function () { rememberAnnotation(); originalUndoExample(); };
confirmInlineSelection = function () {
    if (!inlineKey.value.trim() || !inlineValue.value.trim()) {
        inlineError.textContent = "Key và value đều phải có nội dung.";
        return;
    }
    const id = state.inlineRegionId;
    if (annotation.labels[id] || state.examples.some(pair => allPairIds(pair).includes(id))) {
        inlineError.textContent = "Region đã có nhãn hoặc thuộc mẫu; hãy gỡ nhãn/sửa mẫu trước.";
        return;
    }
    rememberAnnotation();
    if (annotation.editing) {
        state.suppressedPairIds.delete(pairInstanceId(annotation.editing.pair));
        annotation.editing = null;
    }
    original.inline();
    state.examples.at(-1).code = null;
    annotation.selected.clear(); annotation.groups = [];
    recomputePreview();
};

function applySelectedLabel() {
    const ids = new Set(annotation.selected);
    const affected = allPairs().filter(pair => allPairIds(pair).some(id => ids.has(id)));
    const annotated = visibleAnnotations();
    const existingCount = [...ids].filter(id => annotated[id]).length;
    if ((affected.length || existingCount) && !window.confirm(
        `Đổi ${ids.size} region sang ${annotation.mode}: ${existingCount} region đã có nhãn, `
        + `${affected.length} quan hệ sẽ bị gỡ. Các region còn lại của quan hệ bị gỡ sẽ thành UNLABELED. Tiếp tục?`
    )) return;
    rememberAnnotation();
    for (const pair of affected) {
        for (const id of allPairIds(pair)) annotation.labels[id] = ["UNLABELED"];
        state.suppressedPairIds.add(pairInstanceId(pair));
    }
    state.examples = state.examples.filter(pair => !affected.includes(pair));
    annotation.frozenPairs = annotation.frozenPairs.filter(pair => !affected.includes(pair));
    for (const id of ids) annotation.labels[id] = [annotation.mode];
    annotation.groups = [];
    cancelGroupSelection(); recomputePreview();
    instruction.textContent = `Đã gán ${annotation.mode} cho ${ids.size} region. Có thể hoàn tác annotation.`;
}
applyButton.addEventListener("click", applySelectedLabel);
clearAnnotationButton.addEventListener("click", () => {
    rememberAnnotation();
    for (const id of annotation.selected) delete annotation.labels[id];
    cancelGroupSelection(); recomputePreview();
});
document.getElementById("clearSelection").addEventListener("click", cancelGroupSelection);
undoAnnotationButton.addEventListener("click", () => {
    restoreAnnotation(annotation.history.pop());
    instruction.textContent = "Đã hoàn tác annotation và khôi phục các quan hệ liên quan.";
});
modeSelect.addEventListener("change", () => {
    cancelCurrentKey();
    annotation.mode = modeSelect.value;
    updateButtons();
    instruction.textContent = annotation.mode === "sample"
        ? "Click nhiều bbox hoặc kéo khung, rồi Chốt nhóm. Hai nhóm: xác nhận key-value; ba nhóm: tự xác nhận CODE/KEY/VALUE."
        : annotation.mode === "inspect"
            ? "Bấm bbox hoặc Sửa nhóm trong danh sách để kiểm tra cặp."
            : `Click hoặc kéo khung để chọn region; Shift + kéo để thêm vùng. Bấm Gán ${annotation.mode} để áp dụng.`;
});

let drag = null;
let suppressClick = false;
boxLayer.style.touchAction = "none";
function imagePoint(event) {
    const rect = boxLayer.getBoundingClientRect();
    return [
        Math.max(0, Math.min(PAGE_WIDTH, (event.clientX - rect.left) / rect.width * PAGE_WIDTH)),
        Math.max(0, Math.min(PAGE_HEIGHT, (event.clientY - rect.top) / rect.height * PAGE_HEIGHT)),
    ];
}
boxLayer.addEventListener("pointerdown", event => {
    if (state.confirmed || annotation.mode === "inspect" || event.button !== 0) return;
    drag = {start: imagePoint(event), x: event.clientX, y: event.clientY,
        id: event.target.closest(".ocr-box")?.dataset.regionId, add: event.shiftKey, moved: false};
    boxLayer.setPointerCapture(event.pointerId);
});
boxLayer.addEventListener("pointermove", event => {
    if (!drag) return;
    drag.moved ||= Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 5;
    if (!drag.moved) return;
    const end = imagePoint(event);
    const [left, top] = [Math.min(drag.start[0], end[0]), Math.min(drag.start[1], end[1])];
    dragBox.style.cssText += `;display:block;left:${left / PAGE_WIDTH * 100}%;top:${top / PAGE_HEIGHT * 100}%;`
        + `width:${Math.abs(end[0] - drag.start[0]) / PAGE_WIDTH * 100}%;height:${Math.abs(end[1] - drag.start[1]) / PAGE_HEIGHT * 100}%`;
});
boxLayer.addEventListener("pointerup", event => {
    if (!drag) return;
    if (drag.moved) {
        const end = imagePoint(event);
        if (!drag.add) annotation.selected.clear();
        for (const region of regions) {
            if (centerX(region) >= Math.min(drag.start[0], end[0]) && centerX(region) <= Math.max(drag.start[0], end[0])
                && centerY(region) >= Math.min(drag.start[1], end[1]) && centerY(region) <= Math.max(drag.start[1], end[1])) {
                annotation.selected.add(region.id);
            }
        }
        refreshBoxStyles(); updateButtons();
    } else if (drag.id) selectRegion(drag.id);
    suppressClick = true;
    setTimeout(() => { suppressClick = false; }, 0);
    drag = null;
    dragBox.style.display = "none";
});
boxLayer.addEventListener("pointercancel", () => { drag = null; dragBox.style.display = "none"; });

resetAll = function () {
    annotation.selected = new Set(); annotation.groups = []; annotation.labels = {};
    annotation.history = []; annotation.editing = null; annotation.frozenPairs = []; annotation.dirty = false;
    original.reset();
    instruction.textContent = "Chọn một hoặc nhiều bbox rồi Chốt nhóm. Hai nhóm: KEY/VALUE; ba nhóm: CODE/KEY/VALUE.";
};

function pairFromRecord(relation) {
    const key = combineRegions(relation.key_region_ids.map(id => regionById.get(id)));
    const value = combineRegions(relation.value_region_ids.map(id => regionById.get(id)));
    key.text = relation.key_text;
    value.text = relation.value_text;
    return {
        key, value,
        code: relation.code_region_ids === null ? null : combineRegions(relation.code_region_ids.map(id => regionById.get(id))),
        mode: relation.mode, separator: relation.separator, score: relation.rule_score,
        source: relation.source,
    };
}

loadPage = function (pageIndex) {
    if (annotation.activePage >= 0) {
        annotation.sessions.set(annotation.activePage, {
            snapshot: snapshotAnnotation(), history: annotation.history,
        });
    }
    original.load(pageIndex);
    annotation.activePage = pageIndex;
    const session = annotation.sessions.get(pageIndex);
    const saved = annotation.saved[pages[pageIndex].imageName];
    if (session) {
        annotation.history = session.history;
        restoreAnnotation(session.snapshot);
    } else if (saved) {
        if (saved.ocr_fingerprint !== pages[pageIndex].ocrFingerprint) {
            throw new Error("Annotation đã lưu thuộc snapshot OCR khác; không được ghép với ID hiện tại.");
        }
        annotation.labels = Object.fromEntries(Object.entries(saved.annotations).filter(
            ([, labels]) => labels.every(label => ["HEADER", "OTHER", "UNLABELED"].includes(label))
        ));
        state.examples = saved.relations.filter(pair => pair.source === "manual").map(pairFromRecord);
        state.predictions = saved.relations.filter(pair => pair.source === "rule").map(pairFromRecord);
        state.confirmed = true;
        recomputePreview();
    }
    if (state.confirmed) showConfirmedResults();
};

function showConfirmedResults() {
    const body = document.getElementById("resultBody");
    body.replaceChildren();
    document.querySelector("#resultSection thead tr").innerHTML = "<th>#</th><th>CODE</th><th>KEY</th><th>VALUE</th><th>Nguồn</th>";
    allPairs().forEach((pair, index) => {
        const row = document.createElement("tr");
        for (const value of [index + 1, pair.code?.text || "—", pair.key.text, pair.value.text,
            pairSource(pair) === "manual" ? "Người dùng chọn" : "Suy luận"]) {
            const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell);
        }
        body.appendChild(row);
    });
    document.getElementById("resultSummary").textContent = "Đã lưu annotation theo region ID và giữ các nhóm quan hệ trong annotations.json.";
    instruction.textContent = "Phiếu đã được lưu. UNLABELED chủ động được giữ; các region chưa có nhãn được gán OTHER.";
    instruction.className = "instruction confirmed";
    resultSection.hidden = false;
}

document.getElementById("editConfirmed").addEventListener("click", () => {
    state.confirmed = false;
    annotation.dirty = true;
    instruction.className = "instruction";
    instruction.textContent = "Đang sửa phiếu. Bấm Xác nhận kết quả để lưu thay thế bản cũ.";
    recomputePreview();
});

confirmResults = async function () {
    const page = pages[currentPageIndex];
    const labels = visibleAnnotations(true);
    const relations = allPairs().map((pair, index) => ({
        id: `P${index + 1}`,
        code_region_ids: pair.code?.regionIds || null,
        key_region_ids: [...pair.key.regionIds],
        value_region_ids: [...pair.value.regionIds],
        code_text: pair.code?.text || null,
        key_text: pair.key.text, value_text: pair.value.text,
        code_box: pair.code?.box || null, key_box: [...pair.key.box], value_box: [...pair.value.box],
        mode: pair.mode || "spatial", separator: pair.separator || "",
        source: pairSource(pair),
        rule_score: pairSource(pair) === "manual" ? null : pair.score,
    }));
    const record = {image_name: page.imageName, ocr_fingerprint: page.ocrFingerprint,
        annotations: labels, relations};
    const main = document.querySelector("main");
    main.inert = true;
    instruction.textContent = "Đang lưu annotation; phiếu chưa được xác nhận cho đến khi server ghi xong.";
    await fetch("/api/annotations", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(record),
    }).then(async response => {
        if (!response.ok) throw new Error("Server chưa lưu được annotation.");
        annotation.saved[page.imageName] = await response.json();
        annotation.labels = Object.fromEntries(Object.entries(labels).filter(
            ([, roles]) => roles.every(role => ["OTHER", "UNLABELED", "HEADER"].includes(role))
        ));
        annotation.dirty = false;
        state.confirmed = true;
        recomputePreview(); showConfirmedResults();
    }).finally(() => { main.inert = false; });
};

window.addEventListener("beforeunload", event => {
    const unsaved = annotation.dirty || annotation.groups.length || annotation.selected.size
        || [...annotation.sessions.entries()].some(([index, session]) => index !== annotation.activePage
            && !session.snapshot.confirmed && (session.snapshot.dirty
                || session.snapshot.groups.length || session.snapshot.selected.length));
    if (unsaved) { event.preventDefault(); event.returnValue = ""; }
});

async function initializeAnnotations() {
    const response = await fetch("/api/annotations");
    if (!response.ok) throw new Error("Hãy mở giao diện qua serve_page3_annotations.py.");
    annotation.saved = (await response.json()).pages;
    loadPage(0);
}
