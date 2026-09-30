// Mở rộng giao diện gốc trước khi gắn listener; giữ các hàm suy luận hiện có.
const original = {
    reset: resetAll, load: loadPage, styles: refreshBoxStyles,
    logical: renderLogicalBlocks, links: renderLinks, pair: pairHtml,
    buttons: updateButtons, inline: confirmInlineSelection,
};
const annotation = {
    selected: new Set(), groups: [], labels: {}, history: [],
    mode: "sample", sessions: new Map(), activePage: -1,
    saved: {}, editing: null, dirty: false, frozenPairs: [], lockedRows: [],
    keywordSuggestions: new Set(),
};
const explicitRoles = [
    "HEADER", "FIELD_TABLE", "NOTE", "PAGE_NUMBER", "FORM_NUMBER",
    "OTHER", "UNLABELED",
];
const suggestionRoles = new Set(["FIELD_TABLE", "PAGE_NUMBER", "FORM_NUMBER"]);
const annotationStyle = document.createElement("style");
annotationStyle.textContent = `
    .annotation-toolbar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:10px}
    .annotation-toolbar select{padding:8px;font:inherit;border-radius:7px}
    .annotation-toolbar input{min-width:280px;padding:8px;font:inherit;border-radius:7px;border:1px solid #cbd5e1}
    .annotation-toolbar button{padding:8px 10px}
    .annotation-toolbar button:disabled{opacity:.45;cursor:not-allowed}
    #groupSummary{width:100%;font-size:14px;color:#334155;line-height:1.5}
    .ocr-box.annotation-selected{outline:4px solid #eab308;outline-offset:3px}
    .ocr-box.annotation-unlabeled{border-color:#475569;border-style:dashed;background:#47556945}
    .ocr-box.annotation-other{border-color:#94a3b8;background:#94a3b81a}
    .ocr-box.annotation-header{border-color:#7c3aed;background:#7c3aed24}
    .ocr-box.annotation-field_table{border-color:#0891b2;background:#0891b224}
    .ocr-box.annotation-note{border-color:#ca8a04;background:#ca8a0424}
    .ocr-box.annotation-page_number{border-color:#db2777;background:#db277724}
    .ocr-box.annotation-form_number{border-color:#ea580c;background:#ea580c24}
    .ocr-box.annotation-keyword-suggestion{outline:3px dashed #f97316;outline-offset:2px}
    .ocr-box.annotation-frozen-key,.ocr-box.annotation-frozen-value{
        border-color:#64748b;background:#64748b2b;border-style:dotted;opacity:.8}
    .ocr-box.annotation-locked-row{box-shadow:inset 0 0 0 2px #94a3b8;background:#94a3b81a}
    .logical-box.frozen-key,.logical-box.frozen-value{
        border-color:#64748b;border-style:dotted;opacity:.75}
    .logical-box.frozen-key span,.logical-box.frozen-value span{
        background:#475569}
    .frozen-link{opacity:.7}
    .swatch.frozen{background:#64748b2b;border-color:#64748b;border-style:dotted}
    .swatch.locked-row{background:#94a3b81a;border-color:#94a3b8}
    #frozenRuleSummary{width:100%;padding:7px 9px;border:1px solid #cbd5e1;
        border-radius:6px;color:#475569;background:#f8fafc;font-size:13px}
    .annotation-drag{position:absolute;border:3px dashed #eab308;background:#facc1526;
        pointer-events:none;z-index:2147483647;display:none}
    .steps{display:none}.instruction{font-size:14px;padding:8px;margin-bottom:6px}
    .control-panel{padding:12px}.side-panel{top:320px}
    header{padding:14px 24px}.actions{margin-top:6px}
`;
document.head.appendChild(annotationStyle);
document.title = "Annotation page 3 · KEY / VALUE / role";
document.querySelector("header h1").textContent = "Annotation page 3 · KEY / VALUE / role";
document.querySelector("header p").textContent = "Rule Induction từ mẫu; lưu annotation theo ID region.";
const annotationToolbar = document.createElement("div");
annotationToolbar.className = "annotation-toolbar";
annotationToolbar.innerHTML = `
    <label>Chế độ <select id="annotationMode">
        <option value="sample">Chọn nhóm cho mẫu</option>
        <option value="inspect">Kiểm tra / sửa cặp</option>
        <option value="HEADER">Đánh dấu HEADER</option>
        <option value="FIELD_TABLE">Đánh dấu FIELD_TABLE</option>
        <option value="NOTE">Đánh dấu NOTE</option>
        <option value="PAGE_NUMBER">Đánh dấu PAGE_NUMBER</option>
        <option value="FORM_NUMBER">Đánh dấu FORM_NUMBER</option>
        <option value="UNLABELED">Đánh dấu UNLABELED</option>
        <option value="OTHER">Đánh dấu OTHER</option>
    </select></label>
    <input id="fieldTableKeywords" type="text" hidden
        placeholder="Keyword FIELD_TABLE, cách nhau bằng dấu phẩy">
    <button id="commitGroup" class="secondary">Chốt nhóm 1</button>
    <button id="inlineGroup" class="secondary">Key-value trong cùng bbox</button>
    <button id="breakRule" class="secondary">Ngắt rule / Bảng mới</button>
    <button id="applyAnnotation" class="secondary">Gán nhãn</button>
    <button id="clearAnnotation" class="secondary">Gỡ nhãn đã gán</button>
    <button id="clearSelection" class="secondary">Hủy chọn</button>
    <button id="undoAnnotation" class="secondary">Hoàn tác annotation</button>
    <button id="editConfirmed" class="secondary">Sửa phiếu đã xác nhận</button>
    <span>Phím N: page sau · Q: gán nhãn · E: sửa phiếu · S: xác nhận</span>
    <div id="frozenRuleSummary" role="status"></div>
    <div id="groupSummary" role="status" aria-live="polite"></div>
`;
document.querySelector(".control-panel").appendChild(annotationToolbar);
document.querySelector(".legend").insertAdjacentHTML("beforeend", `
    <span class="legend-item"><span class="swatch" style="background:#7c3aed24;border-color:#7c3aed"></span>HEADER</span>
    <span class="legend-item"><span class="swatch" style="background:#0891b224;border-color:#0891b2"></span>FIELD_TABLE</span>
    <span class="legend-item"><span class="swatch" style="background:#ca8a0424;border-color:#ca8a04"></span>NOTE</span>
    <span class="legend-item"><span class="swatch" style="background:#db277724;border-color:#db2777"></span>PAGE_NUMBER</span>
    <span class="legend-item"><span class="swatch" style="background:#ea580c24;border-color:#ea580c"></span>FORM_NUMBER</span>
    <span class="legend-item"><span class="swatch frozen"></span>Đã khóa từ rule trước</span>
    <span class="legend-item"><span class="swatch locked-row"></span>Hàng Y đã khóa</span>
`);
const modeSelect = document.getElementById("annotationMode");
const fieldTableKeywords = document.getElementById("fieldTableKeywords");
const commitButton = document.getElementById("commitGroup");
const inlineGroupButton = document.getElementById("inlineGroup");
const breakRuleButton = document.getElementById("breakRule");
const applyButton = document.getElementById("applyAnnotation");
const clearAnnotationButton = document.getElementById("clearAnnotation");
const undoAnnotationButton = document.getElementById("undoAnnotation");
cancelKeyButton.textContent = "Hủy mẫu đang chọn";
const dragBox = document.createElement("div");
dragBox.className = "annotation-drag";
document.getElementById("imageStage").appendChild(dragBox);

function allPairIds(pair) {
    return [...pair.key.regionIds, ...pair.value.regionIds];
}

function normalizedFieldTableKeyword(value) {
    return String(value || "").normalize("NFC").trim().replace(/\s+/g, " ").toLowerCase();
}

function fieldTableKeywordSet() {
    return new Set(
        fieldTableKeywords.value
            .split(/[\n,;]+/)
            .map(normalizedFieldTableKeyword)
            .filter(Boolean),
    );
}

function updateRoleSuggestions() {
    annotation.keywordSuggestions.clear();
    if (!suggestionRoles.has(annotation.mode)) return;
    const occupiedIds = new Set(allPairs().flatMap(allPairIds));
    const protectedRoles = new Set([
        "FIELD_TABLE", "HEADER", "NOTE", "PAGE_NUMBER", "FORM_NUMBER",
    ]);
    const eligibleRegions = regions.filter(region => {
        const labels = annotation.labels[region.id] || [];
        return !labels.some(role => protectedRoles.has(role))
            && !occupiedIds.has(region.id);
    });
    if (annotation.mode === "FIELD_TABLE") {
        const keywords = fieldTableKeywordSet();
        if (!keywords.size) return;
        for (const region of eligibleRegions) {
            if (keywords.has(normalizedFieldTableKeyword(region.text))) {
                annotation.keywordSuggestions.add(region.id);
            }
        }
        return;
    }
    const hasAssignedRole = regions.some(region => (
        annotation.labels[region.id] || []
    ).includes(annotation.mode));
    if (hasAssignedRole) return;
    const orderedRegions = [...eligibleRegions].sort((first, second) => (
        first.box[1] - second.box[1] || first.box[0] - second.box[0]
    ));
    const candidate = orderedRegions.find(region => {
        const text = String(region.text || "");
        return annotation.mode === "PAGE_NUMBER"
            ? text.includes("/") && !text.includes("-")
            : text.includes("/") && text.includes("-");
    });
    if (candidate) annotation.keywordSuggestions.add(candidate.id);
}

function pairRowRange(pair) {
    const boxes = [pair.key, pair.value].map(block => block.box);
    const top = Math.min(...boxes.map(box => box[1]));
    const bottom = Math.max(...boxes.map(box => box[3]));
    const padding = Math.max(6, (bottom - top) * 0.35);
    return {top: top - padding, bottom: bottom + padding};
}

function isLockedRow(region) {
    if (!region) return false;
    const y = centerY(region);
    return annotation.lockedRows.some(row => y >= row.top && y <= row.bottom);
}

function activePairs() {
    return [...state.examples, ...state.predictions];
}

function visibleFrozenPairs() {
    const activeIds = new Set(activePairs().flatMap(allPairIds));
    return annotation.frozenPairs.filter(pair => (
        pairSource(pair) === "manual"
        || !allPairIds(pair).some(id => activeIds.has(id))
    ));
}

function allPairs() {
    return [...visibleFrozenPairs(), ...activePairs()];
}

function pairSource(pair) {
    return pair.source || (state.examples.includes(pair) ? "manual" : "rule");
}

function frozenPairsTouching(ids) {
    return annotation.frozenPairs.filter(pair => (
        allPairIds(pair).some(id => ids.includes(id))
    ));
}

function isLockedFrozenSample(pair) {
    return annotation.frozenPairs.includes(pair) && pairSource(pair) === "manual";
}

function releaseFrozenPredictions(ids) {
    const conflicts = frozenPairsTouching(ids);
    const manualConflict = conflicts.find(pair => pairSource(pair) === "manual");
    if (manualConflict) {
        instruction.textContent = "Vùng thuộc mẫu người dùng của rule đã ngắt. Hãy chuyển sang Kiểm tra / sửa cặp để sửa hoặc xóa mẫu đó.";
        return false;
    }
    annotation.frozenPairs = annotation.frozenPairs.filter(pair => !conflicts.includes(pair));
    return true;
}

function snapshotAnnotation() {
    return structuredClone({
        examples: state.examples, predictions: state.predictions,
        frozenPairs: annotation.frozenPairs,
        lockedRows: annotation.lockedRows,
        labels: annotation.labels, groups: annotation.groups,
        selected: [...annotation.selected], suppressed: [...state.suppressedPairIds],
        suppressionHistory: state.suppressionHistory, confirmed: state.confirmed,
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
    annotation.labels = snapshot.labels;
    annotation.frozenPairs = snapshot.frozenPairs || [];
    annotation.lockedRows = snapshot.lockedRows || [];
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
        for (const role of ["KEY", "VALUE"]) {
            for (const id of pair[role.toLowerCase()].regionIds) {
                labels[id] = [...new Set([...(labels[id] || []), role])];
            }
        }
    }
    for (const id of state.autoOtherIds) {
        if (!labels[id]) labels[id] = ["OTHER"];
    }
    if (finalize) {
        for (const region of regions) labels[region.id] ||= ["OTHER"];
    }
    return labels;
}

updateButtons = function () {
    original.buttons();
    const selectedCount = annotation.selected.size;
    const suggestionCount = suggestionRoles.has(annotation.mode)
        ? annotation.keywordSuggestions.size
        : 0;
    const groupCount = annotation.groups.length;
    const isSample = annotation.mode === "sample";
    commitButton.disabled = state.confirmed || !isSample || !selectedCount;
    commitButton.textContent = `Chốt nhóm ${groupCount + 1} (hoặc click lại bbox)`;
    inlineGroupButton.disabled = state.confirmed || !isSample || groupCount > 0 || selectedCount !== 1;
    breakRuleButton.disabled = state.confirmed || !state.examples.length || groupCount > 0
        || selectedCount > 0 || Boolean(annotation.editing);
    applyButton.disabled = state.confirmed || isSample || annotation.mode === "inspect"
        || !(selectedCount || suggestionCount);
    applyButton.textContent = `Gán ${annotation.mode === "sample" || annotation.mode === "inspect" ? "nhãn" : annotation.mode}`
        + (suggestionCount ? ` (${suggestionCount} gợi ý)` : "");
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
        + `${suggestionCount} gợi ý ${suggestionRoles.has(annotation.mode) ? annotation.mode : "role"} · `
        + `${visibleFrozenPairs().length} cặp đã ngắt rule · `
        + `${annotation.lockedRows.length} hàng Y đã khóa. ${groups}`;
    document.getElementById("frozenRuleSummary").textContent = visibleFrozenPairs().length
        ? `${visibleFrozenPairs().length} cặp đã khóa từ rule trước; `
            + `${annotation.lockedRows.length} hàng Y của mẫu cũ cũng bị loại khỏi rule mới.`
        : "Chưa có cặp nào được khóa từ rule trước.";
};

updateSteps = function () {
    selectionStatus.textContent = state.confirmed
        ? "Đã xác nhận và lưu phiếu"
        : `${annotation.groups.length} nhóm đã chốt · ${state.examples.length} mẫu · `
        + `${state.predictions.length} suy luận · ${visibleFrozenPairs().length} cặp đã ngắt rule`;
};

refreshBoxStyles = function () {
    original.styles();
    const labels = visibleAnnotations();
    const groupIds = new Set(annotation.groups.flatMap(group => group.regionIds));
    const frozenKeyIds = new Set(visibleFrozenPairs().flatMap(pair => pair.key.regionIds));
    const frozenValueIds = new Set(visibleFrozenPairs().flatMap(pair => pair.value.regionIds));
    for (const box of boxLayer.children) {
        const id = box.dataset.regionId;
        const roles = labels[id] || [];
        for (const role of [
            "KEY", "VALUE", "HEADER", "FIELD_TABLE", "NOTE", "PAGE_NUMBER",
            "FORM_NUMBER", "UNLABELED", "OTHER",
        ]) {
            box.classList.toggle(`annotation-${role.toLowerCase()}`, roles.includes(role));
        }
        box.classList.toggle("annotation-selected", annotation.selected.has(id) || groupIds.has(id));
        box.classList.toggle("annotation-keyword-suggestion", annotation.keywordSuggestions.has(id));
        box.classList.toggle("annotation-frozen-key", frozenKeyIds.has(id));
        box.classList.toggle("annotation-frozen-value", frozenValueIds.has(id));
        box.classList.toggle("annotation-locked-row", isLockedRow(regionById.get(id)));
        if (roles.length || annotation.selected.has(id) || groupIds.has(id)
            || annotation.keywordSuggestions.has(id)) {
            box.classList.remove("hidden-unused");
        }
        box.title = `${id} · ${roles.join("/") || "Chưa gán"} · ${regionById.get(id)?.text || ""}`;
    }
};

renderLogicalBlocks = function () {
    original.logical();
    for (const pair of visibleFrozenPairs()) {
        appendLogicalBox(pair.key, "frozen-key", "KEY đã khóa từ rule trước");
        appendLogicalBox(pair.value, "frozen-value", "VALUE đã khóa từ rule trước");
    }
};

renderLinks = function () {
    original.links();
    for (const pair of visibleFrozenPairs()) {
        svgLine(pair, "frozen-link", "#64748b", "6 8");
    }
};

pairHtml = function (pair, includeScore, allowRemoval = false) {
    const item = original.pair(pair, includeScore, allowRemoval);
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

// Chạy suy luận cho một họ mẫu KEY → VALUE.
recomputePreview = function () {
    if (!state.confirmed) {
        const allRegions = pages[currentPageIndex].regions;
        const examples = state.examples;
        const frozenIds = new Set(annotation.frozenPairs.filter(
            pair => pairSource(pair) === "manual",
        ).flatMap(allPairIds));
        state.autoOtherIds = new Set();
        regions = allRegions.filter(region => !annotation.labels[region.id]
            && !frozenIds.has(region.id) && !isLockedRow(region));
        state.examples = examples;
        state.rule = null;
        state.inlineRule = null;
        state.predictions = [];
        if (examples.length) {
            state.rule = inferRule();
            state.inlineRule = inferInlineRule();
            if (state.rule) classifyPredictions(state.rule);
            const occupied = new Set(examples.flatMap(allPairIds));
            for (const pair of state.predictions) {
                allPairIds(pair).forEach(id => occupied.add(id));
            }
            state.predictions.push(...predictInlinePairs(state.inlineRule, occupied));
        }
        regions = allRegions;
        const occupied = new Set([
            ...annotation.frozenPairs.filter(pair => pairSource(pair) === "manual"),
            ...examples,
        ].flatMap(allPairIds));
        state.predictions = state.predictions.sort((a, b) => b.score - a.score).filter(pair => {
            const ids = allPairIds(pair);
            if (ids.some(id => occupied.has(id)
                || annotation.labels[id]
                || state.autoOtherIds.has(id))) return false;
            ids.forEach(id => occupied.add(id));
            return true;
        }).sort((a, b) => pairY(a) - pairY(b));
    }
    updateRoleSuggestions();
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
    const region = regionById.get(id);
    if (annotation.mode === "sample" && isLockedRow(region)) {
        instruction.textContent = "Hàng Y này đã thuộc rule trước và đang bị khóa; hãy chọn mẫu ở hàng khác.";
        return;
    }
    if (annotation.mode === "sample" && state.autoOtherIds.has(id)) {
        instruction.textContent = "Region header đã được nhận diện là OTHER; không dùng làm mẫu suy luận.";
        return;
    }
    if (annotation.mode === "inspect") {
        const pair = allPairs().find(pair => allPairIds(pair).includes(id));
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
    const key = groups[0];
    const value = groups[1];
    let pair = {key, value, mode: "spatial"};
    if (key.regionIds.length === 1 && value.regionIds.length === 1) {
        const allRegions = regions;
        const frozenIds = new Set(annotation.frozenPairs.filter(
            pair => pairSource(pair) === "manual",
        ).flatMap(allPairIds));
        regions = regions.filter(region => (
            !annotation.labels[region.id]
            && !frozenIds.has(region.id)
            && !isLockedRow(region)
        ));
        const grouped = buildExample(key.regionIds[0], value.regionIds[0]);
        regions = allRegions;
        if (grouped) pair = grouped;
    }
    if (annotation.editing) {
        state.suppressedPairIds.delete(pairInstanceId(annotation.editing.pair));
        annotation.editing = null;
    }
    state.examples.push(pair);
    annotation.groups = [];
    annotation.selected.clear();
    recomputePreview();
    instruction.textContent = "Đã xác nhận mẫu KEY → VALUE và cập nhật preview.";
}

commitButton.addEventListener("click", () => {
    const ids = [...annotation.selected];
    const activeUsed = new Set([
        ...state.examples.flatMap(allPairIds),
        ...state.predictions.flatMap(allPairIds),
        ...annotation.groups.flatMap(group => group.regionIds),
    ]);
    if (ids.some(id => activeUsed.has(id) || annotation.labels[id]
        || state.autoOtherIds.has(id)
        || isLockedRow(regionById.get(id)))) {
        if (ids.some(id => isLockedRow(regionById.get(id)))) {
            instruction.textContent = "Hàng Y này đã thuộc rule trước và đang bị khóa; hãy chọn mẫu ở hàng khác.";
            return;
        }
        instruction.textContent = "Vùng đã thuộc mẫu hoặc có nhãn. Hãy sửa mẫu/gỡ nhãn trước khi dùng lại.";
        return;
    }
    if (!releaseFrozenPredictions(ids)) return;
    rememberAnnotation();
    annotation.groups.push(combineRegions(ids.map(id => regionById.get(id))));
    annotation.selected.clear();
    if (annotation.groups.length === 2) finishGroups();
    else {
        instruction.textContent = "Đã chốt nhóm 1. Chọn các region của nhóm 2 rồi bấm Chốt nhóm.";
        refreshBoxStyles(); updateSteps(); updateButtons();
    }
});
breakRuleButton.addEventListener("click", () => {
    rememberAnnotation();
    annotation.lockedRows.push(
        ...[...state.examples, ...state.predictions].map(pairRowRange),
    );
    for (const id of state.autoOtherIds) {
        if (!annotation.labels[id]) annotation.labels[id] = ["OTHER"];
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
    if (isLockedFrozenSample(pair)) {
        instruction.textContent = "Mẫu của rule đã ngắt đang được khóa; hãy tạo mẫu mới cho rule hiện tại.";
        return;
    }
    if (annotation.groups.length || annotation.selected.size || annotation.editing) {
        instruction.textContent = "Hãy hoàn tất hoặc hủy mẫu đang chọn trước khi sửa cặp khác.";
        return;
    }
    rememberAnnotation();
    const frozen = annotation.frozenPairs.includes(pair);
    const manual = pairSource(pair) === "manual";
    annotation.editing = {pair: structuredClone(pair), manual, frozen};
    state.examples = state.examples.filter(example => example !== pair);
    annotation.frozenPairs = annotation.frozenPairs.filter(example => example !== pair);
    state.suppressedPairIds.add(pairInstanceId(pair));
    annotation.mode = "sample";
    modeSelect.value = "sample";
    closePredictionPopover(); recomputePreview();
    instruction.textContent = "Đang sửa: chọn lại đủ 2 nhóm. Hủy mẫu đang chọn sẽ khôi phục cặp cũ.";
};

cancelCurrentKey = function () {
    if (annotation.editing) {
        const {pair, manual, frozen} = annotation.editing;
        state.suppressedPairIds.delete(pairInstanceId(pair));
        if (frozen) annotation.frozenPairs.push(pair);
        else if (manual) state.examples.push(pair);
    }
    annotation.editing = null;
    annotation.groups = [];
    cancelGroupSelection(); recomputePreview();
    instruction.textContent = "Đã hủy mẫu đang chọn. Click hoặc kéo khung để chọn nhóm mới.";
};

const originalSuppress = suppressPrediction;
suppressPrediction = function (pair) {
    if (isLockedFrozenSample(pair)) {
        instruction.textContent = "Mẫu của rule đã ngắt đang được khóa, không thể xóa từ rule mới.";
        return;
    }
    rememberAnnotation();
    if (annotation.frozenPairs.includes(pair)) {
        annotation.frozenPairs = annotation.frozenPairs.filter(item => item !== pair);
        recomputePreview();
        return;
    }
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
    if (!releaseFrozenPredictions([id])) return;
    rememberAnnotation();
    if (annotation.editing) {
        state.suppressedPairIds.delete(pairInstanceId(annotation.editing.pair));
        annotation.editing = null;
    }
    original.inline();
    annotation.selected.clear(); annotation.groups = [];
    recomputePreview();
};

function applySelectedLabel() {
    const ids = new Set(annotation.selected);
    if (suggestionRoles.has(annotation.mode)) {
        annotation.keywordSuggestions.forEach(id => ids.add(id));
    }
    if (!ids.size) return;
    const affected = allPairs().filter(pair => allPairIds(pair).some(id => ids.has(id)));
    if (affected.some(isLockedFrozenSample)) {
        instruction.textContent = "Region thuộc mẫu của rule đã ngắt; không thể đổi nhãn từ rule mới.";
        return;
    }
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
    annotation.keywordSuggestions.clear();
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
    fieldTableKeywords.hidden = annotation.mode !== "FIELD_TABLE";
    updateRoleSuggestions();
    updateButtons();
    instruction.textContent = annotation.mode === "sample"
        ? "Click nhiều bbox hoặc kéo khung, rồi Chốt nhóm. Chọn đủ 2 nhóm KEY/VALUE sẽ tự xác nhận mẫu."
        : annotation.mode === "inspect"
            ? "Bấm bbox hoặc Sửa nhóm trong danh sách để kiểm tra cặp."
            : annotation.mode === "FIELD_TABLE"
                ? "Nhập keyword rồi kiểm tra các bbox được viền gợi ý. Chỉ khớp toàn bộ text của một bbox độc lập; bấm Q hoặc Gán nhãn để xác nhận."
            : annotation.mode === "PAGE_NUMBER"
                ? "Bbox đầu tiên từ trên xuống có dấu / và không có dấu - sẽ được viền gợi ý; bấm Q hoặc Gán nhãn để xác nhận."
            : annotation.mode === "FORM_NUMBER"
                ? "Bbox đầu tiên từ trên xuống có cả dấu / và dấu - sẽ được viền gợi ý; bấm Q hoặc Gán nhãn để xác nhận."
            : `Click hoặc kéo khung để chọn region; Shift + kéo để thêm vùng. Bấm Gán ${annotation.mode} để áp dụng.`;
});
fieldTableKeywords.addEventListener("input", () => {
    updateRoleSuggestions();
    refreshBoxStyles();
    updateButtons();
});

document.addEventListener("keydown", event => {
    const target = event.target;
    const isEditing = target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || target.isContentEditable;
    if (isEditing || event.ctrlKey || event.altKey || event.metaKey) return;
    const key = event.key.toLowerCase();
    if (key === "n" && currentPageIndex < pages.length - 1) {
        event.preventDefault();
        loadPage(currentPageIndex + 1);
    } else if (key === "e") {
        const editButton = document.getElementById("editConfirmed");
        if (!editButton.disabled) {
            event.preventDefault();
            editButton.click();
        }
    } else if (key === "q" && !applyButton.disabled) {
        event.preventDefault();
        applyButton.click();
    } else if (key === "s" && !confirmResultButton.disabled) {
        event.preventDefault();
        confirmResultButton.click();
    }
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
            if (!isLockedRow(region)
                && centerX(region) >= Math.min(drag.start[0], end[0]) && centerX(region) <= Math.max(drag.start[0], end[0])
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
    annotation.history = []; annotation.editing = null; annotation.frozenPairs = [];
    annotation.lockedRows = []; annotation.dirty = false;
    annotation.keywordSuggestions.clear();
    original.reset();
    instruction.textContent = "Chọn một hoặc nhiều bbox rồi Chốt nhóm. Hai nhóm KEY/VALUE sẽ tự xác nhận mẫu.";
};

function pairFromRecord(relation) {
    const key = combineRegions(relation.key_region_ids.map(id => regionById.get(id)));
    const value = combineRegions(relation.value_region_ids.map(id => regionById.get(id)));
    key.text = relation.key_text;
    value.text = relation.value_text;
    return {
        key, value,
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
            ([, labels]) => labels.every(label => explicitRoles.includes(label))
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
    document.querySelector("#resultSection thead tr").innerHTML = "<th>#</th><th>KEY</th><th>VALUE</th><th>Nguồn</th>";
    allPairs().forEach((pair, index) => {
        const row = document.createElement("tr");
        for (const value of [index + 1, pair.key.text, pair.value.text,
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
        code_region_ids: null,
        key_region_ids: [...pair.key.regionIds],
        value_region_ids: [...pair.value.regionIds],
        code_text: null,
        key_text: pair.key.text, value_text: pair.value.text,
        code_box: null, key_box: [...pair.key.box], value_box: [...pair.value.box],
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
            ([, roles]) => roles.every(role => explicitRoles.includes(role))
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
