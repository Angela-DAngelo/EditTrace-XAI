const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = { session: null, report: null, view: "current", preview: null, selectedRepair: null, dataset: null, traces: [] };

async function api(path, body = null) {
  const options = body === null ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

function busy(on, message = "Replaying interventions") {
  $("#loading").hidden = !on;
  $("#loading p").textContent = message;
  $$("button").forEach((button) => { if (button.id !== "upload-btn") button.disabled = on; });
  if (!on && state.session) {
    $("#undo-btn").disabled = !state.session.can_undo;
    $("#add-edit-btn").disabled = state.session.dataset.mode || state.session.actions.length >= 6;
    $("#reset-btn").disabled = state.session.dataset.mode;
    $("#diagnose-btn").disabled = state.session.actions.length === 0;
  }
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 2800);
}

async function createSession(image = null, filename = null) {
  busy(true, image ? "Loading source image" : "Preparing starter trace");
  try {
    state.session = await api("/api/sessions", image ? { image, filename } : {});
    state.report = null; state.preview = null; state.selectedRepair = null;
    render();
  } catch (error) { toast(error.message); } finally { busy(false); }
}

function render() {
  if (!state.session) return;
  const session = state.session;
  $("#model-name").textContent = session.model.id;
  $("#model-mode").textContent = session.model.mode;
  if (session.dataset.mode) {
    const hashes = `${session.dataset.action_hashes_matching}/${session.dataset.action_hashes_checked}`;
    const ready = session.dataset.eligible === true && session.dataset.replay_valid === true;
    $("#model-warning").innerHTML = `<b>Frozen trace · ${ready ? "ready" : "blocked"}</b><span>GT ${session.dataset.ground_truth_label} · action hashes ${hashes}. ${escapeHtml(session.dataset.eligibility_reason)}; ${escapeHtml(session.dataset.replay_reason)}</span>`;
  } else {
    $("#model-warning").innerHTML = session.model.experimental
      ? `<b>Experimental adapter</b><span>${escapeHtml(session.model.id)} outputs require the frozen dataset protocol before use as evidence.</span>`
      : "<b>Research demo</b><span>Surrogate outputs are workflow checks, not experimental evidence.</span>";
  }
  $("#filename").textContent = session.dataset.mode ? session.dataset.source_id : session.filename;
  $("#history-count").textContent = `${session.actions.length} / 6`;
  $("#empty-history").hidden = session.actions.length > 0;
  $("#undo-btn").disabled = !session.can_undo;
  $("#add-edit-btn").disabled = session.dataset.mode || session.actions.length >= 6;
  $("#reset-btn").disabled = session.dataset.mode;
  const confirmatoryReady = session.dataset.mode && session.dataset.eligible === true && session.dataset.replay_valid === true;
  $("#diagnose-btn").disabled = session.actions.length === 0;
  $("#diagnose-btn").textContent = session.dataset.mode && !confirmatoryReady ? "Run exploratory" : "Run diagnosis";
  $("#diagnose-btn").title = session.dataset.mode && !confirmatoryReady ? "Runs all analyses outside the frozen confirmatory protocol" : "Run action-level diagnosis";
  $("#manual-edit-controls").hidden = session.dataset.mode;
  $("#trace-lock").hidden = !session.dataset.mode;
  syncEditOptions();
  $("#history-list").innerHTML = session.actions.map((action, index) => {
    const rank = state.report?.ranking.find((item) => item.action_id === action.id);
    const score = rank ? `<span class="score">Δ ${signed(rank.effect)}</span>` : "";
    return `<li class="history-item"><span class="history-index">${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(action.label)}</strong><small>${escapeHtml(action.type)} · deterministic</small></div>${score}</li>`;
  }).join("");
  renderPrediction(state.preview?.prediction || session.prediction);
  renderImage();
  renderReport();
}

function renderPrediction(prediction) {
  $("#target-class").textContent = prediction.target_class;
  $("#predicted-class").textContent = prediction.predicted_class;
  $("#target-margin").textContent = signed(prediction.margin);
  $("#target-probability").textContent = `${(prediction.target_probability * 100).toFixed(1)}%`;
  const decision = $("#decision-state");
  decision.className = `decision ${prediction.recovered ? "recovered" : "shifted"}`;
  decision.innerHTML = `<i></i><span>${prediction.recovered ? "Target retained" : "Prediction shifted"}</span>`;
}

function renderImage() {
  const session = state.session;
  const current = state.preview?.image || session.current_image;
  $("#main-image").src = state.view === "source" ? session.source_image : current;
  $("#compare-source").src = session.source_image;
  $("#compare-current").src = current;
  $("#image-single").hidden = state.view === "compare";
  $("#image-compare").hidden = state.view !== "compare";
  $("#image-compare").style.display = state.view === "compare" ? "grid" : "none";
  $("#preview-badge").hidden = !state.preview;
  $("#stage-title").textContent = state.preview ? "Counterfactual replay" : state.view === "source" ? "Source image" : state.view === "compare" ? "Source / replay" : "Current replay";
  $$(".view-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.view === state.view));
}

function formatParameterChange(action) {
  const original = action.original_params || {};
  const repaired = action.repaired_params || {};
  const keys = Object.keys(repaired).filter((key) => key !== "bypass" && key !== "subsampling" && key !== "optimize" && key !== "progressive" && key !== "interpolation");
  if (!keys.length) return "Parameter attenuation";
  const formatValue = (value) => typeof value === "number" ? Number(value).toFixed(Number.isInteger(value) ? 0 : 3).replace(/\.?0+$/, "") : String(value);
  return keys.map((key) => `${key} ${formatValue(original[key])} → ${formatValue(repaired[key])}`).join(" · ");
}

function renderReport() {
  const report = state.report;
  $("#latency").textContent = report ? `${report.latency_ms.toFixed(1)} ms · ${report.model_calls} calls` : "—";
  $("#diagnosis-empty").hidden = !!report;
  $("#diagnosis-content").hidden = !report;
  $("#analysis-scope").hidden = !report;
  if (report) {
    const exploratory = report.analysis_mode === "exploratory";
    $("#analysis-scope").className = `analysis-scope ${exploratory ? "exploratory" : ""}`;
    $("#analysis-scope").textContent = exploratory
      ? "Exploratory analysis: diagnosis, interactions and repairs are executable but excluded from confirmatory paper results."
      : "Confirmatory-ready analysis: model eligibility and frozen replay integrity passed.";
  }
  if (report) {
    const maximum = Math.max(.0001, ...report.ranking.map((item) => Math.abs(item.effect)));
    $("#ranking-list").innerHTML = report.ranking.map((item, index) => `<li class="rank-row ${item.effect < 0 ? "harmful" : ""}" data-preview-action="${item.action_id}"><div class="rank-top"><div class="rank-label"><b>${index + 1}. ${escapeHtml(item.label)}</b><span>adjacent ${signed(item.adjacent_score_drop)} · magnitude ${item.edit_magnitude.toFixed(2)}</span></div><span class="rank-value">${signed(item.effect)}</span></div><div class="effect-track"><i style="width:${Math.max(2, Math.abs(item.effect) / maximum * 100)}%"></i></div>${item.restores_target ? '<span class="flag">Restores target</span>' : ""}</li>`).join("");
  }
  const hasInteractions = !!report && report.interactions.length > 0;
  $("#interaction-empty").hidden = hasInteractions;
  $("#interaction-content").hidden = !hasInteractions;
  if (hasInteractions) {
    $("#interaction-list").innerHTML = report.interactions.map((item) => `<li class="interaction-row" data-preview-pair="${item.action_ids.join(",")}"><div class="interaction-top"><div class="interaction-label"><b>${escapeHtml(item.labels.join(" × "))}</b><span>joint margin ${signed(item.joint_margin)}</span></div><span class="interaction-value">${signed(item.effect)}</span></div>${item.pair_only_recovery ? '<span class="flag">Pair-only recovery</span>' : ""}</li>`).join("");
  }
  const hasRepairs = !!report && report.repairs.length > 0;
  $("#repair-empty").hidden = hasRepairs;
  $("#repair-content").hidden = !hasRepairs;
  if (hasRepairs) {
    if (!state.selectedRepair || !report.repairs.some((item) => item.id === state.selectedRepair)) state.selectedRepair = report.repairs[0].id;
    $("#repair-count").textContent = `${report.repairs.length} feasible`;
    $("#repair-list").innerHTML = report.repairs.map((item, index) => {
      const label = item.kind === "attenuate" ? "Parameter attenuation" : item.kind === "pair_rollback" ? "Pair rollback" : "Single rollback";
      const actions = item.actions || [];
      const actionNames = actions.map((action) => `${action.label} · Action #${action.action_number}`).join(" + ");
      const parameterChange = item.kind === "attenuate" && actions.length
        ? formatParameterChange(actions[0])
        : item.kind === "pair_rollback"
        ? "Remove both actions"
        : "Remove action";
      const selectedByObjective = index === 0 ? '<span class="flag objective-flag">Selected by repair objective</span>' : "";
      return `<li class="repair-row ${item.id === state.selectedRepair ? "selected" : ""}" data-repair="${item.id}"><div class="repair-top"><div class="repair-label"><b>${index + 1}. ${label}</b><span>${escapeHtml(actionNames)}</span><span>${escapeHtml(parameterChange)}</span><span>Tier ${item.tier} · ${actions.length || item.remove.length || 1} action intervention</span></div><span class="rank-value">${signed(item.restored_margin)}</span></div><div class="repair-values"><span>target margin <b>${signed(report.prediction.margin)} → ${signed(item.restored_margin)}</b></span><span>distortion <b>${item.distortion.toFixed(4)}</b></span><span>movement <b>${item.parameter_movement.toFixed(3)}</b></span></div><span class="flag">Recovered</span>${selectedByObjective}</li>`;
    }).join("");
  }
}

function renderParams() {
  const type = $("#edit-type").value;
  const defaults = {
    brightness: ["factor", .25, 1.75, .05, .65], contrast: ["factor", .25, 1.75, .05, .65],
    saturation: ["factor", 0, 2, .05, .4], blur: ["radius", 0, 12, .25, 5], jpeg: ["quality", 5, 95, 1, 24]
  };
  if (type === "crop") {
    $("#param-controls").innerHTML = `<div class="crop-grid">${[["x", .12], ["y", .12], ["width", .76], ["height", .76]].map(([key, value]) => `<label>${key}<input type="number" data-param="${key}" min="0" max="1" step="0.01" value="${value}"></label>`).join("")}</div>`;
    return;
  }
  const [key, min, max, step, value] = defaults[type];
  $("#param-controls").innerHTML = `<div class="parameter"><div class="parameter-head"><span>${key}</span><output>${value}</output></div><input type="range" data-param="${key}" min="${min}" max="${max}" step="${step}" value="${value}"></div>`;
  $("#param-controls input").addEventListener("input", (event) => { $("#param-controls output").textContent = event.target.value; });
}

function syncEditOptions() {
  const used = new Set(state.session.actions.map((action) => action.type));
  const select = $("#edit-type");
  [...select.options].forEach((option) => { option.disabled = used.has(option.value); });
  if (select.selectedOptions[0]?.disabled) {
    const available = [...select.options].find((option) => !option.disabled);
    if (available) select.value = available.value;
  }
  renderParams();
}

async function openDataset() {
  $("#dataset-modal").hidden = false;
  await refreshDataset(false);
}

async function refreshDataset(rescan = true) {
  try {
    state.dataset = rescan ? await api("/api/dataset/refresh", {}) : await api("/api/dataset");
    renderDatasetStatus();
    if (state.dataset.configured) await loadTraces();
  } catch (error) { toast(error.message); }
}

function renderDatasetStatus() {
  const data = state.dataset;
  if (!data?.configured) {
    const root = data?.expected_data_root || "data";
    $("#dataset-status").innerHTML = `<div class="dataset-missing">No catalog found. Create the expected folders, then press Refresh.<code>${escapeHtml(root)}/images/&lt;source_id&gt;<br>${escapeHtml(root)}/01-traces.jsonl</code></div>`;
    $("#trace-catalog").innerHTML = "";
    return;
  }
  $("#dataset-status").innerHTML = [
    ["Traces", data.trace_count], ["Images ready", data.available_images], ["Missing", data.missing_images], ["Invalid lines", data.invalid_lines]
  ].map(([label, value]) => `<div class="dataset-stat"><span>${label}</span><b>${value}</b></div>`).join("");
  $("#dataset-btn").textContent = `Dataset · ${data.available_images}/${data.trace_count}`;
}

async function loadTraces() {
  const query = encodeURIComponent($("#trace-search").value.trim());
  const payload = await api(`/api/traces?limit=500&q=${query}`);
  state.traces = payload.items;
  $("#trace-catalog").innerHTML = state.traces.map((trace) => `<li class="${trace.image_available ? "" : "unavailable"}" data-trace-id="${escapeHtml(trace.trace_id)}" data-available="${trace.image_available}"><div><b>${escapeHtml(trace.trace_id)}</b><small>${escapeHtml(trace.source_id)}</small></div><div class="trace-meta"><strong>GT ${trace.ground_truth_label}</strong>${trace.benchmark} · ${trace.length} edits</div></li>`).join("") || '<li class="unavailable"><div><b>No matching traces</b></div></li>';
}

async function loadFrozenTrace(traceId) {
  busy(true, "Loading frozen trace");
  try {
    state.session = await api("/api/traces/open", { trace_id: traceId });
    state.report = null; state.preview = null; state.selectedRepair = null;
    $("#dataset-modal").hidden = true;
    render();
    if (!state.session.dataset.eligible || !state.session.dataset.replay_valid) {
      toast(`${state.session.dataset.eligibility_reason}; ${state.session.dataset.replay_reason}`);
    }
  } catch (error) { toast(error.message); } finally { busy(false); }
}

async function addEdit() {
  const type = $("#edit-type").value;
  const params = Object.fromEntries($$("#param-controls [data-param]").map((input) => [input.dataset.param, Number(input.value)]));
  busy(true, "Appending and replaying edit");
  try {
    state.session = await api(`/api/sessions/${state.session.id}/actions`, { type, params });
    state.report = null; state.preview = null; render();
  } catch (error) { toast(error.message); } finally { busy(false); }
}

async function diagnose() {
  if (!state.session.actions.length) return toast("Add at least one action before diagnosis.");
  busy(true, "Enumerating counterfactuals");
  try {
    state.preview = null;
    const dataset = state.session.dataset;
    const allowExploratory = dataset.mode && (dataset.eligible !== true || dataset.replay_valid !== true);
    state.report = await api(`/api/sessions/${state.session.id}/diagnose`, { allow_exploratory: allowExploratory });
    render();
  } catch (error) { toast(error.message); } finally { busy(false); }
}

async function preview(body) {
  busy(true, "Rendering counterfactual preview");
  try {
    state.preview = await api(`/api/sessions/${state.session.id}/preview`, body);
    state.view = "current"; render();
  } catch (error) { toast(error.message); } finally { busy(false); }
}

async function applyRepair() {
  if (!state.selectedRepair) return;
  busy(true, "Applying repair to history");
  try {
    state.session = await api(`/api/sessions/${state.session.id}/apply-repair`, { candidate_id: state.selectedRepair });
    state.report = null; state.preview = null; state.selectedRepair = null; render(); toast("Repair applied to executable history.");
  } catch (error) { toast(error.message); } finally { busy(false); }
}

async function command(name) {
  busy(true, name === "undo" ? "Restoring history" : "Resetting trace");
  try {
    state.session = await api(`/api/sessions/${state.session.id}/${name}`, {});
    state.report = null; state.preview = null; state.selectedRepair = null; render();
  } catch (error) { toast(error.message); } finally { busy(false); }
}

function selectedRepair() { return state.report?.repairs.find((item) => item.id === state.selectedRepair); }
function signed(value) { return `${value >= 0 ? "+" : ""}${Number(value).toFixed(3)}`; }
function escapeHtml(value) { const node = document.createElement("span"); node.textContent = value; return node.innerHTML; }

$("#edit-type").addEventListener("change", renderParams);
$("#add-edit-btn").addEventListener("click", addEdit);
$("#diagnose-btn").addEventListener("click", diagnose);
$("#undo-btn").addEventListener("click", () => command("undo"));
$("#reset-btn").addEventListener("click", () => command("reset"));
$("#upload-btn").addEventListener("click", () => $("#file-input").click());
$("#dataset-btn").addEventListener("click", openDataset);
$("#dataset-close").addEventListener("click", () => { $("#dataset-modal").hidden = true; });
$("#dataset-refresh").addEventListener("click", () => refreshDataset(true));
$("#trace-search").addEventListener("input", () => { clearTimeout(state.searchTimer); state.searchTimer = setTimeout(loadTraces, 180); });
$("#trace-catalog").addEventListener("click", (event) => { const row = event.target.closest("[data-trace-id]"); if (row && row.dataset.available === "true") loadFrozenTrace(row.dataset.traceId); });
$("#dataset-modal").addEventListener("click", (event) => { if (event.target.id === "dataset-modal") event.currentTarget.hidden = true; });
$("#file-input").addEventListener("change", (event) => {
  const file = event.target.files[0]; if (!file) return;
  const reader = new FileReader(); reader.onload = () => createSession(reader.result, file.name); reader.readAsDataURL(file);
});
$$(".view-toggle button").forEach((button) => button.addEventListener("click", () => { state.view = button.dataset.view; renderImage(); }));
$$(".tabs button").forEach((button) => button.addEventListener("click", () => {
  $$(".tabs button").forEach((item) => item.classList.toggle("active", item === button));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`));
}));
$("#ranking-list").addEventListener("click", (event) => { const row = event.target.closest("[data-preview-action]"); if (row) preview({ remove: [row.dataset.previewAction] }); });
$("#interaction-list").addEventListener("click", (event) => { const row = event.target.closest("[data-preview-pair]"); if (row) preview({ remove: row.dataset.previewPair.split(",") }); });
$("#repair-list").addEventListener("click", (event) => { const row = event.target.closest("[data-repair]"); if (row) { state.selectedRepair = row.dataset.repair; renderReport(); } });
$("#preview-repair-btn").addEventListener("click", () => { const item = selectedRepair(); if (item) preview({ remove: item.remove, overrides: item.overrides }); });
$("#apply-repair-btn").addEventListener("click", applyRepair);

renderParams();
createSession();
