/* Reactor Advisor — page logic.  Talks to reactor_ui.py over JSON; draws the charts as inline SVG. */
"use strict";

const $ = (id) => document.getElementById(id);
let S = null;          // campaign state from /api/state
let A = null;          // last analysis
let editIndex = null;  // run being edited, or null for "add"
let lsRef = "next";
let setupDirty = false;          // typed in the campaign rail but not applied yet
let renderedSettings = null, renderedSpecies = null;

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({ error: "the server did not answer" }));
  if (!r.ok || j.error) throw new Error(j.error || `request failed (${r.status})`);
  return j;
}
function fmt(v, d = 1) { return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d); }
function money(v) { return v === null || v === undefined ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k);
  return e;
}
function setMsg(id, text, kind) { const m = $(id); m.textContent = text || ""; m.className = "msg" + (kind ? " " + kind : ""); }
function busy(on, text) {
  $("overlay").hidden = !on; $("overlay").setAttribute("aria-hidden", on ? "false" : "true");
  if (text) $("overlay-text").textContent = text;
  document.querySelectorAll("button.btn, .btn.file input").forEach((b) => { b.disabled = on; });   // no double submits
  if (!on && S) renderAdvice();                                                                     // restore the real enabled state
}
function closeForm() { $("run-form").hidden = true; editIndex = null; }
const knobLabel = (i) => S.meta.names[i];
const knobUnit = (i) => S.meta.units[i];
const knobDec = (i) => S.meta.dec[i];

// ------------------------------------------------------------------ theme
(function theme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("ra-theme"); } catch (e) {}
  if (saved) root.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("ra-theme", root.dataset.theme); } catch (e) {}
  });
})();

// ------------------------------------------------------------------ setup rail
function renderSetup() {
  const st = S.settings, m = S.meta;
  const cc = $("chem-choice");
  cc.innerHTML = "";
  for (const key of Object.keys(m.reactions)) {
    const lines = m.reactions[key].map((r) => `${r.eq}  <span class="role">(${r.role})</span>`).join("<br>");
    cc.append(el("label", { class: "chem-opt" },
      el("input", { type: "radio", name: "system", value: key, ...(key === st.system ? { checked: "" } : {}) }),
      el("b", {}, key === "simple" ? "Simple: two reactions" : "Complex: three reactions"),
      el("small", { html: lines })));
  }
  $("s-wcat").value = st.wcat; $("s-aspec").value = st.a_spec;
  ["P", "VA", "VB", "VI", "T"].forEach((k, i) => { $("s-step-" + k).value = st.max_step[i]; });
  $("s-cost").value = st.trial_cost ?? ""; $("s-hours").value = st.hours; $("s-seed").value = st.seed;
  $("s-spec-SC").value = st.spec.S_C ?? ""; $("s-spec-dT").value = st.spec.dT_ad ?? ""; $("s-remaining").value = st.remaining;
  $("s-deact").value = st.deactivation ?? 0;
  $("s-version").value = st.version || "v7";

  const L = $("limits");
  L.innerHTML = "";
  L.append(el("span", { class: "lh" }, "knob"), el("span", { class: "lh" }, "min"), el("span", { class: "lh" }, "max"));
  for (let i = 0; i < 5; i++) {
    const fx = st.fixed[String(i)];
    L.append(el("span", {}, `${knobLabel(i)} (${knobUnit(i)})`),
      el("input", { type: "number", id: "s-lo-" + i, step: "any", value: st.lo[i], "aria-label": `${knobLabel(i)} minimum` }),
      el("input", { type: "number", id: "s-hi-" + i, step: "any", value: st.hi[i], "aria-label": `${knobLabel(i)} maximum` }),
      el("label", { class: "fx" }, el("input", { type: "checkbox", id: "s-fx-" + i, "aria-label": `keep ${knobLabel(i)} fixed`, ...(fx !== undefined ? { checked: "" } : {}) }),
        "fixed at", el("input", { type: "number", id: "s-fxv-" + i, step: "any", value: fx !== undefined ? fx : "", placeholder: knobUnit(i), "aria-label": `${knobLabel(i)} fixed value` })));
  }
  const P = $("prices");
  P.innerHTML = "";
  for (const [k, lab] of Object.entries(m.price_labels)) {
    P.append(el("label", {}, lab, el("input", { type: "number", id: "s-price-" + k, step: "any", value: st.prices[k] })));
  }
  $("chem-badge").textContent = st.system;
}

async function applySettings() {
  const body = {
    system: document.querySelector('input[name="system"]:checked').value,
    wcat: +$("s-wcat").value, a_spec: +$("s-aspec").value,
    max_step: ["P", "VA", "VB", "VI", "T"].map((k) => +$("s-step-" + k).value),
    lo: [0, 1, 2, 3, 4].map((i) => +$("s-lo-" + i).value), hi: [0, 1, 2, 3, 4].map((i) => +$("s-hi-" + i).value),
    fixed: {}, prices: {}, trial_cost: $("s-cost").value === "" ? null : +$("s-cost").value, hours: +$("s-hours").value, seed: +$("s-seed").value,
    spec: { S_C: $("s-spec-SC").value === "" ? null : +$("s-spec-SC").value, dT_ad: $("s-spec-dT").value === "" ? null : +$("s-spec-dT").value },
    remaining: +$("s-remaining").value, deactivation: +($("s-deact").value || 0), version: $("s-version").value,
  };
  for (let i = 0; i < 5; i++) if ($("s-fx-" + i).checked) {
    if ($("s-fxv-" + i).value === "") { setMsg("setup-msg", `give a value for the fixed ${knobLabel(i).toLowerCase()}`, "err"); return false; }
    body.fixed[i] = +$("s-fxv-" + i).value;
  }
  for (const k of Object.keys(S.meta.price_labels)) body.prices[k] = +$("s-price-" + k).value;
  try {
    S = await api("/api/settings", body); A = null; setupDirty = false;
    renderAll(); setMsg("setup-msg", "Settings applied.", "ok");
    return true;
  } catch (e) { setMsg("setup-msg", e.message, "err"); return false; }
}
$("setup").addEventListener("submit", (ev) => { ev.preventDefault(); applySettings(); });
$("setup").addEventListener("input", () => { setupDirty = true; setMsg("setup-msg", "Changed, not applied yet.", "dirty"); });
document.querySelectorAll(".help").forEach((b) => b.addEventListener("click", () => { const t = $("help-" + b.dataset.help); t.hidden = !t.hidden; }));

// ------------------------------------------------------------------ runs table + form
function runInputs() {
  const ri = $("run-inputs"); ri.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    ri.append(el("label", {}, `${knobLabel(i)} `, el("span", { class: "unit" }, knobUnit(i)),
      el("input", { type: "number", id: "r-" + S.meta.knobs[i], step: "any", required: "", min: S.meta.box_lo[i], max: S.meta.box_hi[i], "aria-label": `${knobLabel(i)} of this run` })));
  }
  ri.append(el("label", {}, "Catalyst ", el("span", { class: "unit" }, "g"), el("input", { type: "number", id: "r-W", step: "any", required: "", min: "0.001", "aria-label": "catalyst mass of this run" })));
  ri.append(el("label", {}, "Run lasted ", el("span", { class: "unit" }, "h"), el("input", { type: "number", id: "r-hours", step: "any", required: "", min: "0.1", "aria-label": "duration of this run in hours" })));
  const gi = $("gc-inputs"); gi.innerHTML = "";
  for (const sp of S.meta.species) {
    gi.append(el("label", {}, sp === "I" ? "I (inert)" : sp, el("span", { class: "unit" }, "% or fraction"),
      el("input", { type: "number", id: "r-gc-" + sp, step: "any", min: "0", required: "", "aria-label": `GC reading for ${sp}` })));
  }
}
function openForm(run) {
  editIndex = run ? run.i : null;
  $("run-form-title").textContent = run ? `Edit run ${run.i + 1}` : "Add a run";
  const last = S.runs[S.runs.length - 1];
  const src = run || last || null;
  for (let i = 0; i < 5; i++) $("r-" + S.meta.knobs[i]).value = src ? src[S.meta.knobs[i]] : "";
  $("r-W").value = run ? run.W : (last ? last.W : S.settings.wcat);
  $("r-hours").value = run ? run.hours : (last ? last.hours : S.meta.hours_per_run);
  S.meta.species.forEach((sp, k) => { $("r-gc-" + sp).value = run ? (run.gc[k] * 100).toFixed(3) : ""; });
  setMsg("run-msg", "");
  $("run-form").hidden = false;
  $("r-P").focus();
}
$("btn-add").addEventListener("click", () => openForm(null));
$("run-cancel").addEventListener("click", () => { $("run-form").hidden = true; editIndex = null; });
$("run-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = { gc: S.meta.species.map((sp) => +$("r-gc-" + sp).value), W: +$("r-W").value, hours: +$("r-hours").value, source: "manual" };
  for (let i = 0; i < 5; i++) body[S.meta.knobs[i]] = +$("r-" + S.meta.knobs[i]).value;
  try {
    const wasEdit = editIndex !== null;
    if (!wasEdit) S = await api("/api/runs/add", body);
    else S = await api("/api/runs/update", { ...body, i: editIndex });
    A = null; closeForm(); renderAll();
    setMsg("runs-msg", wasEdit ? "Run updated." : "Run saved.", "ok");
  } catch (e) { setMsg("run-msg", e.message, "err"); }
});
$("file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  try {
    const text = await f.text();
    const replace = S.runs.length === 0 || confirm("Replace the runs in the table with the file? (Cancel = add the file's runs after the existing ones)");
    S = await api("/api/runs/import", { text, replace }); A = null; closeForm(); renderAll();
    setMsg("runs-msg", `Imported ${S.n} run(s) from ${f.name}.`, "ok");
  } catch (e) { setMsg("runs-msg", e.message, "err"); }
  ev.target.value = "";
});
async function loadExample() {
  try {
    busy(true, "Running the 5 screening runs on the virtual reactor…");
    S = await api("/api/example", { seed: +$("s-seed").value || 0 }); A = null; closeForm(); renderAll();
    setMsg("runs-msg", "Loaded the 5 screening runs an operator does first (temperature ladder, pressure up, more B).", "ok");
  } catch (e) { setMsg("runs-msg", e.message, "err"); } finally { busy(false); }
}
$("btn-example").addEventListener("click", loadExample);
$("btn-example-2").addEventListener("click", loadExample);
$("btn-clear").addEventListener("click", async () => {
  if (S.runs.length && !confirm("Remove all runs from the table?")) return;
  S = await api("/api/runs/clear", {}); A = null; closeForm(); renderAll(); setMsg("runs-msg", "");
});

function renderRuns() {
  const T = $("runs-table");
  const sp = S.meta.species;
  const head = T.querySelector("thead"); head.innerHTML = "";
  const tr1 = el("tr", {}, el("th", { class: "first", rowspan: 2 }, "Run"));
  for (let i = 0; i < 5; i++) tr1.append(el("th", { rowspan: 2 }, `${knobLabel(i)} (${knobUnit(i)})`));
  tr1.append(el("th", { rowspan: 2 }, "Cat. (g)"), el("th", { rowspan: 2 }, "Hours"), el("th", { class: "grp", colspan: sp.length }, "GC reading (%)"),
    el("th", { class: "grp", rowspan: 2 }, "Conv."), el("th", { rowspan: 2 }, "Select."), el("th", { rowspan: 2, title: "adiabatic temperature rise implied by the GC reading" }, "ΔT ad (K)"),
    el("th", { rowspan: 2 }, "Profit /h"), el("th", { rowspan: 2 }, "Model /h"), el("th", { rowspan: 2 }, ""));
  const tr2 = el("tr", {}); for (const s of sp) tr2.append(el("th", {}, s));
  head.append(tr1, tr2);
  const body = T.querySelector("tbody"); body.innerHTML = "";
  S.runs.forEach((r, idx) => {
    const tr = el("tr", { class: (idx === S.best_idx ? "best " : "") + (idx === S.runs.length - 1 ? "last" : "") });
    tr.append(el("td", { class: "first num" }, `${r.i + 1} `, el("span", { class: "src", title: r.source === "virtual" ? "measured on the virtual reactor" : "entered by you" }, r.source === "virtual" ? "sim" : "")));
    for (let i = 0; i < 5; i++) tr.append(el("td", { class: "num" }, fmt(r[S.meta.knobs[i]], knobDec(i))));
    tr.append(el("td", { class: "num" }, fmt(r.W, 2)), el("td", { class: "num" }, fmt(r.hours, 0)));
    r.gc.forEach((g) => tr.append(el("td", { class: "gc" }, fmt(100 * g, 1))));
    tr.append(el("td", { class: "num" }, fmt(r.XA, 2)), el("td", { class: "num" }, fmt(r.SC, 2)), el("td", { class: "num" }, fmt(r.dT, 0)));
    tr.append(el("td", { class: "num profit" }, el("span", { class: r.profit < 0 ? "neg" : "" }, fmt(r.profit, 1))));
    tr.append(el("td", { class: "num" }, r.pred === null ? "–" : fmt(r.pred, 1)));
    tr.append(el("td", {}, el("button", { class: "row-btn", type: "button", title: "edit", "aria-label": `edit run ${r.i + 1}`, onclick: () => openForm(r) }, "✎"),
      el("button", { class: "row-btn", type: "button", title: "delete", "aria-label": `delete run ${r.i + 1}`, onclick: async () => {
        if (!confirm(`Delete run ${r.i + 1}?`)) return;
        S = await api("/api/runs/delete", { i: r.i }); A = null; renderAll();
      } }, "✕")));
    body.append(tr);
  });
  $("runs-note").textContent = S.n === 0 ? "" : `${S.n} run(s). Green = best measured profit so far; the orange edge marks your last run (the next suggestion starts from it).`;
}

// ------------------------------------------------------------------ advice
async function advise(ref) {
  if (S.n < 3) { setMsg("advice-msg", "Add at least 3 runs first (or load the 5 screening runs).", "err"); return; }
  if (setupDirty && !(await applySettings())) { setMsg("advice-msg", "Fix the campaign settings first (see the message under Apply settings).", "err"); return; }
  const v7 = S.settings.version === "v7";
  const base = "Fitting the chemistry to your GC readings and simulating candidate runs… (the first time takes 10–20 s, later ones a few seconds)"
    + (v7 ? " v7 also checks its reactions against the GC readings and may search for better ones (up to a minute)." : "");
  let poll = null;
  try {
    busy(true, base);
    if (v7) poll = setInterval(async () => {
      try { const p = (await api("/api/progress")).progress; if (p && p.message) $("overlay-text").textContent = `Checking the reactions (${fmt(p.elapsed, 0)} s): ${p.message}`; } catch (e) {}
    }, 1500);
    const out = await api("/api/analyse", { ref: ref || lsRef });
    A = out.analysis; S = out.state;
    renderAll();
    setMsg("advice-msg", `Advice ready (${A.seconds} s).`, "ok");
  } catch (e) { setMsg("advice-msg", e.message, "err"); } finally { if (poll) clearInterval(poll); busy(false); }
}
$("btn-advise").addEventListener("click", () => advise());
async function simulateDays(n) {
  for (let d = 0; d < n; d++) {
    try {
      busy(true, n === 1 ? "Running the suggested settings on the virtual reactor…" : `Day ${d + 1} of ${n}: advising, then running it on the virtual reactor…`);
      S = await api("/api/simulate", {}); A = null;
      renderAll();
    } catch (e) { setMsg("advice-msg", e.message, "err"); busy(false); return; }
  }
  busy(false);
  await advise();
}
$("btn-sim").addEventListener("click", () => simulateDays(1));
$("btn-sim5").addEventListener("click", () => simulateDays(5));

function renderAdvice() {
  const has = !!A;
  $("advice-empty").hidden = has || S.n >= 3;
  $("advice-body").hidden = !has;
  $("btn-advise").disabled = S.n < 3;
  $("btn-sim").disabled = S.n < 3; $("btn-sim5").disabled = S.n < 3;
  $("dl-next").setAttribute("aria-disabled", has ? "false" : "true");
  if (!has) {
    if (S.n >= 3) setMsg("advice-msg", S.analysis_ready ? "" : `${S.n} runs in the table. Press Get advice.`);
    return;
  }
  const nx = A.next, tiles = $("tiles");
  tiles.innerHTML = "";
  nx.moves.forEach((m, i) => {
    const fixed = S.settings.fixed[String(i)] !== undefined;
    const d = m.changed ? `${m.delta > 0 ? "▲ +" : "▼ "}${fmt(m.delta, knobDec(i))} ${m.unit} from last run` : (fixed ? "fixed" : "same as last run");
    tiles.append(el("div", { class: "tile" + (m.changed ? " changed" : "") },
      el("span", { class: "label" }, m.name),
      el("span", { class: "v" }, fmt(m.to, knobDec(i)), el("small", {}, m.unit)),
      el("span", { class: "d " + (m.changed ? (m.delta > 0 ? "up" : "down") : (fixed ? "fixed" : "")) }, d)));
  });
  $("next-pred").innerHTML = `${fmt(nx.pred, 1)} <small>/ h</small>`;
  $("next-range").textContent = `likely between ${fmt(nx.lo, 1)} and ${fmt(nx.hi, 1)}`;
  $("next-why").textContent = nx.why + (nx.same_as_last && nx.kind !== "stay" ? " (It repeats your last run to confirm it.)" : "")
    + (Object.keys(nx.spec || {}).length ? ` Meets your spec with ${fmt(100 * nx.p_spec, 0)} % probability.` : "");

  const pl = A.plan;
  $("plan-steps").textContent = pl.steps_away === 0 ? "you are there" : `${pl.steps_away} run${pl.steps_away > 1 ? "s" : ""} away`;
  $("plan-pred").textContent = `The model expects about ${fmt(pl.pred, 1)} / h there (likely ${fmt(pl.lo, 1)}–${fmt(pl.hi, 1)}).`;
  $("plan-list").innerHTML = ""; $("today-list").innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const k = S.meta.knobs[i];
    $("plan-list").append(el("li", {}, el("span", {}, knobLabel(i)), el("span", {}, `${fmt(pl[k], knobDec(i))} ${knobUnit(i)}`)));
    $("today-list").append(el("li", {}, el("span", {}, knobLabel(i)), el("span", {}, `${fmt(A.today[k], knobDec(i))} ${knobUnit(i)}`)));
  }
  $("today-run").textContent = `${A.best_idx + 1}`;
  const w = A.worth;
  $("worth-pill").textContent = w.verdict === "yes" ? "YES, run it" : "PROBABLY NOT";
  $("worth-pill").className = "pill " + w.verdict;
  $("worth-text").textContent = w.text;
  const wl = $("worth-list"); wl.innerHTML = "";
  wl.append(el("li", {}, el("span", {}, "This one run, expected"), el("span", {}, `+${fmt(w.ei, 2)} / h`)));
  wl.append(el("li", {}, el("span", {}, "Reaching the predicted best"), el("span", {}, `+${fmt(w.far, 1)} / h`)));
  if (w.yearly !== undefined) {
    wl.append(el("li", {}, el("span", {}, "Worth per year"), el("span", {}, money(w.yearly))));
    wl.append(el("li", {}, el("span", {}, `Cost of ${w.runs} run(s)`), el("span", {}, money(w.cost))));
  }
}

// ------------------------------------------------------------------ chemistry
const RXC = ["var(--c1)", "var(--c2)", "var(--c3)"];
function renderLearned() {
  $("learned-empty").hidden = !!A; $("learned-body").hidden = !A;
  if (!A) return;
  const ch = A.chemistry;
  const rc = $("rx-cards"); rc.innerHTML = "";
  ch.reactions.forEach((r, j) => {
    rc.append(el("div", { class: "rx", style: `--rxc:${RXC[j]}` },
      el("span", { class: "label" }, r.name), el("span", { class: "eq" }, r.eq), el("span", { class: "role" }, r.role),
      el("span", { class: "ea", html: `activation energy <b>${fmt(r.Ea, 0)}</b> kJ/mol <span class="hint">(likely ${fmt(r.Ea_lo, 0)}–${fmt(r.Ea_hi, 0)}${r.Ea_true !== null ? `; true ${fmt(r.Ea_true, 0)}` : ""})</span>` }),
      el("span", { class: "hint" }, `near ${fmt(ch.at_T, 0)} °C every +10 °C makes it ${fmt(r.factor10, 2)}× faster`)));
  });
  const nw = ch.network;
  $("net-box").hidden = !nw;
  if (nw) {
    $("net-lines").innerHTML = ""; nw.lines.forEach((l) => $("net-lines").append(el("li", {}, l.replace(/^\d+\.\s+/, ""))));
    $("net-changed").hidden = !nw.changed;
    $("net-given").innerHTML = ""; nw.given.forEach((l) => $("net-given").append(el("li", {}, l.replace(/^\d+\.\s+/, ""))));
    const nn = $("net-notes"); nn.innerHTML = "";
    if (nw.check) nn.append(el("li", {}, `Last check: ${nw.check}`));
    nw.revisions.forEach((r) => nn.append(el("li", {}, r.words || `After run ${r.run}: ${r.change}`)));
    if (!nw.revisions.length) nn.append(el("li", {}, "No change made so far: the reactions you set up explain the GC readings, or the evidence against them is not yet strong enough to change them."));
    const last = nw.log.length ? nw.log[nw.log.length - 1] : null;
    if (last && last.outcome !== "revised" && last.words) nn.append(el("li", {}, last.words));
  }
  const notes = $("rx-notes"); notes.innerHTML = "";
  for (const n of ch.notes) notes.append(el("li", {}, n));
  notes.append(el("li", {}, `How well this chemistry explains your GC readings: typical miss ${fmt(ch.miss_pct, 2)} percentage points of composition. Your runs cover ${fmt(ch.T_range[0], 0)}–${fmt(ch.T_range[1], 0)} °C; activation energies are only pinned down inside that range.`));
  $("arr-truth-note").textContent = ch.truth_known ? "; solid = the virtual reactor's true chemistry (unknown on a real reactor)" : "";
  drawArrhenius(); drawParity();
}

// ------------------------------------------------------------------ charts (inline SVG)
function scaleLin(d0, d1, r0, r1) { const k = (r1 - r0) / ((d1 - d0) || 1); return (v) => r0 + (v - d0) * k; }
function niceTicks(lo, hi, n = 5) {
  // ticks that COVER [lo, hi]: first tick <= lo, last tick >= hi
  const span = (hi - lo) || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => span / s <= n + 1) || p * 10;
  const out = []; for (let v = Math.floor(lo / step) * step; v <= hi + step - 1e-9; v += step) { out.push(+v.toFixed(10)); if (v >= hi) break; }
  return out;
}
function frame(W, H, m, xs, ys, xlab, ylab, xTicks, yTicks) {
  const X = scaleLin(xs[0], xs[1], m.l, W - m.r), Y = scaleLin(ys[0], ys[1], H - m.b, m.t);
  let s = "";
  for (const t of yTicks) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${t}</text>`;
  for (const t of xTicks) s += `<text x="${X(t)}" y="${H - m.b + 15}" text-anchor="middle">${t}</text>`;
  s += `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
  if (xlab) s += `<text class="t" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${xlab}</text>`;
  if (ylab) s += `<text class="t" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${ylab}</text>`;
  return { X, Y, s };
}
function path(pts) { let d = "", pen = false; for (const [x, y] of pts) { if (y === null || Number.isNaN(y)) { pen = false; continue; } d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; } return d; }
function band(xs, lo, hi, X, Y) {
  const up = [], dn = [];
  xs.forEach((x, i) => { if (lo[i] !== null && hi[i] !== null) { up.push([X(x), Y(hi[i])]); dn.unshift([X(x), Y(lo[i])]); } });
  if (!up.length) return "";
  return path(up) + " " + path(dn).replace(/^M/, "L") + " Z";
}
const tip = $("tip");
function hoverLayer(container, xsPix, lines) {
  // lines(i) -> text for the nearest index; container.svg must be rendered
  const svg = container.querySelector("svg");
  const move = (ev) => {
    const r = svg.getBoundingClientRect(); const vb = svg.viewBox.baseVal;
    const px = (ev.clientX - r.left) * vb.width / r.width;
    let best = 0, bd = 1e9; xsPix.forEach((x, i) => { const d = Math.abs(x - px); if (d < bd) { bd = d; best = i; } });
    const t = lines(best); if (!t) { tip.hidden = true; return; }
    tip.innerHTML = t; tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tw - 8) + "px";
    tip.style.top = Math.max(ev.clientY - th - 12, 8) + "px";
  };
  svg.addEventListener("mousemove", move);
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
}

function drawProfit() {
  const se = A.series, n = se.measured.length, W = 720, H = 240, m = { l: 48, r: 14, t: 12, b: 36 };
  const all = [...se.measured, ...se.best, ...se.model];
  const ylo = Math.min(...all), yhi = Math.max(...all), pad = (yhi - ylo || 10) * .06;
  const yT = niceTicks(ylo - pad, yhi + pad), xT = niceTicks(1, n, Math.min(n, 8)).filter((t) => Number.isInteger(t) && t >= 1 && t <= n);
  const f = frame(W, H, m, [0.5, n + 0.5], [yT[0], yT[yT.length - 1]], "run number", "profit, money / h", xT, yT);
  const idx = [...Array(n).keys()];
  let s = f.s;
  s += `<path d="${path(idx.map((i) => [f.X(i + 1), f.Y(se.best[i])]))}" fill="none" style="stroke:var(--c2)" stroke-width="2"/>`;
  s += `<path d="${path(idx.map((i) => [f.X(i + 1), f.Y(se.model[i])]))}" fill="none" style="stroke:var(--c3)" stroke-width="2" stroke-dasharray="5 4"/>`;
  idx.forEach((i) => { s += `<circle cx="${f.X(i + 1)}" cy="${f.Y(se.measured[i])}" r="4" style="fill:var(--c1);stroke:var(--surface)" stroke-width="1.5"/>`; });
  if (n >= 5) s += `<rect x="${f.X(0.5)}" y="${m.t}" width="${f.X(5.5) - f.X(0.5)}" height="${H - m.b - m.t}" style="fill:var(--ink)" opacity=".05"/>`;
  const c = $("chart-profit");
  c.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="profit of every run">${s}</svg>
    <div class="legend"><span class="dot" style="--lc:var(--c1)">measured profit</span><span style="--lc:var(--c2)">best so far</span><span class="dash" style="--lc:var(--c3)">model's estimate</span></div>`;
  hoverLayer(c, idx.map((i) => f.X(i + 1)), (i) => `run ${i + 1}<br>measured ${fmt(se.measured[i], 1)} / h<br>model ${fmt(se.model[i], 1)} ± ${fmt(se.model_sd[i], 1)}<br>best so far ${fmt(se.best[i], 1)}`);
}

function drawArrhenius() {
  const ar = A.arrhenius, W = 520, H = 280, m = { l: 46, r: 14, t: 12, b: 38 };
  let lo = Infinity, hi = -Infinity;
  for (const r of ar.reactions) for (const arr of [r.lo, r.hi, r.truth || []]) for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const yT = niceTicks(lo, hi), xT = niceTicks(ar.x[ar.x.length - 1], ar.x[0], 5);
  const f = frame(W, H, m, [ar.x[ar.x.length - 1], ar.x[0]], [yT[0], yT[yT.length - 1]], "1000 / T (1/K)   ← hotter", "log10 rate constant", xT, yT);
  let s = f.s;
  ar.reactions.forEach((r, j) => {
    s += `<path d="${band(ar.x, r.lo, r.hi, f.X, f.Y)}" style="fill:${RXC[j]}" opacity=".16"/>`;
    if (r.truth) s += `<path d="${path(ar.x.map((x, i) => [f.X(x), f.Y(r.truth[i])]))}" fill="none" style="stroke:${RXC[j]}" stroke-width="2"/>`;
    s += `<path d="${path(ar.x.map((x, i) => [f.X(x), f.Y(r.learned[i])]))}" fill="none" style="stroke:${RXC[j]}" stroke-width="2" stroke-dasharray="6 4"/>`;
  });
  const c = $("chart-arr");
  c.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Arrhenius plot of learned rate constants">${s}</svg>
    <div class="legend">${ar.reactions.map((r, j) => `<span class="dash" style="--lc:${RXC[j]}">${r.eq}</span>`).join("")}</div>`;
  hoverLayer(c, ar.x.map((x) => f.X(x)), (i) => `${ar.T_C[i]} °C<br>` + ar.reactions.map((r) => `${r.eq}: ${fmt(r.learned[i], 2)}${r.truth ? ` (true ${fmt(r.truth[i], 2)})` : ""}`).join("<br>"));
}

function drawParity() { parityChart($("chart-parity"), A.parity); }
function parityChart(c, pa) {
  const W = 520, H = 280, m = { l: 46, r: 14, t: 12, b: 38 };
  let mx = 0; for (const row of [...pa.meas, ...pa.pred]) for (const v of row) if (v > mx) mx = v;
  mx = Math.max(mx * 1.08, 0.05);
  const T = niceTicks(0, mx, 5);
  const f = frame(W, H, m, [0, mx], [0, mx], "GC reading (mole fraction)", "model prediction", T, T);
  let s = f.s + `<line x1="${f.X(0)}" y1="${f.Y(0)}" x2="${f.X(mx)}" y2="${f.Y(mx)}" class="ax" stroke-dasharray="3 3"/>`;
  const pts = [];
  pa.meas.forEach((row, r) => row.forEach((v, k) => { pts.push({ x: f.X(v), y: f.Y(pa.pred[r][k]), r, k, v, p: pa.pred[r][k] }); s += `<circle cx="${f.X(v)}" cy="${f.Y(pa.pred[r][k])}" r="4" style="fill:var(--c${k + 1});stroke:var(--surface)" stroke-width="1.2"/>`; }));
  c.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="GC readings against model predictions">${s}</svg>
    <div class="legend">${pa.species.map((sp, k) => `<span class="dot" style="--lc:var(--c${k + 1})">${sp}</span>`).join("")}</div>`;
  const svg = c.querySelector("svg");
  svg.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(); const vb = svg.viewBox.baseVal;
    const px = (ev.clientX - r.left) * vb.width / r.width, py = (ev.clientY - r.top) * vb.height / r.height;
    let b = null, bd = 100; for (const p of pts) { const d = Math.hypot(p.x - px, p.y - py); if (d < bd) { bd = d; b = p; } }
    if (!b) { tip.hidden = true; return; }
    tip.innerHTML = `run ${b.r + 1}, species ${pa.species[b.k]}<br>GC ${fmt(100 * b.v, 2)} %, model ${fmt(100 * b.p, 2)} %`; tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = Math.max(ev.clientY - tip.offsetHeight - 12, 8) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
}

function drawLandscape() {
  const L = $("landscape"); L.innerHTML = "";
  const refName = { next: "the next run", best: "your best run", plan: "the predicted best" }[A.landscape.ref];
  A.landscape.knobs.forEach((k, i) => {
    const W = 300, H = 190, m = { l: 40, r: 10, t: 10, b: 32 };
    const idx = k.mu.map((v, j) => v === null ? null : j).filter((j) => j !== null);
    if (!idx.length) return;
    const top = Math.max(...idx.map((j) => k.mu[j] + k.sd[j]));
    const floor = Math.max(Math.min(...idx.map((j) => k.mu[j] - k.sd[j])), top - Math.max(30, 2 * Math.abs(top)));   // keep the hill readable
    const yT = niceTicks(floor, top, 4), xT = niceTicks(k.x[0], k.x[k.x.length - 1], 4).filter((t) => t >= k.x[0] - 1e-9 && t <= k.x[k.x.length - 1] + 1e-9);
    const f = frame(W, H, m, [k.x[0], k.x[k.x.length - 1]], [yT[0], yT[yT.length - 1]], `${k.name} (${k.unit})`, "", xT, yT);
    const cid = "clip-" + k.knob;
    let s = `<clipPath id="${cid}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></clipPath>` + f.s;
    s += `<rect x="${f.X(k.reach[0])}" y="${m.t}" width="${Math.max(f.X(k.reach[1]) - f.X(k.reach[0]), 1)}" height="${H - m.b - m.t}" style="fill:var(--accent)" opacity=".12"/>`;
    const loB = k.mu.map((v, j) => v === null ? null : v - k.sd[j]), hiB = k.mu.map((v, j) => v === null ? null : v + k.sd[j]);
    s += `<g clip-path="url(#${cid})"><path d="${band(k.x, loB, hiB, f.X, f.Y)}" style="fill:var(--c1)" opacity=".15"/>`;
    s += `<path d="${path(k.x.map((x, j) => [f.X(x), k.mu[j] === null ? null : f.Y(k.mu[j])]))}" fill="none" style="stroke:var(--c1)" stroke-width="2"/></g>`;
    s += `<line x1="${f.X(k.cur)}" x2="${f.X(k.cur)}" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--accent)" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    const d = el("div", { class: "ls" }, el("h4", {}, k.name, el("small", {}, k.fixed ? " · fixed" : "")));
    const c = el("div", { class: "chart" }); c.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="predicted profit against ${k.name}">${s}</svg>`;
    d.append(c); L.append(d);
    hoverLayer(c, k.x.map((x) => f.X(x)), (j) => k.mu[j] === null ? "" : `${k.name} ${fmt(k.x[j], knobDec(i))} ${k.unit}<br>expected ${fmt(k.mu[j], 1)} ± ${fmt(k.sd[j], 1)} / h`);
  });
  L.append(el("div", { class: "legend", style: "grid-column:1/-1" },
    el("span", { style: "--lc:var(--c1)" }, "expected profit (band = uncertainty)"),
    el("span", { class: "dash", style: "--lc:var(--accent)" }, `value at ${refName}`),
    el("span", { style: "--lc:var(--accent);opacity:.5" }, "reachable from the last run in one step")));
}
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", async () => {
  lsRef = b.dataset.ref;
  document.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
  if (!A) return;
  try {
    busy(true, "Recomputing the what-if charts…");
    const out = await api("/api/landscape", { ref: lsRef });
    A.landscape = out.landscape; drawLandscape();
  } catch (e) { setMsg("advice-msg", e.message, "err"); } finally { busy(false); }
}));

function renderExplore() {
  $("explore-empty").hidden = !!A; $("explore-body").hidden = !A;
  if (!A) return;
  drawProfit(); drawLandscape();
}

// ------------------------------------------------------------------ optional: profit vs selectivity trade-off
let TR = null;           // the curve, for the advice it was drawn from
function renderTradeoff() {
  if (TR && TR.forA !== A) TR = null;              // the advice changed: the curve no longer applies
  $("tradeoff-body").hidden = !TR;
  $("btn-tradeoff").textContent = TR ? "Draw the curve again" : "Show the trade-off curve";
  if (!TR) return;
  const T = TR.res;
  drawTradeoff(T);
  const nl = $("tradeoff-notes"); nl.innerHTML = "";
  for (const n of T.explanation || []) nl.append(el("li", {}, n));
  const tb = $("tradeoff-table");
  tb.querySelector("thead").innerHTML = "";
  tb.querySelector("thead").append(el("tr", {}, ...["", "Selectivity", "Profit / h", "Conversion", "P (bar)", "A", "B", "Inert (mL/min)", "T (°C)", "Runs away", ""].map((h) => el("th", {}, h))));
  const body = tb.querySelector("tbody"); body.innerHTML = "";
  T.points.forEach((p, i) => {
    const tag = p.label ? el("span", { class: "pill" + (p.label === "balanced" ? " yes" : "") }, p.label) : "";
    body.append(el("tr", {},
      el("td", { class: "first" }, tag, p.extrapolation ? el("span", { class: "src", title: "far from every run you have made: the model is guessing more here" }, " far from your runs") : ""),
      el("td", { class: "num" }, `${fmt(100 * p.selectivity, 1)} %`, el("span", { class: "src" }, ` ${fmt(100 * p.selectivity_lo, 0)}–${fmt(100 * p.selectivity_hi, 0)}`)),
      el("td", { class: "num profit" }, fmt(p.profit, 1), el("span", { class: "src" }, ` ${fmt(p.profit_lo, 0)}–${fmt(p.profit_hi, 0)}`)),
      el("td", { class: "num" }, `${fmt(100 * p.conversion, 0)} %`),
      el("td", { class: "num" }, fmt(p.P, 1)), el("td", { class: "num" }, fmt(p.V_A, 1)), el("td", { class: "num" }, fmt(p.V_B, 1)),
      el("td", { class: "num" }, fmt(p.V_I, 1)), el("td", { class: "num" }, fmt(p.T_C, 0)), el("td", { class: "num" }, String(p.steps_away)),
      el("td", {}, el("button", { type: "button", class: "btn seg-btn", title: `set the minimum selectivity to ${fmt(100 * (p.min_selectivity_spec ?? p.selectivity), 1)} % and get advice again`, onclick: () => aimAt(i) }, "Aim here"))));
  });
}
function drawTradeoff(T) {
  const W = 720, H = 300, m = { l: 52, r: 16, t: 14, b: 40 };
  // the axes follow the curve; a run far off it (e.g. a cold screening run) would squeeze the curve into a corner
  const xs = [...T.curve.selectivity, ...T.points.map((p) => p.selectivity)].filter((v) => v !== null).map((v) => 100 * v);
  const ys = [...T.curve.profit, ...T.points.flatMap((p) => [p.profit_lo, p.profit_hi])].filter((v) => v !== null);
  const xpad = Math.max((Math.max(...xs) - Math.min(...xs)) * 0.08, 0.5);
  const xT = niceTicks(Math.max(0, Math.min(...xs) - xpad), Math.min(100, Math.max(...xs) + xpad), 6), yT = niceTicks(Math.min(...ys), Math.max(...ys), 5);
  const inside = (r) => r && r.selectivity !== null && r.profit_measured !== null && 100 * r.selectivity >= xT[0] && 100 * r.selectivity <= xT[xT.length - 1] && r.profit_measured >= yT[0] && r.profit_measured <= yT[yT.length - 1];
  const ref = [T.reference.best_run, T.reference.last_run].map((r) => (inside(r) ? r : null));
  const f = frame(W, H, m, [xT[0], xT[xT.length - 1]], [yT[0], yT[yT.length - 1]], `selectivity to ${T.product || "C"} (%)   purer →`, "profit, money / h", xT, yT);
  let s = f.s;
  s += `<path d="${path(T.curve.selectivity.map((x, i) => [f.X(100 * x), T.curve.profit[i] === null ? null : f.Y(T.curve.profit[i])]))}" fill="none" style="stroke:var(--c1)" stroke-width="2"/>`;
  const pts = [];
  T.points.forEach((p, i) => {
    const x = f.X(100 * p.selectivity), y = f.Y(p.profit);
    s += `<line x1="${x}" x2="${x}" y1="${f.Y(p.profit_lo)}" y2="${f.Y(p.profit_hi)}" style="stroke:var(--c1)" stroke-width="1.5" opacity=".45"/>`;
    const mark = p.label === "balanced" ? "var(--c3)" : p.label ? "var(--c2)" : "var(--c1)";
    s += `<circle cx="${x}" cy="${y}" r="${p.label ? 6 : 4.5}" style="fill:${mark};stroke:var(--surface)" stroke-width="2"/>`;
    if (p.label) {
      const sel = p.label === "most selective";
      s += `<text x="${x + (sel ? -10 : 10)}" y="${y + (sel ? 18 : -10)}" text-anchor="${sel ? "end" : "start"}" class="t">${p.label}</text>`;
    }
    pts.push({ x, y, p, i });
  });
  ref.forEach((r, k) => {
    if (!r) return;
    const x = f.X(100 * r.selectivity), y = f.Y(r.profit_measured);
    s += `<rect x="${x - 5}" y="${y - 5}" width="10" height="10" style="fill:${k === 0 ? "var(--c4)" : "var(--surface)"};stroke:var(--c4)" stroke-width="2"/>`;
  });
  const out = [T.reference.best_run, T.reference.last_run].map((r, k) => (r && !ref[k] && r.selectivity !== null ? `${k === 0 ? "best" : "last"} run ${r.run}: ${fmt(100 * r.selectivity, 0)} %, ${fmt(r.profit_measured, 1)} / h` : null)).filter(Boolean);
  const c = $("chart-tradeoff");
  c.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="predicted profit against selectivity">${s}</svg>
    <div class="legend"><span style="--lc:var(--c1)">best profit expected at each selectivity</span><span class="dot" style="--lc:var(--c2)">most profitable / most selective</span><span class="dot" style="--lc:var(--c3)">balanced (the knee)</span><span class="dot" style="--lc:var(--c4)">your best run (filled) and last run, measured${out.length ? ` (outside the plotted range: ${out.join("; ")})` : ""}</span></div>`;
  const svg = c.querySelector("svg");
  svg.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    const px = (ev.clientX - r.left) * vb.width / r.width, py = (ev.clientY - r.top) * vb.height / r.height;
    let b = null, bd = 60; for (const q of pts) { const d = Math.hypot(q.x - px, q.y - py); if (d < bd) { bd = d; b = q; } }
    if (!b) { tip.hidden = true; return; }
    const p = b.p;
    tip.innerHTML = `${p.label ? p.label + "<br>" : ""}selectivity ${fmt(100 * p.selectivity, 1)} %<br>profit ${fmt(p.profit, 1)} / h (${fmt(p.profit_lo, 1)}–${fmt(p.profit_hi, 1)})<br>${fmt(p.P, 1)} bar, ${fmt(p.T_C, 0)} °C, A ${fmt(p.V_A, 1)}, B ${fmt(p.V_B, 1)}, inert ${fmt(p.V_I, 1)} mL/min<br>${p.steps_away} run(s) away`;
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = Math.max(ev.clientY - tip.offsetHeight - 12, 8) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
}
$("btn-tradeoff").addEventListener("click", async () => {
  if (!A) { setMsg("tradeoff-msg", "Get advice first.", "err"); return; }
  try {
    busy(true, "Finding the best profit at every selectivity… (a few seconds)");
    const out = await api("/api/tradeoff", {});
    TR = { res: out.tradeoff, forA: A };
    renderTradeoff();
    setMsg("tradeoff-msg", `Curve ready (${out.tradeoff.seconds} s). It changes nothing unless you press Aim here.`, "ok");
  } catch (e) { setMsg("tradeoff-msg", e.message, "err"); } finally { busy(false); }
});
async function aimAt(i) {
  const p = TR.res.points[i];
  try {
    S = await api("/api/tradeoff/aim", { i }); A = null; TR = null;
    renderAll();
    setMsg("tradeoff-msg", "");
    await advise();
    setMsg("advice-msg", `Minimum selectivity set to ${fmt(100 * S.settings.spec.S_C, 1)} % (Product spec, on the left; clear it there to go back to the most profitable run). ${$("advice-msg").textContent}`, "ok");
  } catch (e) { setMsg("tradeoff-msg", e.message, "err"); }
}

// ------------------------------------------------------------------ the reaction finder
const F = { ex: null, src: "example", cands: [], upload: null, poll: null, job: null };
function fCandRow(c, i) {
  const row = el("div", { class: "f-cand" },
    el("div", { class: "f-cand-head" },
      el("input", { type: "text", value: c.name, "aria-label": `name of candidate ${i + 1}`, oninput: (ev) => { F.cands[i].name = ev.target.value; fStartOptions(); } }),
      el("button", { type: "button", class: "row-btn", title: "remove", "aria-label": `remove candidate ${i + 1}`, onclick: () => { F.cands.splice(i, 1); fRenderCands(); } }, "✕")),
    el("textarea", { rows: "4", spellcheck: "false", "aria-label": `reactions of candidate ${i + 1}`, oninput: (ev) => { F.cands[i].text = ev.target.value; } }));
  row.querySelector("textarea").value = c.text;
  return row;
}
function fRenderCands() {
  const box = $("f-cands"); box.innerHTML = "";
  F.cands.forEach((c, i) => box.append(fCandRow(c, i)));
  if (!F.cands.length) box.append(el("p", { class: "hint" }, "No candidates yet: add one, or load the 4 examples."));
  fStartOptions();
}
function fStartOptions() {
  const sel = $("f-start"), cur = sel.value;
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "no reactions (search from scratch)"));
  F.cands.forEach((c, i) => sel.append(el("option", { value: String(i) }, `candidate '${c.name || i + 1}' (repair it)`)));
  sel.value = [...sel.options].some((o) => o.value === cur) ? cur : "";
}
function fLoadExampleCands() { F.cands = F.ex.candidates.map((c) => ({ name: c.name, text: c.text })); fRenderCands(); }
function fShowExample() {
  const e = F.ex.examples.find((x) => x.file === $("f-example").value);
  if (!e) return;
  $("f-example-text").textContent = e.text + (e.mode === "discover" ? " Suggested: Find the reactions automatically." : " Suggested: Rank candidate networks.");
  $("f-example-dl").href = "/examples/discovery/" + e.file;
  $("f-example-dl").setAttribute("download", e.file.split("/").pop());
}
document.querySelectorAll(".f-src .seg-btn").forEach((b) => b.addEventListener("click", () => {
  F.src = b.dataset.src;
  document.querySelectorAll(".f-src .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
  ["example", "upload", "runs"].forEach((k) => { $("f-src-" + k).hidden = k !== F.src; });
  if (F.src === "runs" && S) $("f-runs-text").textContent = `Uses the ${S.n} run(s) in the table above, as they are now` + (S.n < 3 ? " (at least 3 are needed)." : ".") + " Few, step-limited runs from an optimisation campaign carry far less information about the reactions than a kinetic study: take the answer as a first guess.";
}));
$("f-example").addEventListener("change", fShowExample);
$("f-file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  F.upload = { name: f.name, text: await f.text() };
  $("f-file-name").textContent = `${f.name} (${F.upload.text.split(/\r?\n/).filter((l) => l.trim()).length - 1} rows)`;
});
$("f-add").addEventListener("click", () => { F.cands.push({ name: `candidate ${F.cands.length + 1}`, text: "" }); fRenderCands(); });
$("f-load-ex").addEventListener("click", fLoadExampleCands);
function fBody(mode) {
  const b = { mode, source: F.src };
  if (F.src === "example") b.example = $("f-example").value;
  if (F.src === "upload") { if (!F.upload) throw new Error("choose a CSV file first"); b.text = F.upload.text; b.name = F.upload.name; }
  if (mode === "rank") b.candidates = F.cands;
  if (mode === "discover" && $("f-start").value !== "") b.start = F.cands[+$("f-start").value].text;
  return b;
}
async function fStart(mode) {
  try {
    setMsg("f-msg", "");
    const out = await api("/api/finder/start", fBody(mode));
    fRender(out.job); fPoll();
  } catch (e) { setMsg("f-msg", e.message, "err"); }
}
$("f-discover").addEventListener("click", () => fStart("discover"));
$("f-rank").addEventListener("click", () => fStart("rank"));
$("f-stop").addEventListener("click", async () => { try { fRender((await api("/api/finder/stop", {})).job); } catch (e) { setMsg("f-msg", e.message, "err"); } });
function fPoll() {
  if (F.poll) return;
  F.poll = setInterval(async () => {
    try {
      const j = (await api("/api/finder/status")).job;
      fRender(j);
      if (!j || j.status !== "running") { clearInterval(F.poll); F.poll = null; }
    } catch (e) { clearInterval(F.poll); F.poll = null; setMsg("f-msg", e.message, "err"); }
  }, 1000);
}
function fRender(j) {
  const prevId = F.job && F.job.id, prevStatus = F.job && F.job.status;
  F.job = j;
  const running = !!j && j.status === "running";
  $("f-progress").hidden = !running;
  ["f-discover", "f-rank"].forEach((id) => { $(id).disabled = running; });
  if (!j) return;
  if (running) {
    $("f-prog-title").textContent = `${j.mode === "rank" ? "Ranking the candidates" : j.repair ? "Repairing the network" : "Searching for the reactions"} · ${j.label} (${j.n_runs} runs) · ${fmt(j.elapsed, 0)} s` + (j.stopping ? " · stopping…" : "");
    const lg = $("f-log"); lg.innerHTML = "";
    j.log.forEach((l) => lg.append(el("li", {}, el("span", { class: "src" }, `${fmt(l.t, 0)} s `), l.msg)));
    return;
  }
  if (j.status === "done") {
    if (prevId !== j.id || prevStatus !== "done") fResult(j);
    setMsg("f-msg", `Done in ${fmt(j.elapsed, 0)} s (${j.label}, ${j.n_runs} runs).`, "ok");
  } else if (j.status === "error" || j.status === "stopped") {
    setMsg("f-msg", j.error || "stopped", j.status === "error" ? "err" : "");
  }
}
function fResult(j) {
  const R = j.result, best = R.networks[0];
  $("f-result").hidden = false;
  $("f-next-body").hidden = true; setMsg("f-next-msg", "");
  const v = $("f-verdict"); v.innerHTML = "";
  v.append(el("div", { class: "f-verdict-head" },
    el("span", { class: "label" }, R.mode === "rank" ? "Best supported by the data" : "Reactions found"),
    el("span", { class: "pill " + (best.fits_noise ? "yes" : "no") }, best.fits_noise ? "fits within the GC noise" : "does not fit within the GC noise"),
    el("span", { class: "pill " + (R.decided ? "yes" : "no") }, R.decided ? "no other network comes close" : "the data cannot decide everything")));
  if (R.mode === "rank") v.append(el("p", { class: "f-best-name" }, `'${best.name}'`));
  const ol = el("ol", { class: "mono f-eqs" }); best.equations.forEach((l) => ol.append(el("li", {}, l.replace(/^\d+\.\s+/, "")))); v.append(ol);
  const verdict = (best.verdict || "").replace(/\.?\s*$/, ".");
  v.append(el("p", {}, verdict.charAt(0).toUpperCase() + verdict.slice(1) + (best.model === "inhib" ? ` ${best.model_words.charAt(0).toUpperCase()}${best.model_words.slice(1)}.` : "")));
  R.undecided.forEach((u) => v.append(el("p", { class: "hint" }, u)));
  const an = $("f-answer");
  an.hidden = !R.answer; an.innerHTML = "";
  if (R.answer) {
    an.className = "f-answer " + (R.answer.match ? "ok" : "warn");
    an.append(el("span", { class: "label" }, "The answer: the reactions this example file was calculated from"));
    const ao = el("ol", { class: "mono f-eqs" }); R.answer.equations.forEach((l) => ao.append(el("li", {}, l.replace(/^\d+\.\s+/, "")))); an.append(ao);
    an.append(el("p", {}, R.answer.match ? "✓ The finder's best network is exactly these reactions with these rate laws."
      : "✗ The finder's best network differs from these reactions or rate laws (see the table: the data may not be able to tell them apart, or the search missed it)."));
  }
  const T = $("f-table");
  T.querySelector("thead").innerHTML = "";
  T.querySelector("thead").append(el("tr", {}, ...["#", "Network", "Reactions and rate laws", "Typical miss (× GC noise)", "BIC difference", "Evidence weight", "Verdict"].map((h, k) => el("th", { class: k < 3 || k === 6 ? "first" : "" }, h))));
  const tb = T.querySelector("tbody"); tb.innerHTML = "";
  R.networks.forEach((n, k) => {
    const eqs = el("td", { class: "first mono f-eq-cell" }); n.equations.forEach((l) => eqs.append(el("div", {}, l.replace(/^\d+\.\s+/, ""))));
    if (n.model === "inhib") eqs.append(el("div", { class: "src" }, "+ product C blocks catalyst sites"));
    tb.append(el("tr", { class: k === 0 ? "best" : "" },
      el("td", { class: "first num" }, String(n.rank ?? k + 1)), el("td", { class: "first" }, n.name || (k === 0 ? "best" : `runner-up ${k}`)), eqs,
      el("td", { class: "num" }, fmt(n.x_noise, 2)), el("td", { class: "num" }, k === 0 ? "0 (best)" : (n.delta_bic > 1e4 ? n.delta_bic.toExponential(1) : fmt(n.delta_bic, 1))),
      el("td", { class: "num" }, n.weight === undefined || n.weight === null ? "–" : `${fmt(100 * n.weight, n.weight > 0.01 ? 0 : 2)} %`),
      el("td", { class: "first f-verdict-cell" }, (n.tied_with_best && k > 0 ? "cannot be told apart from the best. " : "") + (n.differs_by && k > 0 ? `Differs by: ${n.differs_by}. ` : "") + (n.verdict || ""))));
  });
  const K = $("f-k");
  K.querySelector("thead").innerHTML = "";
  K.querySelector("thead").append(el("tr", {}, ...["Reaction", "Rate law", "k at 300 °C", "Ea (kJ/mol)"].map((h, k) => el("th", { class: k < 2 ? "first" : "" }, h))));
  const kb = K.querySelector("tbody"); kb.innerHTML = "";
  best.rate_constants.forEach((r) => kb.append(el("tr", {},
    el("td", { class: "first mono" }, r.reaction), el("td", { class: "first mono" }, r.rate_law),
    el("td", { class: "num", title: `likely ${Number(r.k_lo).toExponential(2)} – ${Number(r.k_hi).toExponential(2)}` }, `${Number(r.k).toExponential(2)}`, el("span", { class: "src" }, ` ${r.units}`)),
    el("td", { class: "num" }, `${fmt(r.Ea, 0)} ± ${fmt(r.Ea_sd, 0)}`))));
  $("f-k-note").textContent = "k per gram of catalyst at the BET area of the data, at 300 °C; ± = one standard deviation of the fit; hover a k for its likely range."
    + (best.inhibition ? ` Site blocking by C: K_C = ${Number(best.inhibition.K_C).toPrecision(3)} per bar (likely ${Number(best.inhibition.K_C_low).toPrecision(2)}–${Number(best.inhibition.K_C_high).toPrecision(2)}).` : "");
  parityChart($("chart-f-parity"), R.parity);
  const su = $("f-summary"); su.innerHTML = "";
  [...R.summary].forEach((n) => su.append(el("li", {}, n)));
  su.append(el("li", {}, `${R.n_runs} runs (${R.n_settings} different settings)` + (R.n_fits ? `; ${R.n_fits} networks fitted` : "") + (R.restarts ? `; the search started again ${R.restarts} time(s)` : "") + `; ${fmt(R.seconds, 0)} s.`));
  $("f-path-fold").hidden = !R.path.length;
  const pa = $("f-path"); pa.innerHTML = "";
  R.path.forEach((p) => pa.append(el("li", {}, typeof p === "string" ? p : (p.words || p.message || JSON.stringify(p)))));
  $("f-next").disabled = R.networks.length < 2;
  if (R.networks.length < 2) setMsg("f-next-msg", "Needs at least two networks.");
}
$("f-next").addEventListener("click", async () => {
  try {
    setMsg("f-next-msg", "Searching the operating box… (10–30 s)");
    $("f-next").disabled = true;
    const nr = (await api("/api/finder/next_run", { from_last: $("f-step").checked })).next_run;
    const b = $("f-next-body"); b.innerHTML = ""; b.hidden = false;
    const r = nr.run;
    const ul = el("ul", { class: "kv mono" });
    [["Pressure", `${fmt(r.pressure_bar, 1)} bar`], ["Flow of A", `${fmt(r.flow_A_mL_min, 1)} mL/min`], ["Flow of B", `${fmt(r.flow_B_mL_min, 1)} mL/min`],
      ["Inert flow", `${fmt(r.flow_I_mL_min, 1)} mL/min`], ["Temperature", `${fmt(r.temperature_C, 0)} °C`], ["Catalyst", `${fmt(r.catalyst_g, 2)} g`]]
      .forEach(([k, v]) => ul.append(el("li", {}, el("span", {}, k), el("span", {}, v))));
    b.append(ul);
    const names = nr.networks || Object.keys(nr.predicted || {});
    const t = el("table", { class: "runs-table f-table" });
    const sp = Object.keys(nr.predicted[names[0]] || {});
    t.append(el("thead", {}, el("tr", {}, el("th", { class: "first" }, "Predicted GC (mol-%)"), ...sp.map((s) => el("th", {}, s)))));
    const tb = el("tbody");
    names.forEach((nm) => tb.append(el("tr", {}, el("td", { class: "first" }, nm), ...sp.map((s) => el("td", { class: "num" }, fmt(nr.predicted[nm][s], 2))))));
    tb.append(el("tr", {}, el("td", { class: "first" }, "difference (× GC noise)"), ...sp.map((s) => el("td", { class: "num" }, fmt((nr.difference_x_noise || {})[s], 1)))));
    t.append(tb);
    b.append(el("div", { class: "table-wrap" }, t), el("p", {}, nr.message));
    setMsg("f-next-msg", "");
  } catch (e) { setMsg("f-next-msg", e.message, "err"); } finally { $("f-next").disabled = false; }
});
async function finderInit() {
  try {
    F.ex = await api("/api/finder/examples");
    const sel = $("f-example"); sel.innerHTML = "";
    F.ex.examples.forEach((e) => sel.append(el("option", { value: e.file }, `${e.title} (${e.file})`)));
    fShowExample(); fLoadExampleCands();
    const j = (await api("/api/finder/status")).job;
    if (j) { fRender(j); if (j.status === "running") fPoll(); }
  } catch (e) { setMsg("f-msg", "The reaction finder is not available: " + e.message, "err"); }
}

// ------------------------------------------------------------------ everything
function renderAll() {
  const sj = JSON.stringify([S.settings, S.meta.names]);
  if (sj !== renderedSettings) { renderSetup(); renderedSettings = sj; }        // keeps what is being typed otherwise
  const pj = JSON.stringify(S.meta.species);
  if (pj !== renderedSpecies) { runInputs(); closeForm(); renderedSpecies = pj; }
  $("chem-badge").textContent = S.settings.system;
  renderRuns(); renderAdvice(); renderTradeoff(); renderLearned(); renderExplore();
  const best = S.best_idx === null ? null : S.runs[S.best_idx].profit;
  $("status").textContent = S.n === 0 ? "no runs yet" : `${S.n} run${S.n > 1 ? "s" : ""} · best ${fmt(best, 1)} / h` + (A ? " · advice ready" : "");
}
(async function init() {
  try {
    S = await api("/api/state");
    if (S.analysis_ready) { try { A = (await api("/api/analysis")).analysis; } catch (e) { A = null; } }
    renderAll();
  } catch (e) { $("status").textContent = "cannot reach the server: " + e.message; }
  finderInit();
})();
