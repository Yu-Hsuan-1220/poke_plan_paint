// Annotation flow: get an image -> pick a poke count -> drag arrows -> save (then optionally a 2nd set with another
// count) / skip / pass. Pokes are [y0, x0, y1, x1] normalized to [-1, 1] on the 576x320 frame (= bench_pexels).
const NS = "http://www.w3.org/2000/svg";
const svg = $("#svg"), stage = $("#stage");
const gExisting = document.createElementNS(NS, "g"), gCur = document.createElementNS(NS, "g"), gDrag = document.createElementNS(NS, "g");
gExisting.setAttribute("opacity", "0.35");
svg.append(gExisting, gCur, gDrag);

let IM = null;          // current image (public_image from the server)
let target = null;      // chosen poke count
let pokes = [];         // drawn so far
let drag = null;        // [y0, x0] while dragging
const passed = new Set();

function show(id) {
  for (const s of ["gate", "work", "empty"]) $("#" + s).hidden = s !== id;
}

function setProgress(p) {
  if (!p) return;
  $("#prog").textContent = `完成 ${p.done} 張 · 標了一組 ${p.half} 張 · 跳過 ${p.skipped} · 剩 ${p.open} / ${p.images}`;
}

function usedCounts() { return new Set((IM?.sets || []).map(s => s.n)); }

function render() {
  drawPokes(gCur, pokes);
  gExisting.replaceChildren();
  if ($("#show-existing").checked) for (const s of IM?.sets || []) {
    const g = document.createElementNS(NS, "g");
    drawPokes(g, s.pokes);
    gExisting.appendChild(g);
  }
  const used = usedCounts();
  document.querySelectorAll("#counts button").forEach(b => {
    const n = +b.dataset.n;
    b.disabled = used.has(n) || pokes.length > n;
    b.setAttribute("aria-pressed", String(n === target));
  });
  $("#drawn").textContent = target ? `${pokes.length} / ${target}` : "先選數量";
  $("#save").disabled = !(target && pokes.length === target);
  stage.classList.toggle("can-draw", !!target && pokes.length < target);
  const ex = IM?.sets || [];
  $("#existing").hidden = !ex.length;
  $("#existing-list").innerHTML = ex.map(s => `<div class="ex-row"><span>${s.n} 個 poke <span class="muted small">· ${esc(s.annotator)}${s.prompt ? " · “" + esc(s.prompt) + "”" : ""}</span></span>
      <button class="link small" data-del="${esc(s.id)}">刪除</button></div>`).join("");
  $("#draw-block").hidden = ex.length >= 2 || !$("#after").hidden;
}

function load(im, progress) {
  setProgress(progress);
  if (!im) { IM = null; show("empty"); return; }
  IM = im; pokes = []; drag = null; $("#prompt").value = "";
  const used = usedCounts();
  target = [1, 4, 2, 3].find(n => !used.has(n)) ?? null;  // default: 1 poke, then 4 for the second set
  $("#after").hidden = true;
  $("#msg").textContent = "";
  $("#img").src = im.file;
  $("#credit").innerHTML = creditHtml(im);
  show("work");
  render();
}

async function next() {
  try {
    const d = await api("next", {exclude: [...passed]});
    load(d.image, d.progress);
  } catch (e) { onError(e); }
}

function onError(e) {
  if (e.status === 401) { store.set("teamKey", null); gate("團隊密碼不正確"); return; }
  $("#msg").textContent = e.message;
  $("#msg").className = "small err";
}

// ---------------- drawing
function norm(ev) {
  const r = stage.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
  return [y * 2 - 1, x * 2 - 1];
}
stage.addEventListener("pointerdown", ev => {
  if (!target || pokes.length >= target || ev.button > 0) return;
  ev.preventDefault();
  stage.setPointerCapture(ev.pointerId);
  drag = norm(ev);
  drawPokes(gDrag, [[drag[0], drag[1], drag[0], drag[1]]], pokes.length);
});
stage.addEventListener("pointermove", ev => {
  if (!drag) return;
  const [y, x] = norm(ev);
  drawPokes(gDrag, [[drag[0], drag[1], y, x]], pokes.length);
});
stage.addEventListener("pointerup", ev => {
  if (!drag) return;
  let [y, x] = norm(ev);
  const r = stage.getBoundingClientRect();
  if (Math.hypot((x - drag[1]) * r.width, (y - drag[0]) * r.height) / 2 < 6) { y = drag[0]; x = drag[1]; }  // a click = stay
  pokes.push([drag[0], drag[1], y, x].map(v => Math.round(v * 1e5) / 1e5));
  drag = null;
  gDrag.replaceChildren();
  render();
});
stage.addEventListener("pointercancel", () => { drag = null; gDrag.replaceChildren(); });

// ---------------- controls
$("#counts").addEventListener("click", ev => {
  const b = ev.target.closest("button");
  if (!b || b.disabled) return;
  target = +b.dataset.n;
  render();
});
$("#undo").addEventListener("click", () => { pokes.pop(); render(); });
$("#clear").addEventListener("click", () => { pokes = []; render(); });
$("#show-existing").addEventListener("change", render);

$("#save").addEventListener("click", async () => {
  if ($("#save").disabled) return;
  $("#save").disabled = true;
  try {
    const d = await api("annotate", {image: IM.id, n: target, pokes, prompt: $("#prompt").value});
    setProgress(d.progress);
    IM = d.image; pokes = []; $("#prompt").value = "";
    if (IM.sets.length < 2) {
      $("#after").hidden = false;
      target = null;
      render();
    } else {
      next();
    }
  } catch (e) { onError(e); render(); }
});
$("#again").addEventListener("click", () => {
  $("#after").hidden = true;
  const used = usedCounts();
  target = [4, 1, 2, 3].find(n => !used.has(n)) ?? null;
  render();
});
$("#after-next").addEventListener("click", next);

$("#existing-list").addEventListener("click", async ev => {
  const id = ev.target.dataset?.del;
  if (!id) return;
  try {
    const d = await api("delete", {image: IM.id, id});
    IM = d.image; setProgress(d.progress);
    $("#after").hidden = true;
    if (!target) target = [1, 4, 2, 3].find(n => !usedCounts().has(n)) ?? null;
    render();
  } catch (e) { onError(e); }
});

document.querySelectorAll(".skip").forEach(b => b.addEventListener("click", async () => {
  try {
    await api("skip", {image: IM.id, reason: b.dataset.r});
    next();
  } catch (e) { onError(e); }
}));
$("#pass").addEventListener("click", () => { if (IM) passed.add(IM.id); next(); });

document.addEventListener("keydown", ev => {
  if ($("#work").hidden || ev.target.matches("input, textarea") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (/^[1-4]$/.test(ev.key)) { const b = $(`#counts button[data-n="${ev.key}"]`); if (b && !b.disabled) { target = +ev.key; render(); } }
  else if (ev.key === "z" || ev.key === "Z") $("#undo").click();
  else if (ev.key === "Enter") $("#save").click();
  else if (ev.key === "n" || ev.key === "N") $("#pass").click();
});

// ---------------- gate
function gate(err) {
  $("#name").value = store.get("name") || "";
  $("#gate-err").textContent = err || "";
  show("gate");
}
$("#gate-form").addEventListener("submit", ev => {
  ev.preventDefault();
  store.set("name", $("#name").value.trim());
  store.set("teamKey", $("#key").value.trim());
  $("#who").textContent = store.get("name");
  next();
});
$("#who").textContent = store.get("name") || "";
if (store.get("name") && store.get("teamKey")) next(); else gate();
