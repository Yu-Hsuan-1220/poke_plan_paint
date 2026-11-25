// Every poke set as a card (image + arrows), with filters, delete / unskip, and a JSON download.
let ROWS = [];

function card(im, s) {
  const el = document.createElement("article");
  el.className = "rcard" + (s ? "" : " skipped");
  el.innerHTML = `<div class="stage small-stage"><img loading="lazy" src="${esc(im.file)}" alt=""><svg viewBox="0 0 576 320" preserveAspectRatio="none"></svg></div>
    <div class="rmeta"><span>${s ? `<b>${s.n} 個 poke</b> · ${esc(s.annotator)}${s.prompt ? `<br><span class="muted small">“${esc(s.prompt)}”</span>` : ""}` : `<b>已跳過</b> · ${esc(im.skipped)}`}</span>
      <button class="link small" data-act="${s ? "delete" : "unskip"}">${s ? "刪除" : "取消跳過"}</button></div>
    <div class="credit">${creditHtml(im)}</div>`;
  if (s) drawPokes(el.querySelector("svg"), s.pokes);
  el.querySelector("button").addEventListener("click", async () => {
    try {
      await api(s ? "delete" : "unskip", s ? {image: im.id, id: s.id} : {image: im.id});
      load();
    } catch (e) { alert(e.message); }
  });
  return el;
}

function render() {
  const who = $("#f-who").value, n = $("#f-n").value, sk = $("#f-skip").checked;
  const grid = $("#grid");
  grid.replaceChildren();
  let k = 0;
  for (const im of ROWS) {
    for (const s of im.sets) {
      if ((who && s.annotator !== who) || (n && String(s.n) !== n)) continue;
      grid.appendChild(card(im, s)); k++;
    }
    if (sk && im.skipped && !who && !n) { grid.appendChild(card(im, null)); k++; }
  }
  $("#count").textContent = `${k} 筆`;
}

async function load() {
  try {
    const d = await api("all");
    ROWS = d.rows;
    const p = d.progress;
    $("#prog").textContent = `${p.sets} 組 poke · 完成 ${p.done} 張 · 跳過 ${p.skipped} · 剩 ${p.open} / ${p.images}`;
    const sel = $("#f-who"), cur = sel.value;
    sel.innerHTML = '<option value="">全部</option>' + Object.keys(p.by_annotator).map(a => `<option>${esc(a)}</option>`).join("");
    sel.value = cur;
    $("#tools").hidden = false;
    render();
  } catch (e) {
    if (e.status === 401) { $("#need-key").hidden = false; return; }
    alert(e.message);
  }
}
for (const id of ["#f-who", "#f-n", "#f-skip"]) $(id).addEventListener("change", render);
$("#export").addEventListener("click", async () => {
  try {
    const d = await api("export");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(d, null, 1)], {type: "application/json"}));
    a.download = "poke_annotations.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } catch (e) { alert(e.message); }
});
if (store.get("teamKey")) load(); else $("#need-key").hidden = false;
