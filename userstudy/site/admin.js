const $ = s => document.querySelector(s);
const ss = {
  get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch { /* ignore */ } },
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = v => v == null ? "—" : (100 * v).toFixed(1) + "%";
const FAMILIAR = {no: "不熟悉", some: "看過一些", yes: "很熟悉"};

async function get(path) {
  const r = await fetch("api/admin/" + path, {headers: {Authorization: "Bearer " + ss.get("token")}, cache: "no-store"});
  if (r.status === 401) throw new Error("token 不正確");
  if (r.status === 503) throw new Error("伺服器沒有設定 admin token");
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r;
}

function qCell(q) {
  if (!q || q.rate == null) return "<td>—</td><td></td><td>—</td>";
  const cls = q.p < 0.05 ? (q.rate > 0.5 ? "sig" : "sig neg") : "";
  const bar = `<div class="winbar" title="95% CI ${pct(q.lo)} – ${pct(q.hi)}">
      <div class="ci" style="left:${100 * q.lo}%;width:${100 * (q.hi - q.lo)}%"></div>
      <div class="half"></div><div class="pt" style="left:${100 * q.rate}%"></div></div>`;
  const p = q.p < 0.001 ? "<0.001" : q.p.toFixed(3);
  return `<td class="${cls}">${pct(q.rate)} <span class="muted small">(${q.wins})</span></td><td>${bar}</td><td class="${cls}">${p}</td>`;
}

async function load() {
  let d;
  try {
    d = await (await get("summary?exclude_failed=" + ($("#exclude").checked ? 1 : 0))).json();
  } catch (e) {
    ss.set("token", null);
    $("#results").hidden = true; $("#login").hidden = false;
    $("#login-err").textContent = e.message;
    return;
  }
  $("#login").hidden = true; $("#results").hidden = false;
  const P = d.people, fin = P.filter(p => p.finished || p.answered === p.trials).length;
  const passed = P.filter(p => p.passed && p.catch !== "0/0").length, checked = P.filter(p => p.catch !== "0/0").length;
  $("#tiles").innerHTML = [
    ["受試者（開始）", P.length], ["完成全部題目", fin], ["比較作答數", d.n_answers],
    ["注意力檢查通過", checked ? `${passed} / ${checked}` : "—"],
  ].map(([k, v]) => `<div class="tile"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("");

  $("#by-title").textContent = `${d.ref_label || d.ref} 的勝率`;
  $("#by-table").innerHTML = `<tr><th>對手</th><th>n 題</th><th>clip 數</th><th>人數</th>
      <th>動作：main 勝率</th><th></th><th>p</th><th>真實感：main 勝率</th><th></th><th>p</th></tr>` +
    d.by_opp.map(e => `<tr><td>${esc(e.label)}</td><td>${e.n}</td><td>${e.clips}</td><td>${e.people}</td>${qCell(e.q1)}${qCell(e.q2)}</tr>`).join("");

  $("#people-table").innerHTML = `<tr><th>id</th><th>開始</th><th>進度</th><th>注意力檢查</th><th>中位作答秒數</th><th>熟悉度</th><th>留言</th></tr>` +
    P.sort((a, b) => b.started.localeCompare(a.started)).map(p => `<tr>
      <td><code>${esc(p.pid)}</code></td><td>${esc(p.started)}</td><td>${p.answered} / ${p.trials}${p.finished ? " ✓" : ""}</td>
      <td class="${p.passed ? "" : "sig neg"}">${esc(p.catch)}</td><td>${p.median_rt_s ?? "—"}</td>
      <td>${esc(FAMILIAR[p.familiar] || "")}</td><td class="wrap-cell">${esc(p.comment)}</td></tr>`).join("");
  $("#updated").textContent = "更新於 " + new Date().toLocaleTimeString();
}

$("#login-form").addEventListener("submit", e => {
  e.preventDefault();
  ss.set("token", $("#token").value.trim());
  $("#login-err").textContent = "";
  load();
});
$("#refresh").addEventListener("click", load);
$("#exclude").addEventListener("change", load);
$("#logout").addEventListener("click", () => { ss.set("token", null); location.reload(); });
$("#csv").addEventListener("click", async () => {
  try {
    const blob = await (await get("answers.csv")).blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "userstudy_answers.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } catch (e) { alert(e.message); }
});
if (ss.get("token")) load();
