// Participant flow: intro -> trials (A/B videos, 2 questions) -> done. Answers are saved per trial, so a reload
// resumes at the first unanswered trial. All URLs are relative, so the site works under any nginx prefix.
const $ = (s, r = document) => r.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: no resume */ } },
};

let S = null;          // {pid, trials, answered:Set}
let cur = -1;          // index into S.trials
let ans = {};          // {q1, q2} for the current trial
let plays = 0, ended = {A: false, B: false}, shownAt = 0, restartTimer = null;
const vA = $("#vA"), vB = $("#vB");

function show(id) {
  document.querySelectorAll(".screen").forEach(s => (s.hidden = s.id !== id));
  window.scrollTo(0, 0);
}

function fail(msg) {
  $("#error-text").textContent = msg;
  show("error");
}

async function api(path, body) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch("api/" + path, {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
      });
      if (r.ok) return await r.json();
      if (r.status < 500) throw new Error("HTTP " + r.status);
    } catch (e) {
      if (attempt === 2) throw e;
    }
    await new Promise(res => setTimeout(res, 800 * (attempt + 1)));
  }
}

function progress() {
  const n = S.trials.length, done = S.answered.size;
  $("#progress-fill").style.width = (100 * done / n) + "%";
  $("#progress-text").textContent = cur >= 0 && cur < n ? `第 ${cur + 1} / ${n} 題` : "";
}

function nextUnanswered(from) {
  for (let i = from; i < S.trials.length; i++) if (!S.answered.has(i)) return i;
  return -1;
}

function preload(i) {
  if (i < 0 || i >= S.trials.length) return;
  const t = S.trials[i];
  new Image().src = t.image;
  for (const u of [t.A, t.B]) {
    const l = document.createElement("link");
    l.rel = "prefetch"; l.href = u; l.as = "video";
    document.head.appendChild(l);
  }
}

function setChoicesEnabled(on) {
  document.querySelectorAll(".choice").forEach(b => (b.disabled = !on));
}

function updateWatchHint() {
  $("#watch-hint").textContent = plays >= 1 ? "" : "請先看完一次兩部影片…";
  setChoicesEnabled(plays >= 1);
}

function loadTrial(i) {
  cur = i;
  const t = S.trials[i];
  ans = {};
  plays = 0; ended = {A: false, B: false};
  clearTimeout(restartTimer);
  document.querySelectorAll(".choice").forEach(b => b.classList.remove("on"));
  $("#next").disabled = true;
  $("#save-status").textContent = "";
  $("#input-img").src = t.image;
  drawPokes($("#input-pokes"), t.pokes);
  document.querySelectorAll(".pokes.overlay").forEach(s => drawPokes(s, t.pokes));
  vA.src = t.A; vB.src = t.B;
  startBoth();
  updateWatchHint();
  progress();
  show("trial");
  shownAt = performance.now();
  preload(nextUnanswered(i + 1));
}

function startBoth() {
  ended = {A: false, B: false};
  for (const v of [vA, vB]) { v.pause(); v.currentTime = 0; }
  // start together once both can play, so the two videos stay in sync
  let ready = 0;
  const go = () => { if (++ready === 2) { vA.play().catch(() => {}); vB.play().catch(() => {}); } };
  for (const v of [vA, vB]) {
    if (v.readyState >= 3) go();
    else v.addEventListener("canplay", go, {once: true});
  }
}

function onEnded(which) {
  ended[which] = true;
  if (ended.A && ended.B) {
    plays++;
    updateWatchHint();
    restartTimer = setTimeout(startBoth, 700);  // short pause on the last frame, then loop both
  }
}
vA.addEventListener("ended", () => onEnded("A"));
vB.addEventListener("ended", () => onEnded("B"));
vA.addEventListener("error", () => { $("#watch-hint").textContent = "影片載入失敗，請重新整理頁面。"; });

$("#replay").addEventListener("click", () => { clearTimeout(restartTimer); startBoth(); });

const ov = $("#overlay-toggle");
ov.checked = store.get("overlay") === "1";
const applyOverlay = () => document.body.classList.toggle("show-overlay", ov.checked);
ov.addEventListener("change", () => { store.set("overlay", ov.checked ? "1" : "0"); applyOverlay(); });
applyOverlay();

document.querySelectorAll(".q").forEach(q => {
  q.querySelectorAll(".choice").forEach(b => b.addEventListener("click", () => {
    q.querySelectorAll(".choice").forEach(x => x.classList.toggle("on", x === b));
    ans[q.dataset.q] = b.dataset.v;
    $("#next").disabled = !(ans.q1 && ans.q2);
  }));
});

$("#next").addEventListener("click", async () => {
  const btn = $("#next");
  btn.disabled = true;
  $("#save-status").textContent = "儲存中…";
  try {
    await api("answer", {pid: S.pid, i: cur, q1: ans.q1, q2: ans.q2, rt_ms: Math.round(performance.now() - shownAt)});
  } catch (e) {
    $("#save-status").textContent = "儲存失敗，請再按一次。";
    btn.disabled = false;
    return;
  }
  S.answered.add(cur);
  const n = nextUnanswered(cur + 1) >= 0 ? nextUnanswered(cur + 1) : nextUnanswered(0);
  if (n >= 0) loadTrial(n); else finish();
});

function finish() {
  vA.pause(); vB.pause();
  vA.removeAttribute("src"); vB.removeAttribute("src");
  cur = -1;
  progress();
  show("done");
}

$("#send-feedback").addEventListener("click", async () => {
  const f = document.querySelector('input[name="familiar"]:checked');
  try {
    await api("finish", {pid: S.pid, familiar: f ? f.value : "", comment: $("#comment").value});
  } catch { /* optional anyway */ }
  $("#feedback").hidden = true;
  $("#feedback-thanks").hidden = false;
});

$("#start").addEventListener("click", () => {
  const i = nextUnanswered(0);
  if (i >= 0) loadTrial(i); else finish();
});

(async function init() {
  try {
    const r = await api("session", {pid: store.get("pid")});
    store.set("pid", r.pid);
    S = {pid: r.pid, trials: r.trials, answered: new Set(r.answered)};
  } catch (e) {
    return fail("無法連到伺服器（" + e.message + "），請稍後再試。");
  }
  progress();
  if (S.answered.size === S.trials.length) {
    show("done");
  } else {
    if (S.answered.size > 0) $("#start").textContent = `繼續（已完成 ${S.answered.size} / ${S.trials.length} 題）`;
    show("intro");
    preload(nextUnanswered(0));
  }
})();
