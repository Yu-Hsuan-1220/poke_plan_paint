// Shared by annotate.js and review.js: team key / name in localStorage, API calls with the key header.
const $ = (s, r = document) => r.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* ignore */ } },
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

class ApiError extends Error {
  constructor(status, msg) { super(msg); this.status = status; }
}

async function api(path, body) {
  const opt = {headers: {"X-Team-Key": store.get("teamKey") || ""}, cache: "no-store"};
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify({annotator: store.get("name") || "", ...body});
  }
  const r = await fetch("api/" + path, opt);
  let d = {};
  try { d = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new ApiError(r.status, d.error || "HTTP " + r.status);
  return d;
}

function creditHtml(im) {
  const lic = im.license_url ? `<a href="${esc(im.license_url)}" target="_blank" rel="noopener">${esc(im.license)}</a>` : esc(im.license);
  return `“${esc(im.title)}” by ${esc(im.author)}, ${lic}, via <a href="${esc(im.source)}" target="_blank" rel="noopener">Wikimedia Commons</a>`;
}
