"""Poke-annotation backend for the qualitative results. Python stdlib only; nginx serves site/, proxies /api/ here.

    python3 annotate/server.py --new-key              # team key -> annotate/data/team_key.txt (share with teammates)
    python3 annotate/server.py --port 8766            # behind nginx (see annotate/nginx.conf)
    python3 annotate/server.py --port 8766 --serve-static   # local test: http://localhost:8766/

Rules: an image gets at most two poke sets, with different poke counts (1-4); anyone can skip an image, which
takes it out of the queue. An image handed to one annotator is held for 10 min so two people don't do the same
one. Every write needs the team key (X-Team-Key header). State is an append-only JSONL log replayed at start-up.
    GET /api/export   -> all current poke sets (what src/render_custom.py reads)
"""

import argparse
import hmac
import json
import os
import random
import secrets
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
DATA = os.path.join(HERE, "data")
IMAGES = os.path.join(DATA, "images.json")
KEY_FILE = os.path.join(DATA, "team_key.txt")
HOLD_S = 600
MAX_SETS = 2

LOCK = threading.Lock()
S = {"images": {}, "images_mtime": 0, "annots": {}, "skips": {}, "holds": {}}
CFG = {}


def images():
    m = os.path.getmtime(IMAGES)
    if m != S["images_mtime"]:
        S["images"] = {im["id"]: im for im in json.load(open(IMAGES))}
        S["images_mtime"] = m
    return S["images"]


def log(ev):
    ev["t"] = time.time()
    with open(CFG["log"], "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    apply(ev)


def apply(ev):
    if ev["type"] == "annot":
        S["annots"][ev["id"]] = ev
    elif ev["type"] == "delete":
        S["annots"].pop(ev["id"], None)
    elif ev["type"] == "skip":
        S["skips"][ev["image"]] = ev
    elif ev["type"] == "unskip":
        S["skips"].pop(ev["image"], None)


def sets_of(iid):
    return sorted((a for a in S["annots"].values() if a["image"] == iid), key=lambda a: a["t"])


def public_image(iid):
    im = images()[iid]
    return {"id": iid, "file": im["file"], "title": im["title"], "author": im["author"], "license": im["license"],
            "license_url": im.get("license_url", ""), "source": im["source"],
            "sets": [{"id": a["id"], "n": a["n"], "pokes": a["pokes"], "annotator": a["annotator"],
                      "prompt": a.get("prompt", "")}
                     for a in sets_of(iid)]}


def next_image(who, exclude):
    """open images: not skipped, < MAX_SETS sets, not held by someone else. Half-done images first."""
    now, ims = time.time(), images()
    S["holds"] = {k: v for k, v in S["holds"].items() if now - v[1] < HOLD_S}
    cand = []
    for iid in ims:
        if iid in S["skips"] or iid in exclude:
            continue
        n = len(sets_of(iid))
        if n >= MAX_SETS:
            continue
        h = S["holds"].get(iid)
        if h and h[0] != who:
            continue
        cand.append((-n, random.random(), iid))
    if not cand:
        return None
    iid = min(cand)[2]
    S["holds"][iid] = (who, now)
    return iid


def progress():
    ims = images()
    done = sum(len(sets_of(i)) >= MAX_SETS for i in ims if i not in S["skips"])
    half = sum(len(sets_of(i)) == 1 for i in ims if i not in S["skips"])
    by_n, by_who = {}, {}
    for a in S["annots"].values():
        by_n[a["n"]] = by_n.get(a["n"], 0) + 1
        by_who[a["annotator"]] = by_who.get(a["annotator"], 0) + 1
    return {"images": len(ims), "done": done, "half": half, "skipped": len(S["skips"]),
            "open": len(ims) - done - len(S["skips"]), "sets": len(S["annots"]),
            "by_n": {str(k): v for k, v in sorted(by_n.items())}, "by_annotator": by_who}


def valid_pokes(p, n):
    return isinstance(p, list) and len(p) == n and all(
        isinstance(q, list) and len(q) == 4 and all(isinstance(v, (int, float)) and -1 <= v <= 1 for v in q)
        for q in p)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SITE, **k)

    def log_message(self, fmt, *args):
        pass

    def send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def key_ok(self):
        got = self.headers.get("X-Team-Key", "").strip()
        return bool(CFG["key"]) and hmac.compare_digest(got.encode(), CFG["key"].encode())

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return super().do_GET() if CFG["serve_static"] else self.send(404, {"error": "not found"})
        if not self.key_ok():
            return self.send(401, {"error": "team key required"})
        with LOCK:
            if u.path == "/api/progress":
                return self.send(200, progress())
            if u.path == "/api/all":  # review page: every image that has sets or was skipped
                ims = images()
                rows = [dict(public_image(i), skipped=S["skips"].get(i, {}).get("annotator"))
                        for i in ims if sets_of(i) or i in S["skips"]]
                return self.send(200, {"rows": rows, "progress": progress()})
            if u.path == "/api/export":
                return self.send(200, {"sets": [
                    {"id": a["id"], "image": a["image"], "file": images()[a["image"]]["file"], "n": a["n"],
                     "pokes": a["pokes"], "annotator": a["annotator"], "prompt": a.get("prompt", ""),
                     "credit": {k: images()[a["image"]][k] for k in ("title", "author", "license", "source")}}
                    for a in sorted(S["annots"].values(), key=lambda a: a["t"]) if a["image"] in images()],
                    "format": "pokes = [y0, x0, y1, x1] normalized to [-1, 1] on the 576x320 frame"})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20000:
                return self.send(413, {"error": "too large"})
            b = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send(400, {"error": "bad json"})
        if not self.key_ok():
            return self.send(401, {"error": "team key required"})
        who = str(b.get("annotator", "")).strip()[:30]
        if not who:
            return self.send(400, {"error": "annotator name required"})
        with LOCK:
            ims = images()
            iid = b.get("image")
            if u.path == "/api/next":
                ex = set(b.get("exclude", [])) if isinstance(b.get("exclude"), list) else set()
                nid = next_image(who, ex)
                return self.send(200, {"image": public_image(nid) if nid else None, "progress": progress()})
            if u.path == "/api/get" and iid in ims:
                return self.send(200, {"image": public_image(iid), "progress": progress()})
            if iid not in ims:
                return self.send(400, {"error": "unknown image"})
            if u.path == "/api/annotate":
                k = b.get("n")
                sets = sets_of(iid)
                if not isinstance(k, int) or not 1 <= k <= 4 or not valid_pokes(b.get("pokes"), k):
                    return self.send(400, {"error": "need n in 1..4 and n pokes [y0,x0,y1,x1] in [-1,1]"})
                if iid in S["skips"]:
                    return self.send(409, {"error": "this image was skipped"})
                if len(sets) >= MAX_SETS:
                    return self.send(409, {"error": f"already has {MAX_SETS} poke sets"})
                if any(a["n"] == k for a in sets):
                    return self.send(409, {"error": f"already has a {k}-poke set"})
                aid = uuid.uuid4().hex[:12]
                log({"type": "annot", "id": aid, "image": iid, "n": k,
                     "pokes": [[round(v, 5) for v in q] for q in b["pokes"]], "annotator": who,
                     "prompt": str(b.get("prompt") or "").strip()[:300]})
                return self.send(200, {"ok": True, "id": aid, "image": public_image(iid), "progress": progress()})
            if u.path == "/api/delete":
                a = S["annots"].get(b.get("id"))
                if not a or a["image"] != iid:
                    return self.send(400, {"error": "unknown set"})
                log({"type": "delete", "id": a["id"], "image": iid, "annotator": who})
                return self.send(200, {"ok": True, "image": public_image(iid), "progress": progress()})
            if u.path == "/api/skip":
                log({"type": "skip", "image": iid, "annotator": who, "reason": str(b.get("reason", ""))[:60]})
                S["holds"].pop(iid, None)
                return self.send(200, {"ok": True, "progress": progress()})
            if u.path == "/api/unskip":
                log({"type": "unskip", "image": iid, "annotator": who})
                return self.send(200, {"ok": True, "progress": progress()})
        return self.send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--log", default=os.path.join(DATA, "annotations.jsonl"))
    ap.add_argument("--serve-static", action="store_true")
    ap.add_argument("--new-key", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.new_key:
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, (secrets.token_urlsafe(9) + "\n").encode())
        os.close(fd)
        print(f"team key written to {KEY_FILE}")
        return
    CFG.update(log=a.log, serve_static=a.serve_static,
               key=os.environ.get("ANNOTATE_TEAM_KEY") or (open(KEY_FILE).read().strip()
                                                           if os.path.exists(KEY_FILE) else None))
    if not CFG["key"]:
        raise SystemExit("no team key: run with --new-key first (or set ANNOTATE_TEAM_KEY)")
    images()
    if os.path.exists(a.log):
        for line in open(a.log):
            if line.strip():
                apply(json.loads(line))
    print(f"{len(S['images'])} images, {len(S['annots'])} poke sets, {len(S['skips'])} skipped; "
          f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
