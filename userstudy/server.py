"""User-study backend: hands out trials, records answers, and serves the admin results. Python stdlib only.

    python3 userstudy/server.py --new-token          # write a random admin token to userstudy/data/admin_token.txt
    python3 userstudy/server.py --port 8765          # behind nginx (nginx serves site/, proxies /api/ here)
    python3 userstudy/server.py --port 8765 --serve-static   # local test without nginx: http://localhost:8765/

Every trial pairs the reference method (main) with one opponent; which video is A (left) is randomized and
balanced per participant. (clip, opponent) pairs are handed out least-covered first, so answers spread over all
68 clips x opponents. Each participant also gets --catch attention checks: the real clip played forward vs
reversed, where Q1 has a right answer (forward follows the arrows).

State is an append-only JSONL log (userstudy/data/responses.jsonl) replayed at start-up; one process, one lock.
No IPs or other personal data are stored.
"""

import argparse
import csv
import hmac
import io
import json
import math
import os
import random
import secrets
import threading
import time
import uuid
from fractions import Fraction
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
DATA = os.path.join(HERE, "data")
MANIFEST = os.path.join(DATA, "manifest.json")
TOKEN_FILE = os.path.join(DATA, "admin_token.txt")
PENDING_S = 2 * 3600  # an unanswered assignment counts toward coverage for this long

LOCK = threading.Lock()
STATE = {"sessions": {}, "man": None, "man_mtime": 0}
CFG = {}


# ------------------------------------------------------------------ state
def manifest():
    m = os.path.getmtime(MANIFEST)
    if m != STATE["man_mtime"]:
        STATE["man"], STATE["man_mtime"] = json.load(open(MANIFEST)), m
    return STATE["man"]


def log(ev):
    ev["t"] = time.time()
    with open(CFG["log"], "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    apply(ev)


def apply(ev):
    S = STATE["sessions"]
    if ev["type"] == "session":
        S[ev["pid"]] = {"created": ev["t"], "trials": ev["trials"], "answers": {}, "finish": None}
    elif ev["type"] == "answer" and ev["pid"] in S:
        S[ev["pid"]]["answers"][ev["i"]] = ev
    elif ev["type"] == "finish" and ev["pid"] in S:
        S[ev["pid"]]["finish"] = ev


def replay():
    if os.path.exists(CFG["log"]):
        for line in open(CFG["log"]):
            if line.strip():
                apply(json.loads(line))


# ------------------------------------------------------------------ trials
def coverage(man):
    """(clip, opponent) -> answered + recently assigned count."""
    cnt, now = {}, time.time()
    for s in STATE["sessions"].values():
        fresh = now - s["created"] < PENDING_S
        for t in s["trials"]:
            if t["kind"] == "cmp" and (t["i"] in s["answers"] or fresh):
                cnt[(t["clip"], t["opp"])] = cnt.get((t["clip"], t["opp"]), 0) + 1
    return cnt


def new_trials(man):
    rng = random.Random()
    ref, opps, clips = man["ref"], man["opponents"], man["clips"]
    n = CFG["trials"]
    slots = opps * (n // len(opps)) + rng.sample(opps, n % len(opps))
    rng.shuffle(slots)
    cnt, used, trials = coverage(man), set(), []
    for opp in slots:
        cand = [c for c, v in clips.items() if c not in used and ref in v["videos"] and opp in v["videos"]]
        if not cand:
            continue
        rng.shuffle(cand)
        c = min(cand, key=lambda c: cnt.get((c, opp), 0))
        used.add(c)
        cnt[(c, opp)] = cnt.get((c, opp), 0) + 1
        trials.append({"kind": "cmp", "clip": c, "opp": opp})
    left = [True] * (len(trials) // 2) + [False] * (len(trials) - len(trials) // 2)
    rng.shuffle(left)
    for t, l in zip(trials, left):
        t["a"], t["b"] = (ref, t["opp"]) if l else (t["opp"], ref)
    catch = [c for c, v in clips.items() if "catch" in v]
    catch = [c for c in catch if c not in used] or catch
    for c in rng.sample(catch, min(CFG["catch"], len(catch))):
        a, b = ("gt_fwd", "gt_rev") if rng.random() < 0.5 else ("gt_rev", "gt_fwd")
        trials.insert(rng.randint(min(2, len(trials)), len(trials)), {"kind": "catch", "clip": c, "a": a, "b": b})
    for i, t in enumerate(trials):
        t["i"] = i
    return trials


def video_url(man, clip, key):
    v = man["clips"][clip]
    return v["catch"][key[3:]] if key.startswith("gt_") else v["videos"][key]


def public(man, s):
    out = []
    for t in s["trials"]:
        c = man["clips"][t["clip"]]
        out.append({"i": t["i"], "image": c["image"], "pokes": c["pokes"],
                    "A": video_url(man, t["clip"], t["a"]), "B": video_url(man, t["clip"], t["b"])})
    return out


# ------------------------------------------------------------------ results
def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def binom_p(k, n):
    """exact two-sided binomial test against 0.5."""
    if n == 0:
        return None
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1))
    return min(1.0, float(Fraction(2 * tail, 2 ** n)))


def rows(man, exclude_failed):
    """one dict per answered trial, plus per-participant info."""
    out, people = [], []
    ref = man["ref"]
    for pid, s in STATE["sessions"].items():
        tr = {t["i"]: t for t in s["trials"]}
        ch = [(tr[i], a) for i, a in s["answers"].items() if tr[i]["kind"] == "catch"]
        ok = sum((a["q1"] == "A") == (t["a"] == "gt_fwd") for t, a in ch)
        passed = ok == len(ch)
        rts = sorted(a.get("rt_ms", 0) for a in s["answers"].values())
        fin = s["finish"] or {}
        people.append({"pid": pid[:8], "started": time.strftime("%m-%d %H:%M", time.localtime(s["created"])),
                       "answered": len(s["answers"]), "trials": len(s["trials"]), "finished": bool(s["finish"]),
                       "catch": f"{ok}/{len(ch)}", "passed": passed,
                       "median_rt_s": round(rts[len(rts) // 2] / 1000, 1) if rts else None,
                       "familiar": fin.get("familiar", ""), "comment": fin.get("comment", "")})
        if exclude_failed and not passed:
            continue
        for i, a in sorted(s["answers"].items()):
            t = tr[i]
            pick = {"A": t["a"], "B": t["b"]}
            r = {"pid": pid, "trial": i, "kind": t["kind"], "clip": t["clip"], "opponent": t.get("opp", ""),
                 "A": t["a"], "B": t["b"], "q1": a["q1"], "q2": a["q2"], "q1_pick": pick[a["q1"]],
                 "q2_pick": pick[a["q2"]], "rt_ms": a.get("rt_ms", ""), "passed_catch": passed,
                 "answered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["t"]))}
            if t["kind"] == "cmp":
                r["ref_wins_q1"], r["ref_wins_q2"] = int(r["q1_pick"] == ref), int(r["q2_pick"] == ref)
            out.append(r)
    return out, people


def summary(man, exclude_failed):
    rs, people = rows(man, exclude_failed)
    by = []
    for opp in sorted({r["opponent"] for r in rs if r["kind"] == "cmp"} | set(man["opponents"]),
                      key=lambda o: (o not in man["opponents"], o)):
        sel = [r for r in rs if r["kind"] == "cmp" and r["opponent"] == opp]
        e = {"opp": opp, "label": man["labels"].get(opp, opp), "n": len(sel),
             "clips": len({r["clip"] for r in sel}), "people": len({r["pid"] for r in sel})}
        for q in ("q1", "q2"):
            k = sum(r[f"ref_wins_{q}"] for r in sel)
            lo, hi = wilson(k, len(sel))
            e[q] = {"wins": k, "rate": k / len(sel) if sel else None, "lo": lo, "hi": hi, "p": binom_p(k, len(sel))}
        by.append(e)
    catch = [r for r in rs if r["kind"] == "catch"]
    return {"ref": man["ref"], "ref_label": man["labels"].get(man["ref"]), "by_opp": by, "people": people,
            "n_answers": sum(r["kind"] == "cmp" for r in rs),
            "catch": {"n": len(catch), "correct": sum(r["q1_pick"] == "gt_fwd" for r in catch)},
            "exclude_failed": exclude_failed}


def to_csv(man):
    rs, _ = rows(man, False)
    cols = ["pid", "trial", "kind", "clip", "opponent", "A", "B", "q1", "q2", "q1_pick", "q2_pick", "ref_wins_q1",
            "ref_wins_q2", "rt_ms", "passed_catch", "answered_at"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rs)
    return buf.getvalue()


# ------------------------------------------------------------------ http
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SITE, **k)

    def log_message(self, fmt, *args):  # no client addresses in the logs
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 20000:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def admin_ok(self):
        tok = CFG.get("token")
        got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return bool(tok) and hmac.compare_digest(got.encode(), tok.encode())

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            if CFG["serve_static"]:
                return super().do_GET()
            return self.send(404, {"error": "not found"})
        try:
            with LOCK:
                man = manifest()
                if u.path in ("/api/admin/summary", "/api/admin/answers.csv"):
                    if not CFG.get("token"):
                        return self.send(503, {"error": "admin disabled: no token configured"})
                    if not self.admin_ok():
                        return self.send(401, {"error": "invalid token"})
                    if u.path.endswith(".csv"):
                        return self.send(200, to_csv(man), "text/csv; charset=utf-8",
                                         {"Content-Disposition": 'attachment; filename="userstudy_answers.csv"'})
                    ex = parse_qs(u.query).get("exclude_failed", ["0"])[0] == "1"
                    return self.send(200, summary(man, ex))
            return self.send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self.send(500, {"error": type(e).__name__})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self.body()
            with LOCK:
                man = manifest()
                if u.path == "/api/session":
                    pid = b.get("pid")
                    s = STATE["sessions"].get(pid) if isinstance(pid, str) else None
                    if s is None:
                        pid = uuid.uuid4().hex
                        log({"type": "session", "pid": pid, "trials": new_trials(man)})
                        s = STATE["sessions"][pid]
                    return self.send(200, {"pid": pid, "trials": public(man, s),
                                           "answered": sorted(s["answers"]), "finished": bool(s["finish"])})
                if u.path == "/api/answer":
                    s = STATE["sessions"].get(b.get("pid"))
                    i = b.get("i")
                    if s is None or not isinstance(i, int) or not 0 <= i < len(s["trials"]) \
                            or b.get("q1") not in ("A", "B") or b.get("q2") not in ("A", "B"):
                        return self.send(400, {"error": "bad answer"})
                    rt = b.get("rt_ms")
                    log({"type": "answer", "pid": b["pid"], "i": i, "q1": b["q1"], "q2": b["q2"],
                         "rt_ms": int(rt) if isinstance(rt, (int, float)) else None})
                    return self.send(200, {"ok": True})
                if u.path == "/api/finish":
                    if b.get("pid") not in STATE["sessions"]:
                        return self.send(400, {"error": "unknown session"})
                    log({"type": "finish", "pid": b["pid"], "familiar": str(b.get("familiar", ""))[:20],
                         "comment": str(b.get("comment", ""))[:2000]})
                    return self.send(200, {"ok": True})
            return self.send(404, {"error": "not found"})
        except (ValueError, json.JSONDecodeError):
            return self.send(400, {"error": "bad request"})
        except Exception as e:  # noqa: BLE001
            return self.send(500, {"error": type(e).__name__})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--trials", type=int, default=24, help="comparison trials per participant")
    ap.add_argument("--catch", type=int, default=2, help="attention-check trials per participant")
    ap.add_argument("--log", default=os.path.join(DATA, "responses.jsonl"))
    ap.add_argument("--serve-static", action="store_true", help="also serve site/ (local testing without nginx)")
    ap.add_argument("--new-token", action="store_true", help="write a new admin token and exit")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.new_token:
        fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, (secrets.token_urlsafe(24) + "\n").encode())
        os.close(fd)
        print(f"admin token written to {TOKEN_FILE}")
        return
    CFG.update(trials=a.trials, catch=a.catch, log=a.log, serve_static=a.serve_static,
               token=os.environ.get("STUDY_ADMIN_TOKEN") or (
                   open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else None))
    manifest()
    replay()
    print(f"{len(STATE['sessions'])} sessions replayed; opponents {STATE['man']['opponents']}; "
          f"admin {'on' if CFG['token'] else 'OFF (no token)'}; http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
