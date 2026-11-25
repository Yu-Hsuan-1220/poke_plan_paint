# User study: main vs. baselines (pairwise, 2 questions)

Each trial shows the input first frame with the pokes drawn as arrows, then two videos A / B played in sync:
one from the main method and one from an opponent (B0, B1, zlat; vanilla optional). Left/right is randomized.
The participant answers:

- **動作** (motion): 哪一部影片的運動比較符合箭頭的指示？ (Which video's motion better follows the arrows?)
- **真實感** (realism): 先不管箭頭，哪一部影片看起來比較像真實拍攝的影片？ (Ignoring the arrows, which video looks more
  like real footage?)

The site itself is in Traditional Chinese.

Forced choice (no tie), answer buttons unlock after both videos played once. Defaults: 24 comparisons + 2
attention checks (real clip forward vs. reversed; Q1 has a right answer) per participant, ~10 minutes.
(clip, opponent) pairs are handed out least-covered first. The 4-poke renders (`p4`, k0) of the 68 Pexels clips are used.

## Setup

```bash
# 1. media + private manifest (rerun when zlat is rendered; the server picks the new manifest up by itself)
python userstudy/build.py      # p2v env; needs outputs/bench/* renders + data/bench_pexels

# 2. admin token (or set STUDY_ADMIN_TOKEN); skip to run without the admin page
python3 userstudy/server.py --new-token && cat userstudy/data/admin_token.txt

# 3. backend (stdlib only), behind nginx
nohup python3 userstudy/server.py --port 8765 > userstudy/data/server.log 2>&1 &
# (on gpu7 port 8765 is taken by another user: pick a free one, e.g. 8767, and change nginx.conf to match)

# 4. nginx: see userstudy/nginx.conf  ->  https://<host>/study/  and  https://<host>/study/admin.html
```

Local test without nginx: `python3 userstudy/server.py --port 8765 --serve-static` and open http://localhost:8765/.

## Files

| path | public? | |
|---|---|---|
| `site/` (`index.html`, `app.js`, `pokes.js`, `style.css`, `admin.*`) | yes | static frontend |
| `site/media/` | yes | first frames + videos, names are salted hashes (do not reveal the method) |
| `data/manifest.json` | **no** | method ↔ file mapping + salt |
| `data/responses.jsonl` | **no** | append-only log of sessions / answers / feedback |
| `data/admin_token.txt` | **no** | admin token |

`site/media/` and `data/` are git-ignored. Only Pexels clips and videos generated from them are published —
never OpenVid / Koala frames. No IPs or personal data are stored.

## Results

`admin.html` (token required): main's win rate per opponent for both questions, 95% Wilson CI, exact binomial
p-value, attention-check pass rate, per-participant progress and comments, and a CSV of all answers
(`ref_wins_q1/q2` per trial, for a per-participant / mixed-effects analysis in the report).
