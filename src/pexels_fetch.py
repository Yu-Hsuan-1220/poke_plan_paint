"""Pick Pexels stock videos by title (people moving / animals / hands + objects) and download only their start.

Metadata: jovianzm/Pexels-400k on HF (title, duration, https://pexels.com/download/video/<id>). The download
URL redirects to the original file on videos.pexels.com; ffmpeg reads just the first --seconds of it over HTTP
(stream copy, no re-encode), so a 45 s 1080p clip costs a few MB.

Titles with aerial / drone / time-lapse / slow motion / landscape / abstract words are excluded. Camera motion is
not in the metadata: the CoTracker camera rules decide it later (src/openvid_windows.py + openvid_select_train.py).

    python src/pexels_fetch.py --per_cat 130 --out data/pexels_pilot
    -> <out>/videos/<id>.mp4, <out>/selected.csv (columns used by src/openvid_windows.py)
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from common import ROOT

CATS = {
    "people": r"\b(?:walk\w*|run|running|runs|jog\w*|danc\w*|jump\w*|skat\w*|cycling|bik(?:e|ing)|exercis\w*|workout|"
              r"play\w*|swim\w*|climb\w*)",
    "animal": r"\b(?:dog|dogs|puppy|cat|cats|kitten|bird|birds|horse|horses|cow|cows|sheep|goat|chicken|duck|ducks|deer|"
              r"monkey|animal|animals|fish|squirrel|rabbit|elephant|lion|tiger|giraffe|zebra|bear|swan|goose|pig|parrot)\b",
    "object": r"\b(?:pour\w*|cut|cuts|cutting|chop\w*|slic\w*|stir\w*|mix\w*|knead\w*|fold\w*|push\w*|pull\w*|throw\w*|"
              r"toss\w*|roll\w*|assembl\w*|paint\w*|dig\w*|wash\w*|hammer\w*|squeez\w*|spread\w*|flip\w*|whisk\w*|sew\w*|"
              r"knit\w*|typing|writing|drawing|cooking|baking|hands?)\b",
}
NEG = (r"\b(?:aerial|drone|birds? eye|timelapse|time.lapse|slow motion|slowmo|cityscape|landscape|sky|clouds|abstract|"
       r"animation|3d|render)")
TARGET_FPS = 12.0
UA = {"User-Agent": "Mozilla/5.0"}


def load_meta():
    p = hf_hub_download("jovianzm/Pexels-400k", "pexels_400k.parquet", repo_type="dataset", local_dir="data/pexels")
    df = pd.read_parquet(p)
    df["caption"] = df["title"].str.replace(r"\s*·\s*Free Stock Video\s*$", "", regex=True)
    df["id"] = df["video"].str.extract(r"/(\d+)$")[0]
    return df


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=avg_frame_rate,width,height,nb_read_packets", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(float, s["avg_frame_rate"].split("/"))
    return num / den, int(s["width"]), int(s["height"]), int(s["nb_read_packets"])


def fetch(row, vid_dir, seconds):
    dst = os.path.join(vid_dir, f"{row.id}.mp4")
    try:
        if not os.path.exists(dst):
            url = requests.head(row.video, headers=UA, allow_redirects=True, timeout=30).url
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-user_agent", UA["User-Agent"], "-t", str(seconds),
                            "-i", url, "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", dst + ".tmp.mp4"],
                           check=True, timeout=300)
            os.replace(dst + ".tmp.mp4", dst)
        fps, w, h, n = probe(dst)
        return dict(id=row.id, fps=fps, W=w, H=h, n_frames=n)
    except Exception as e:  # noqa: BLE001
        print(f"{row.id}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cat", type=int, default=130)
    ap.add_argument("--seconds", type=float, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--exclude", nargs="*", default=[], help="selected.csv files whose ids to skip")
    ap.add_argument("--out", default="data/pexels_pilot")
    args = ap.parse_args()
    os.chdir(ROOT)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    df = load_meta()
    df = df[df["sfw"] & (df["duration"] >= 6) & ~df["caption"].str.contains(NEG, case=False)]
    for f in args.exclude:
        df = df[~df["id"].isin(pd.read_csv(f, dtype={"id": str})["id"])]
    rng = np.random.default_rng(args.seed)
    picks, taken = [], set()
    for cat, pat in CATS.items():
        c = df[df["caption"].str.contains(pat, case=False) & ~df["id"].isin(taken)]
        c = c.iloc[rng.choice(len(c), min(args.per_cat, len(c)), replace=False)].assign(bucket=cat)
        taken |= set(c["id"])
        picks.append(c)
        print(f"{cat}: {len(c)} picked", flush=True)
    sel = pd.concat(picks)

    with ThreadPoolExecutor(args.workers) as ex:
        info = [r for r in ex.map(lambda r: fetch(r, vid_dir, args.seconds), sel.itertuples()) if r]
    sel = sel.merge(pd.DataFrame(info), on="id")
    sel["skip"] = (sel["fps"] / TARGET_FPS).round().clip(lower=1).astype(int)
    sel["out_fps"] = sel["fps"] / sel["skip"]
    sel["frame"] = sel["n_frames"]
    sel = sel[sel["out_fps"].between(10, 16) & (sel["n_frames"] >= 64 * sel["skip"] + 1)
              & (sel["W"] >= 1.3 * sel["H"])]  # landscape only (training crops 16:9)
    sel = sel.assign(video=sel["id"] + ".mp4", motion_score=np.nan)
    sel = sel.rename(columns={"motion_score": "motion score"})
    sel.to_csv(os.path.join(args.out, "selected.csv"), index=False)
    print(f"saved {len(sel)} usable -> {args.out}/selected.csv", flush=True)


if __name__ == "__main__":
    main()
