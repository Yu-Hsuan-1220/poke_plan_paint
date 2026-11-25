"""Download a large random OpenVid-1M sample for the training set (candidates; filtering happens after tracking).

Pool: camera motion == static, one 65-frame window at ~12 fps fits (64 * skip + 1 raw frames, 10-16 fps after
skipping), not celebv / magictime (talking faces / time-lapse), 1.5 <= motion score < 25 (>= 25 was the most
static bucket in docs/openvid_check.md; below that the score does not predict motion, so it is not used further).
Captions about talking / interviews / news are ~40% of the pool; they are capped at --talk_frac of the sample.

All single-zip parts are listed in parallel (each listing reads the zip's central directory, ~20 s), the sample
is drawn uniformly from the pool members found, and each part's members are extracted by one thread with HTTP
range reads. Resumable: existing videos are skipped.

    python src/openvid_fetch_train.py --n 5500 --out data/openvid_train
    python src/openvid_fetch_train.py --from_csv datasets/clips.csv --out data/openvid_rebuild   # re-fetch exactly
        the train + heldout videos of the project (no filtering / sampling)
    -> data/openvid_train/videos/*.mp4, data/openvid_train/selected.csv (same columns as openvid_fetch.py)
"""

import argparse
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from common import ROOT
from openvid_fetch import BUCKETS, REPO, load_meta

TALK = r"\b(?:talk|talking|speak|speaking|interview|news|podcast|conversation|presenter|anchor|lecture|discuss)"
# --clean: captions that describe someone addressing the camera, or non-real footage (2026-11-20 check: of the
# candidates without these words, 9% end up in the training set, vs 1-2% of the rest after the face filter)
CLEAN_NEG = (r"\b(?:talk|talking|speak|speaking|interview|news|podcast|conversation|presenter|anchor|lecture|discuss|"
             r"explain\w*|microphone|addressing|reporter|host|camera|gestur\w*|expression\w*|commentary|vlog|selfie|"
             r"mouth|smil\w*|animat\w*|cartoon|3d|cgi|render\w*|video game|gameplay|text|title|logo|slideshow)")


def list_part(fs, path):
    for attempt in range(3):
        try:
            z = zipfile.ZipFile(fs.open(path, "rb", block_size=1 << 22))
            return path, z, {os.path.basename(i.filename): i for i in z.infolist() if i.filename.endswith(".mp4")}
        except Exception as e:  # noqa: BLE001  (network hiccups)
            print(f"{path}: listing failed ({e}), retry {attempt + 1}", flush=True)
            time.sleep(10)
    return path, None, {}


def extract(z, infos, vid_dir):
    n = 0
    for info in infos:
        dst = os.path.join(vid_dir, os.path.basename(info.filename))
        if os.path.exists(dst):
            continue
        for attempt in range(3):
            try:
                with open(dst + ".tmp", "wb") as f:
                    f.write(z.read(info))
                os.replace(dst + ".tmp", dst)
                n += 1
                break
            except Exception as e:  # noqa: BLE001
                print(f"{info.filename}: {e}, retry {attempt + 1}", flush=True)
                time.sleep(10)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5500)
    ap.add_argument("--talk_frac", type=float, default=0.2)
    ap.add_argument("--ms_min", type=float, default=1.5)
    ap.add_argument("--ms_max", type=float, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/openvid_train")
    ap.add_argument("--clean", action="store_true", help="drop CLEAN_NEG captions (then --talk_frac is ignored)")
    ap.add_argument("--exclude", nargs="*", default=[], help="selected.csv files whose videos to skip")
    ap.add_argument("--from_csv", default=None, help="datasets/clips.csv: fetch its OpenVid train + heldout videos")
    args = ap.parse_args()
    os.chdir(ROOT)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    df = load_meta()
    want = None
    if args.from_csv:
        c = pd.read_csv(args.from_csv)
        want = set(c.loc[c["split"].isin(["train", "heldout"]), "source_video"])
        df = df[df["video"].isin(want)].copy()
        print(f"--from_csv: {len(want)} videos wanted, {len(df)} in the OpenVid-1M metadata", flush=True)
    ok = (df["camera motion"] == "static") & (df["frame"] >= 64 * df["skip"] + 1) & df["out_fps"].between(10, 16) \
        & ~df["video"].str.startswith(("celebv", "magictime")) \
        & (df["motion score"] >= args.ms_min) & (df["motion score"] < args.ms_max)
    if args.clean:
        ok &= ~df["caption"].str.contains(CLEAN_NEG, case=False)
        args.talk_frac = 0.0
    for f in args.exclude:
        ok &= ~df["video"].isin(pd.read_csv(f)["video"])
    if want is None:
        df = df[ok].copy()
        print(f"pool: {len(df)}", flush=True)

    fs = HfFileSystem()
    paths = sorted(f["name"] for f in fs.ls(REPO, detail=True)
                   if re.search(r"OpenVid_part\d+\.zip$", f["name"]))
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        listed = list(ex.map(lambda p: list_part(fs, p), paths))
    zips, member = {}, {}
    for path, z, infos in listed:
        zips[path] = z
        member.update({k: (path, v) for k, v in infos.items()})
    print(f"listed {len(paths)} parts in {time.time() - t0:.0f}s, {len(member)} videos", flush=True)

    df = df[df["video"].isin(member)]
    df["talk"] = df["caption"].str.contains(TALK, case=False)
    if want is not None:
        sel = df.copy()
        if len(sel) < len(want):
            print(f"WARNING: {len(want) - len(sel)} wanted videos not found in any part", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        n_talk = int(round(args.n * args.talk_frac))
        picks = []
        for flag, n in ((True, n_talk), (False, args.n - n_talk)):
            g = df[df["talk"] == flag]
            picks.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)])
        sel = pd.concat(picks).sample(frac=1, random_state=args.seed)
    sel["bucket"] = pd.cut(sel["motion score"], [b[0] for b in BUCKETS] + [BUCKETS[-1][1]], right=False,
                           labels=[f"{lo:g}-{hi:g}" for lo, hi in BUCKETS]).astype(str)
    sel["part"] = sel["video"].map(lambda v: member[v][0].split("/")[-1])
    sel.to_csv(os.path.join(args.out, "selected.csv"), index=False)
    print(f"pool members in listed parts: {len(df)}; picked {len(sel)} ({sel['talk'].mean():.0%} talk), "
          f"{sel['part'].nunique()} parts -> {args.out}/selected.csv", flush=True)

    by_part = {}
    for v in sel["video"]:
        path, info = member[v]
        by_part.setdefault(path, []).append(info)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        for n in ex.map(lambda kv: extract(zips[kv[0]], kv[1], vid_dir), by_part.items()):
            done += n
            print(f"{done} new videos, {time.time() - t0:.0f}s", flush=True)
    print(f"DONE: {len([f for f in os.listdir(vid_dir) if f.endswith('.mp4')])} videos in {vid_dir}", flush=True)


if __name__ == "__main__":
    main()
