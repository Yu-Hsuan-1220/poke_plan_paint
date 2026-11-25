"""Pick OpenVid-1M videos from the metadata and pull only those files out of the remote zip parts.

OpenVid-1M ships ~186 zip parts of 20-50 GB each. Python's zipfile works on HfFileSystem's seekable remote file,
so we read a part's central directory (~20 s) and then extract single members with HTTP range requests; no
part is downloaded in full.

Selection: camera motion == "static", long enough for one 65-frame window at ~12 fps
(skip = round(fps / 12), needs 64 * skip + 1 raw frames), stratified by OpenVid's motion score (mean optical
flow) so we can see how data quality changes with the amount of motion.

    python src/openvid_fetch.py --parts 0 1 2 3 4 5 --per_bucket 30
    python src/openvid_fetch.py --parts 20 50 80 110 140 170 --exclude_prefix celebv magictime --out data/openvid_yt
Pexels / pixabay / mixkit clips are 64 frames at 24 fps, too short for 65 frames at ~12 fps, so they never qualify.
    -> data/openvid/videos/*.mp4, data/openvid/selected.csv
"""

import argparse
import os
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

from common import ROOT

REPO = "datasets/nkp37/OpenVid-1M"
TARGET_FPS = 12.0
BUCKETS = [(1.5, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)]  # motion score ranges


def frame_skip(fps):
    return max(1, round(fps / TARGET_FPS))


def load_meta():
    p = hf_hub_download("nkp37/OpenVid-1M", "data/train/OpenVid-1M.csv", repo_type="dataset",
                        local_dir="data/openvid")
    df = pd.read_csv(p)
    df["skip"] = df["fps"].map(frame_skip)
    df["out_fps"] = df["fps"] / df["skip"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--per_bucket", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/openvid")
    ap.add_argument("--exclude_prefix", nargs="*", default=[],
                    help="e.g. celebv magictime (talking heads / time-lapse sources at the start of the parts)")
    args = ap.parse_args()
    os.chdir(ROOT)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    df = load_meta()
    ok = (df["camera motion"] == "static") & (df["frame"] >= 64 * df["skip"] + 1) & \
         df["out_fps"].between(10, 16)
    for pre in args.exclude_prefix:
        ok &= ~df["video"].str.startswith(pre)
    df = df[ok]
    print(f"static, long enough, 10-16 fps after skipping: {len(df)}")

    fs = HfFileSystem()
    members = {}  # basename -> (part file, ZipInfo)
    for k in args.parts:
        path = f"{REPO}/OpenVid_part{k}.zip"
        if not fs.exists(path):
            print(f"part {k}: not a single zip (split into _partaa/_partab), skipped")
            continue
        z = zipfile.ZipFile(fs.open(path, "rb", block_size=1 << 20))
        for info in z.infolist():
            if info.filename.endswith(".mp4"):
                members[os.path.basename(info.filename)] = (path, info)
        print(f"part {k}: {len(members)} videos listed so far", flush=True)
    df = df[df["video"].isin(members)]
    print(f"candidates inside the listed parts: {len(df)}")

    rng = np.random.default_rng(args.seed)
    picks = []
    for lo, hi in BUCKETS:
        b = df[(df["motion score"] >= lo) & (df["motion score"] < hi)]
        n = min(args.per_bucket, len(b))
        picks.append(b.iloc[rng.choice(len(b), n, replace=False)].assign(bucket=f"{lo:g}-{hi:g}"))
        print(f"motion {lo:g}-{hi:g}: {len(b)} candidates, picked {n}")
    sel = pd.concat(picks)

    opened = {}
    for i, r in enumerate(sel.itertuples()):
        dst = os.path.join(vid_dir, r.video)
        if os.path.exists(dst):
            continue
        part, info = members[r.video]
        if part not in opened:
            opened[part] = zipfile.ZipFile(fs.open(part, "rb", block_size=1 << 22))
        with open(dst + ".tmp", "wb") as f:
            f.write(opened[part].read(info))
        os.replace(dst + ".tmp", dst)
        print(f"[{i + 1}/{len(sel)}] {r.video} ({info.file_size / 1e6:.1f} MB)", flush=True)
    sel.to_csv(os.path.join(args.out, "selected.csv"), index=False)
    print(f"saved {len(sel)} -> {args.out}/selected.csv")


if __name__ == "__main__":
    main()
