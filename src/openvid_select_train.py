"""Pick the OpenVid training windows from the tracked candidates (src/openvid_fetch_train.py -> openvid_windows.py).

A window is kept when it passes
    ZipMo's filters       static_frac > 0.5, >= 80 in-frame valid tracks (so it has z.npy)
    Spencer's camera rules R1-R3 (camera_ok)
    an object-motion check obj_frac_moving >= 0.2 and obj_p90 >= 4  (src/motion_stats.py; [0,128] scale)
and, optionally (the "clean" set, 2026-11-20),
    --clean      no caption words about talking to / addressing the camera or non-real footage (CLEAN_NEG)
    --max_face   largest frontal face <= this fraction of the frame (tracking.face_area)
    --max_cam    camera displacement cam_disp < this ([0,128] scale)
then at most --per_source windows per YouTube source video and at most --talk_frac talking / interview /
news captions. --n windows are drawn at random (seeded) from what is left.

    python src/openvid_select_train.py --roots data/openvid_train --n 2000 --out data/openvid_train
    python src/openvid_select_train.py --roots data/openvid_train data/openvid_clean --clean --max_face 0.02 \
        --max_cam 2.5 --talk_frac 0 --out data/openvid_clean_set
    -> <out>/windows/train_windows.csv  (read by src/prep_koala.py --src <out>/windows; when <out> is not the
                                         single root, <out>/windows/<window> are symlinks to the picked windows)
       <out>/all_windows.csv            (every window with its stats and the reason it was dropped)
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import ROOT
from motion_stats import window_stats
from openvid_fetch_train import CLEAN_NEG, TALK
from tracking import face_area


def load(w):
    m = json.load(open(os.path.join(w, "meta.json"))) | dict(path=os.path.abspath(w))
    if m.get("skipped"):  # face prefilter in openvid_windows.py: no tracks
        return m
    if "face_area" not in m:
        import decord
        m["face_area"] = face_area(decord.VideoReader(os.path.join(w, "frames.mp4")).get_batch([0, 32, 64]).asnumpy(),
                                   at=(0, 1, 2))
    cam = m.pop("camera_filter")
    return m | {k: cam[k] for k in ("cam_motion", "zoom_total", "cut_ratio", "camera_ok", "rejected_by")} \
        | window_stats(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/openvid_train"])
    ap.add_argument("--out", default=None, help="default: the single root")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--max_face", type=float, default=None)
    ap.add_argument("--max_cam", type=float, default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--talk_frac", type=float, default=0.2)
    ap.add_argument("--per_source", type=int, default=2)
    ap.add_argument("--obj_frac", type=float, default=0.2)
    ap.add_argument("--obj_p90", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    out = args.out or args.roots[0]
    wroot = os.path.join(out, "windows")
    os.makedirs(wroot, exist_ok=True)
    wins = sorted(os.path.dirname(p) for r in args.roots for p in glob.glob(os.path.join(r, "windows", "*", "meta.json")))
    with ProcessPoolExecutor(32) as ex:
        df = pd.DataFrame(list(ex.map(load, wins, chunksize=16)))

    vid = df["video"].str.replace(".mp4", "", regex=False)
    df["source_id"] = vid.str.extract(r"^(.*)_\d+_\d+to\d+$")[0].fillna(vid)  # YouTube id; else the video
    df["talk"] = df["caption"].str.contains(TALK, case=False)
    df["clean"] = ~df["caption"].str.contains(CLEAN_NEG, case=False)
    df["drop"] = ""
    for cond, why in [(df["skipped"].notna() if "skipped" in df else False, "face"),
                      (~df["clean"] if args.clean else False, "caption"),
                      (df["face_area"] > args.max_face if args.max_face is not None else False, "face"),
                      (df["static_frac"] <= 0.5, "static_frac"),
                      (df["n_z"].fillna(0) == 0, "few_tracks"),
                      (~df["camera_ok"].astype(bool), "camera"),
                      ((df["obj_frac_moving"].fillna(0) < args.obj_frac) | (df["obj_p90"].fillna(0) < args.obj_p90),
                       "low_object_motion"),
                      (df["cam_disp"].fillna(1e9) >= args.max_cam if args.max_cam is not None else False, "camera_disp")]:
        df.loc[(df["drop"] == "") & cond, "drop"] = why

    rng = np.random.default_rng(args.seed)
    ok = df[df["drop"] == ""].sample(frac=1, random_state=args.seed)
    ok = ok[ok.groupby("source_id").cumcount() < args.per_source]
    n_talk = min(int(ok["talk"].sum()), int(round(args.n * args.talk_frac)))
    n_rest = min(int((~ok["talk"]).sum()), args.n - n_talk)
    pick = pd.concat([ok[ok["talk"]].head(n_talk), ok[~ok["talk"]].head(n_rest)])
    df.loc[(df["drop"] == "") & ~df["window"].isin(pick["window"]), "drop"] = "not_sampled"
    df.to_csv(os.path.join(out, "all_windows.csv"), index=False)
    for p in pick["path"]:  # windows from other roots -> symlinks, so prep_koala.py sees one --src directory
        dst = os.path.join(wroot, os.path.basename(p))
        if os.path.abspath(os.path.dirname(p)) != os.path.abspath(wroot) and not os.path.lexists(dst):
            os.symlink(p, dst)

    pick = pick.sort_values("window").assign(dir="openvid", videoID=pick["video"].str.replace(".mp4", "", regex=False),
                                             youtube_id=pick["source_id"], k=0)
    cols = ["window", "dir", "videoID", "youtube_id", "k", "H", "W", "fps", "frame_skip", "n_valid_in_frame",
            "motion_score", "static_frac", "obj_frac_moving", "obj_p90", "cam_disp", "face_area", "talk", "caption"]
    pick[cols].to_csv(os.path.join(wroot, "train_windows.csv"), index=False)

    print(f"windows: {len(df)}")
    print(df["drop"].replace("", "kept").value_counts().to_string())
    print(f"picked {len(pick)} ({pick['talk'].mean():.0%} talk, {pick['source_id'].nunique()} source videos) "
          f"-> {wroot}/train_windows.csv")
    print("picked medians:", pick[["obj_frac_moving", "obj_p90", "cam_disp", "face_area", "static_frac", "motion_score"]]
          .median().round(3).to_dict())
    if len(pick) < args.n:
        print(f"WARNING: only {len(pick)} < {args.n} windows pass; fetch more candidates")


if __name__ == "__main__":
    main()
