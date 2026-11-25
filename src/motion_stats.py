"""Object motion vs camera motion per window, from the saved CoTracker3 tracks (no model, CPU only).

For t in {16, 32, 48, 63}: fit a similarity (shift + zoom + rotation) from frame 0 to frame t with RANSAC on
the valid tracks (the background = the camera), then
    cam_disp  = median displacement the camera transform gives the tracks            (camera motion)
    residual  = |x_t - M_t x_0| per track                                             (object motion)
and take the max over t. Units: ZipMo's [0,128] scale (same as zipmo_recon_check.py).
    obj_frac_moving  fraction of valid tracks with residual > 2 (something besides the camera moves there)
    obj_p90          90th-percentile residual (how strongly the moving part moves)
    cam_disp         camera displacement
A window with large cam_disp but tiny obj_frac_moving is the "camera moves, nothing else does" case.

    python src/motion_stats.py --dirs data/koala_train_v2 data/openvid_yt/windows --names koala openvid_yt
    -> outputs/motion_stats.csv
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

from common import ROOT
from tracking import track_mask

TS = (16, 32, 48, 63)
MOVING = 2.0


def window_stats(wdir):
    meta = json.load(open(os.path.join(wdir, "meta.json")))
    tr = np.load(os.path.join(wdir, "tracks_yx.npy")).astype(np.float32)
    vis = np.load(os.path.join(wdir, "visibility.npy")).astype(np.float32)
    conf = np.load(os.path.join(wdir, "confidence.npy")).astype(np.float32)
    m = track_mask(torch.from_numpy(vis), torch.from_numpy(conf), torch.from_numpy(tr)).numpy()
    row = dict(window=meta["window"], n_valid=int(m.sum()))
    if m.sum() < 20:
        return row
    p = (tr[:, m] + 1) / 2 * 128  # [T,N,2] on the [0,128] scale (y, x)
    res, cam = np.zeros(p.shape[1]), 0.0
    for t in TS:
        M, _ = cv2.estimateAffinePartial2D(p[0, :, ::-1].copy(), p[t, :, ::-1].copy(), method=cv2.RANSAC,
                                           ransacReprojThreshold=1.0)
        if M is None:
            continue
        pred = p[0, :, ::-1] @ M[:, :2].T + M[:, 2]
        res = np.maximum(res, np.linalg.norm(p[t, :, ::-1] - pred, axis=1))
        cam = max(cam, float(np.median(np.linalg.norm(pred - p[0, :, ::-1], axis=1))))
    return row | dict(obj_frac_moving=round(float((res > MOVING).mean()), 3),
                      obj_p90=round(float(np.percentile(res, 90)), 3), cam_disp=round(cam, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/motion_stats.csv")
    args = ap.parse_args()
    os.chdir(ROOT)
    rows = []
    for root, name in zip(args.dirs, args.names):
        for w in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "meta.json"))):
            if os.path.exists(os.path.join(w, "tracks_yx.npy")):
                rows.append(window_stats(w) | dict(dataset=name))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("dataset")[["n_valid", "obj_frac_moving", "obj_p90", "cam_disp"]].median())


if __name__ == "__main__":
    main()
