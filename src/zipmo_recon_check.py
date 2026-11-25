"""How well does the ZipMo VAE reconstruct a dataset's motion? Same protocol for every dataset.

Per window: take 80 valid CoTracker3 tracks (visible at t=0, confidence > 0.5 for all 64 steps, always in frame),
encode 40 of them (+ first frame) into z, decode z at the start points of all 80, and compare with CoTracker3:
  held  = the 40 tracks the encoder did NOT see (the real test: does z describe the scene's motion?)
  seen  = the 40 tracks it did see (pure reconstruction)
  static baseline = every point stays at its start (error = how far the GT moves)
Distances are on ZipMo's [0, 128] scale (both axes), like the ZipMo paper and check_z_consistency.py.
Because error grows with the amount of motion, compare `ratio_end` = held EPE / static baseline, too.
The window-level data statistics (static-camera fraction, camera rules, how much the valid tracks move) come
along so data quality can be compared at the same time.

    python src/zipmo_recon_check.py --dirs data/koala_train_v2 data/openvid/windows --names koala openvid
    -> outputs/zipmo_recon/{per_window.csv, summary.txt, vis/*.jpg}
"""

import argparse
import glob
import json
import os
import zlib

import cv2
import numpy as np
import pandas as pd
import torch  # before decord

import decord

from common import ROOT, T_STEPS, load_planner
from tracking import track_mask

N_ENC = 40
MOVING = 2.0  # [0,128] units of end-point displacement (~1.6% of the frame)


def to128(a, b):
    return ((a - b) / 2 * 128).norm(dim=-1)  # [..., T]


@torch.no_grad()
def check_window(planner, wdir, dev):
    meta = json.load(open(os.path.join(wdir, "meta.json")))
    tr = torch.from_numpy(np.load(os.path.join(wdir, "tracks_yx.npy")).astype(np.float32))  # [64,N,2]
    vis = torch.from_numpy(np.load(os.path.join(wdir, "visibility.npy")).astype(np.float32))
    conf = torch.from_numpy(np.load(os.path.join(wdir, "confidence.npy")).astype(np.float32))
    cam = meta.get("camera_filter", {})
    valid = torch.nonzero(track_mask(vis, conf, tr)).squeeze(1)
    allv = tr[:, valid].permute(1, 0, 2)  # [Nv,64,2]
    disp_all = to128(allv[:, -1], allv[:, 0]) if len(valid) else torch.zeros(1)
    row = dict(window=meta["window"], H=meta["H"], W=meta["W"], fps=round(meta["fps"], 2),
               static_frac=meta.get("static_frac"), n_valid=len(valid),
               cam_motion=cam.get("cam_motion"), zoom_total=cam.get("zoom_total"), cut_ratio=cam.get("cut_ratio"),
               camera_ok=cam.get("camera_ok"), motion_score=meta.get("motion_score"), bucket=meta.get("bucket"),
               data_disp_med=round(float(disp_all.median()), 3),
               data_frac_moving=round(float((disp_all > MOVING).float().mean()), 3))
    if len(valid) < 2 * N_ENC:
        return row, None

    g = torch.Generator().manual_seed(zlib.crc32(meta["window"].encode()))
    pick = valid[torch.randperm(len(valid), generator=g)[:2 * N_ENC]]
    gt = tr[:, pick].permute(1, 0, 2).to(dev)  # [80,64,2]; first 40 = encoded, last 40 = held out
    frame0 = decord.VideoReader(os.path.join(wdir, "frames.mp4"))[0].asnumpy()
    dt = next(planner.parameters()).dtype
    sf = (torch.from_numpy(frame0).to(dev).float() / 127.5 - 1)[None].to(dt)
    emb = planner.vae.img_embedder(sf)
    z = planner.vae.encode(gt[None, :N_ENC].to(dt), start_emb=emb).mean
    pred = planner.vae.decode(z, query_pos=gt[None, :, 0].to(dt), points_per_track=T_STEPS,
                              start_emb=emb)[0].float()  # [80,64,2]

    d = to128(pred, gt)  # [80,64]
    stat = to128(gt[:, :1].expand_as(gt), gt)
    seen, held = slice(0, N_ENC), slice(N_ENC, 2 * N_ENC)
    mov = stat[held, -1] > MOVING
    f = lambda x: round(float(x), 3)
    row |= dict(held_epe=f(d[held].mean()), held_epe_end=f(d[held, -1].mean()),
                seen_epe_end=f(d[seen, -1].mean()), static_end=f(stat[held, -1].mean()),
                ratio_end=f(d[held, -1].mean() / max(float(stat[held, -1].mean()), 0.5)),
                n_moving=int(mov.sum()),
                held_epe_end_moving=f(d[held][mov, -1].mean()) if mov.any() else None,
                static_end_moving=f(stat[held][mov, -1].mean()) if mov.any() else None)
    return row, (frame0, gt.cpu(), pred.cpu())


def draw(frame0, gt, pred, text):
    """GT (green) vs ZipMo reconstruction (red) for the 40 held-out tracks, on the first frame."""
    H, W = frame0.shape[:2]
    s = 480 / W
    img = cv2.resize(frame0, (480, round(H * s)))
    px = lambda t: np.ascontiguousarray(((t.flip(-1) + 1) / 2 * torch.tensor([W - 1, H - 1]) * s).numpy(), np.int32)
    for tr, col in ((gt[N_ENC:], (0, 255, 0)), (pred[N_ENC:], (255, 60, 60))):
        for p in px(tr):
            cv2.polylines(img, [p.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in p[-1]), 2, col, -1)
    cv2.putText(img, text, (5, 18), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def summarize(df):
    cols = ["held_epe", "held_epe_end", "seen_epe_end", "static_end", "ratio_end", "held_epe_end_moving",
            "static_end_moving", "data_disp_med", "data_frac_moving", "static_frac", "cam_motion"]
    rows = []
    groups = [("all", df), ("ZipMo static>0.5", df[df.static_frac > 0.5]),
              ("static>0.5 & camera_ok", df[(df.static_frac > 0.5) & (df.camera_ok == True)])]  # noqa: E712
    for gname, gdf in groups:
        for ds, d in gdf.groupby("dataset"):
            r = dict(subset=gname, dataset=ds, n_windows=len(d), n_with_80=int(d.held_epe.notna().sum()))
            r |= {c: round(float(d[c].median()), 3) for c in cols}
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None, help="max windows per dataset")
    ap.add_argument("--n_vis", type=int, default=12)
    ap.add_argument("--out", default="outputs/zipmo_recon")
    ap.add_argument("--device", default="cuda", help="cpu works too (fp32, ~1-2 s/window)")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    dev = args.device
    planner = load_planner(dev, torch.bfloat16 if dev.startswith("cuda") else torch.float32)

    rows = []
    for root, name in zip(args.dirs, args.names):
        wdirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "tracks_yx.npy")))[:args.n]
        rng = np.random.default_rng(0)
        vis_set = set(rng.choice(len(wdirs), min(args.n_vis, len(wdirs)), replace=False).tolist())
        for i, w in enumerate(wdirs):
            row, out = check_window(planner, w, dev)
            rows.append(row | dict(dataset=name))
            if out is not None and i in vis_set:
                txt = f"{name} held EPE_end {row['held_epe_end']:.1f} static {row['static_end']:.1f}"
                cv2.imwrite(os.path.join(args.out, "vis", f"{name}_{row['window']}.jpg"),
                            cv2.cvtColor(draw(*out, txt), cv2.COLOR_RGB2BGR))
            if i % 100 == 0:
                print(f"{name} {i}/{len(wdirs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_window.csv"), index=False)
    s = summarize(df)
    txt = "medians over windows; distances on ZipMo's [0,128] scale\n" + s.to_string(index=False)
    if df.bucket.notna().any():
        b = df[df.held_epe.notna()].groupby(["dataset", "bucket"])[
            ["held_epe_end", "static_end", "ratio_end", "data_frac_moving", "static_frac", "cam_motion"]].median()
        b["n"] = df.groupby(["dataset", "bucket"]).size()
        b["frac_static>0.5"] = df.groupby(["dataset", "bucket"]).static_frac.apply(lambda x: (x > 0.5).mean())
        b["frac_camera_ok"] = df.groupby(["dataset", "bucket"]).camera_ok.apply(lambda x: (x == True).mean())  # noqa
        txt += "\n\nby OpenVid motion-score bucket (all windows)\n" + b.round(3).to_string()
    open(os.path.join(args.out, "summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
