"""OpenVid videos -> Koala-format windows (same layout as koala_train_v2, so every later script works on both).

One window per video, from its first frame: 65 frames taken every `skip` raw frames (~12 fps), CoTracker3 on
the first 64 with 1,024 random queries. Every window is written (nothing is dropped here); the ZipMo
static-camera fraction, the track-mask count and the three camera rules go into meta.json so filters can be
applied (and changed) later.
With --n_z K (default 3), windows with >= 80 in-frame valid tracks also get enc_idx.npy [K,40] and z.npy
[K,256,16] (normalized), exactly as Spencer's preprocess_koala.py (seed crc32(name)+seed+1), so
src/prep_koala.py can read them.

    python src/openvid_windows.py                 # data/openvid/videos -> data/openvid/windows/<name>_w0/
"""

import argparse
import json
import os
import zlib

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch  # before decord

import decord

from common import ROOT, N_FRAMES, load_planner
from tracking import (T_ZIPMO, camera_rules, camera_track_metrics, cut_ratio, face_area, load_cotracker,
                      static_fraction, track_mask, track_points, xy_to_yx_norm)


def write_mp4(path, frames, fps):
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
                            ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            w.append_data(f)


N_ENC, N_MIN = 40, 80
MAX_W = 1920


@torch.no_grad()
def encode_z(planner, name, tyx, mask, frame0, n_z, seed, dev):
    """Spencer's encode_window: n_z random 40-track subsets of the in-frame valid tracks -> normalized z."""
    g = torch.Generator().manual_seed(zlib.crc32(name.encode()) + seed + 1)
    valid = torch.nonzero(mask).squeeze(1).cpu()
    enc_idx = torch.stack([valid[torch.randperm(len(valid), generator=g)[:N_ENC]] for _ in range(n_z)])
    tracks_enc = tyx.permute(1, 0, 2)[enc_idx.to(dev)]  # [n_z,40,64,2]
    start = (torch.from_numpy(frame0).to(dev).float() / 127.5 - 1.0)[None].expand(n_z, -1, -1, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tyx.is_cuda):
        z = planner.normalize_latents(planner.vae.encode(tracks_enc, start_frame=start).mean)
    return enc_idx.numpy().astype(np.int16), z.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/openvid")
    ap.add_argument("--n_queries", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n_z", type=int, default=3, help="0 = no z")
    ap.add_argument("--max_face", type=float, default=None,
                    help="skip CoTracker when the largest frontal face covers more than this fraction (meta only)")
    args = ap.parse_args()
    os.chdir(ROOT)
    sel = pd.read_csv(os.path.join(args.root, "selected.csv"))
    sel.columns = [c.replace(" ", "_") for c in sel.columns]
    dev = args.device
    cot = load_cotracker(dev)
    planner = load_planner(dev, torch.bfloat16 if dev.startswith("cuda") else torch.float32) if args.n_z else None
    for i, r in enumerate(sel.itertuples()):
        if i % args.n_shards != args.shard:
            continue
        src = os.path.join(args.root, "videos", r.video)
        name = os.path.splitext(r.video)[0] + "_w0"
        out = os.path.join(args.root, "windows", name)
        if os.path.exists(os.path.join(out, "meta.json")) or not os.path.exists(src):
            continue
        vr = decord.VideoReader(src, num_threads=4)
        skip = int(r.skip)
        idx = list(range(0, (N_FRAMES - 1) * skip + 1, skip))
        if idx[-1] >= len(vr):
            print(f"{name}: only {len(vr)} frames, skipped", flush=True)
            continue
        frames = vr.get_batch(idx).asnumpy()  # [65,H,W,3]
        if frames.shape[2] > MAX_W:  # 4K stock footage -> 1920 wide (tracks are normalized; saves decode/VAE time)
            h = int(round(frames.shape[1] * MAX_W / frames.shape[2] / 2)) * 2
            frames = np.stack([cv2.resize(f, (MAX_W, h), interpolation=cv2.INTER_AREA) for f in frames])
        T, H, W, _ = frames.shape
        face = face_area(frames)
        if args.max_face is not None and face > args.max_face:  # talking head: no tracks, meta records why
            os.makedirs(out, exist_ok=True)
            json.dump(dict(window=name, video=r.video, source="openvid", H=H, W=W, fps=float(r.fps) / skip,
                           caption=r.caption, face_area=face, skipped="face"),
                      open(os.path.join(out, "meta.json"), "w"), indent=1, ensure_ascii=False)
            continue

        g = torch.Generator().manual_seed(zlib.crc32(name.encode()) + args.seed)
        q = torch.rand(args.n_queries, 2, generator=g) * torch.tensor([W - 1, H - 1])
        txy, vis, conf = track_points(cot, torch.from_numpy(frames[:T_ZIPMO]).to(dev), q.to(dev))
        tyx = xy_to_yx_norm(txy, H, W)

        os.makedirs(out, exist_ok=True)
        tyx_np, vis_np, conf_np = (x.cpu().numpy().astype(np.float16) for x in (tyx, vis, conf))
        cam = camera_track_metrics(tyx_np, vis_np, conf_np, H, W) | dict(cut_ratio=cut_ratio(frames))
        rules = camera_rules(cam)
        meta = dict(window=name, video=r.video, source="openvid", H=H, W=W, fps=float(r.fps) / skip,
                    raw_fps=float(r.fps), frame_skip=skip, caption=r.caption, motion_score=float(r.motion_score),
                    bucket=r.bucket, face_area=face, static_frac=round(static_fraction(tyx), 4),
                    n_valid_tracks=int(track_mask(vis, conf).sum()),
                    n_valid_in_frame=int(track_mask(vis, conf, tyx).sum()),
                    camera_filter=cam | dict(camera_ok=not rules, rejected_by=",".join(rules)))
        write_mp4(os.path.join(out, "frames.mp4"), frames, meta["fps"])
        np.save(os.path.join(out, "tracks_yx.npy"), tyx_np)
        np.save(os.path.join(out, "visibility.npy"), vis_np)
        np.save(os.path.join(out, "confidence.npy"), conf_np)
        meta["n_z"] = 0
        if planner is not None and meta["n_valid_in_frame"] >= N_MIN:
            mask = track_mask(vis, conf, tyx)
            enc_idx, z = encode_z(planner, name, tyx.float(), mask, frames[0], args.n_z, args.seed, dev)
            np.save(os.path.join(out, "enc_idx.npy"), enc_idx)
            np.save(os.path.join(out, "z.npy"), z)
            meta["n_z"] = args.n_z
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, ensure_ascii=False)
        print(f"[{i + 1}/{len(sel)}] {name} {W}x{H} skip {skip} | static {meta['static_frac']:.2f} | "
              f"valid {meta['n_valid_in_frame']} | cam {cam} {rules}", flush=True)


if __name__ == "__main__":
    main()
