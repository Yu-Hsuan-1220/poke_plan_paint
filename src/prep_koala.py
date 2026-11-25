"""Koala window folders -> training clips for Wan2.2-I2V-A14B.

Input:  data/koala_train_v2/<window>/{frames.mp4, z.npy, enc_idx.npy, tracks_yx.npy, visibility.npy, confidence.npy, meta.json}
        (65 frames at original resolution, ~12-15 fps; z normalized with planner.normalize_latents)
Output: data/clips_a14b/<window>.pt with
    first_frame  [320,576,3] uint8           center crop of frame 0
    latents      [16,17,40,72] bf16          Wan2.1 VAE latents of the 65 cropped frames (normalized) = target
    cond_lat     [16,17,40,72] bf16          the pipeline's I2V condition latent VAE([frame0, 0, ..., 0])
    dec_tracks   [3,2880,64,2] fp16          every z decoded at the 40x72 latent-grid cell centers (crop coords)
    tracks, vis, conf                        raw CoTracker3 tracks in crop coords [1024,64,2] + probs [1024,64]
    box, hw, fps, roundtrip_err              crop box, original size, fps, z round-trip error (sanity)

The z are decoded with the ORIGINAL first frame (the one they were encoded with) at query points given in
original coords; the result is mapped to crop coords. Resumable: existing outputs are skipped.

    python src/prep_koala.py --n 8                    # first 8 windows (sanity)
    python src/prep_koala.py                          # all
    python src/prep_koala.py --shard 0 --n_shards 2   # split over GPUs
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

torch.cuda.init()  # before decord (avoids CUDA init segfault)

from common import (LAT_H, LAT_W, N_FRAMES, ROOT, T_STEPS, WanVAE, crop_box, crop_resize, crop_to_orig,  # noqa: E402
                    decode_queries, denormalize_z, grid_queries, load_planner, orig_to_crop, read_video)


def roundtrip_err(vae, z_raw, sf, tracks_orig, enc_idx, hw):
    """Decode z[k] at the start points of its own 40 encoder tracks; mean endpoint-free L2 error in pixels of
    the original frame, averaged over steps. Small (a few px) <=> z was denormalized exactly once."""
    errs = []
    for k in range(z_raw.shape[0]):
        gt = tracks_orig[enc_idx[k]]  # 40,64,2
        pred = decode_queries(vae, z_raw[k:k + 1], sf, gt[:, 0])[0].cpu()
        scale = torch.tensor([(hw[0] - 1) / 2, (hw[1] - 1) / 2])
        errs.append(((pred - gt) * scale).norm(dim=-1).mean().item())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/koala_train_v2")
    ap.add_argument("--out", default="data/clips_a14b")
    ap.add_argument("--n", type=int, default=None, help="only the first n windows")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(args.out, exist_ok=True)

    windows = [r["window"] for r in csv.DictReader(open(os.path.join(args.src, "train_windows.csv")))]
    windows = windows[:args.n] if args.n else windows
    windows = windows[args.shard::args.n_shards]
    print(f"{len(windows)} windows (shard {args.shard}/{args.n_shards})", flush=True)

    vae = WanVAE()
    zm = load_planner().vae
    q_crop = grid_queries(LAT_H, LAT_W)  # 2880 cell centers, crop coords
    t0, n_done, bad = time.time(), 0, []
    for i, win in enumerate(windows):
        out = os.path.join(args.out, win + ".pt")
        if os.path.exists(out):
            continue
        d = os.path.join(args.src, win)
        meta = json.load(open(os.path.join(d, "meta.json")))
        fr = read_video(os.path.join(d, "frames.mp4"))  # T,h,w,3
        if fr.shape[0] < N_FRAMES:
            print(f"skip {win}: only {fr.shape[0]} frames", flush=True)
            bad.append(win)
            continue
        fr = fr[:N_FRAMES]
        hw = tuple(fr.shape[1:3])
        box = crop_box(*hw)
        frc = crop_resize(fr, box)  # 65,320,576,3

        latents = vae.encode_frames(frc)  # 16,17,40,72
        cond_lat = vae.encode_i2v_condition(frc[0])

        z_raw = denormalize_z(torch.from_numpy(np.load(os.path.join(d, "z.npy"))))  # 3,256,16
        K = z_raw.shape[0]
        sf = (fr[0].float() / 127.5 - 1)[None].expand(K, -1, -1, -1)  # original first frame
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dec = decode_queries(zm, z_raw, sf, crop_to_orig(q_crop, box, hw)).cpu()  # K,2880,64,2 orig
            tr = torch.from_numpy(np.load(os.path.join(d, "tracks_yx.npy")).astype(np.float32)).transpose(0, 1)
            enc_idx = torch.from_numpy(np.load(os.path.join(d, "enc_idx.npy")).astype(np.int64))
            rt = roundtrip_err(zm, z_raw, sf[:1], tr, enc_idx, hw)
        vis = torch.from_numpy(np.load(os.path.join(d, "visibility.npy")).astype(np.float32)).T
        conf = torch.from_numpy(np.load(os.path.join(d, "confidence.npy")).astype(np.float32)).T

        torch.save({
            "id": win, "fps": meta["fps"], "box": box, "hw": hw, "n_frames": N_FRAMES,
            "first_frame": frc[0].clone(),
            "latents": latents.bfloat16().cpu(), "cond_lat": cond_lat.bfloat16().cpu(),
            "dec_tracks": orig_to_crop(dec, box, hw).half(),
            "tracks": orig_to_crop(tr, box, hw).half(), "vis": vis.half(), "conf": conf.half(),
            "enc_idx": enc_idx, "roundtrip_err": rt,
        }, out)
        n_done += 1
        el = time.time() - t0
        print(f"[{i + 1}/{len(windows)}] {win} roundtrip {rt:.2f}px | {el / n_done:.1f}s/clip | "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    print(f"DONE {n_done} new, {len(bad)} skipped", flush=True)


if __name__ == "__main__":
    main()
