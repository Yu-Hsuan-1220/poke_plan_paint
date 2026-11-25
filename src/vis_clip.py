"""Visual sanity check of preprocessed clips.

For each clip writes to <out>/<id>/:
    tracks.jpg     first frame + ZipMo-decoded dense tracks of z[k] (every 4th cell), next to raw CoTracker tracks
    gt.mp4         the cropped 65-frame clip (decoded from the target latents)
    splat.mp4      VAE decode of the "y" condition latent (first frame carried along the decoded tracks)
If preprocessing and the splat are right, splat.mp4 moves like gt.mp4.

    python src/vis_clip.py --n 4
"""

import argparse
import glob
import os

import numpy as np
import torch

from common import ROOT, WanVAE, draw_tracks, save_video
from motion_cond import build_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips_a14b")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=0, help="which z")
    ap.add_argument("--out", default="outputs/vis_clips")
    args = ap.parse_args()
    os.chdir(ROOT)
    import cv2

    vae = WanVAE()
    for f in sorted(glob.glob(os.path.join(args.clips, "*.pt")))[:args.n]:
        c = torch.load(f, map_location="cpu")
        od = os.path.join(args.out, c["id"])
        os.makedirs(od, exist_ok=True)
        img = c["first_frame"].numpy()
        dense = c["dec_tracks"][args.k].float()  # 2880,64,2
        raw = c["tracks"].float()
        ok = (c["vis"][:, 0] > 0.5) & (c["conf"] > 0.5).all(1)
        cv2.imwrite(os.path.join(od, "tracks.jpg"), cv2.cvtColor(np.concatenate(
            [draw_tracks(img, dense.view(40, 72, 64, 2)[::4, ::4].reshape(-1, 64, 2)),
             np.full((img.shape[0], 6, 3), 255, np.uint8),
             draw_tracks(img, raw[ok][::4])], 1), cv2.COLOR_RGB2BGR))
        save_video(os.path.join(od, "gt.mp4"), vae.decode(c["latents"][None].float()))
        y, _ = build_condition("y", c["cond_lat"][None].float().cuda(), dense[None].cuda())
        save_video(os.path.join(od, "splat.mp4"), vae.decode(y))
        mov = (dense[:, -1] - dense[:, 0]).norm(dim=-1)
        print(f"{c['id']}: fps {c['fps']:.1f}, roundtrip {c['roundtrip_err']:.2f}px, "
              f"dense tracks moving >2% frame: {(mov > 0.04).float().mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
