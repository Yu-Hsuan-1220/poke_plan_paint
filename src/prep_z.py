"""ZipMo z for the preprocessed clips (needed by --mode latent): <clips>_z.pt = {clip id: z [3,256,16] fp32}.

z = windows/<id>/z.npy, i.e. planner.normalize_latents(vae.encode(...).mean) -- the same normalized space the
planner samples in (and the same 3 z that prep_koala.py decoded into dec_tracks, in the same order). A side file,
so the clip .pt files that other runs are reading are not rewritten.

    python src/prep_z.py --clips data/clips_a14b_openvid data/clips_a14b_openvid_test --src data/openvid_clean_set/windows
"""

import argparse
import glob
import os

import numpy as np
import torch

from common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--src", required=True, help="window dirs with z.npy")
    args = ap.parse_args()
    os.chdir(ROOT)
    for clips in args.clips:
        ids = [os.path.basename(f)[:-3] for f in sorted(glob.glob(os.path.join(clips, "*.pt")))]
        bank = {i: torch.from_numpy(np.load(os.path.join(args.src, i, "z.npy"))).float() for i in ids}
        z = torch.stack(list(bank.values()))
        out = clips.rstrip("/") + "_z.pt"
        torch.save(bank, out)
        print(f"{out}: {len(bank)} clips, z {tuple(z.shape[1:])}, mean {z.mean():.3f} std {z.std():.3f}")


if __name__ == "__main__":
    main()
