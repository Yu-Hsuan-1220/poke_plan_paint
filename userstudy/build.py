"""Build the media + private manifest for the user study from the Pexels benchmark renders.

    /home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python userstudy/build.py            # main, b0, b1, zlat (whatever exists)
    ... userstudy/build.py --methods main b0 b1 zlat vanilla                            # add vanilla

Writes
  userstudy/site/media/<hash>.jpg|.mp4   public; names are salted hashes so the URL never reveals the method
  userstudy/data/manifest.json           private (method <-> file mapping), read by server.py
Rerun after zlat is rendered: existing files are kept, new methods are added, the salt is reused, and the
running server picks the new manifest up by itself.

Only the 68 Pexels clips (Pexels license) and videos generated from them go into site/ -- never OpenVid / Koala.
"""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "data/bench_pexels")
SITE = os.path.join(ROOT, "userstudy/site")
MEDIA = os.path.join(SITE, "media")
MANIFEST = os.path.join(ROOT, "userstudy/data/manifest.json")

# method key -> (label, render dir; {n} = number of pokes)
METHODS = {
    "main": ("Main: hybrid + ZipMo planner", "outputs/bench/main/planner/p{n}"),
    "b0": ("B0: fine-tune, no motion", "outputs/bench/b0/none/all"),
    "b1": ("B1: raw poke as straight line", "outputs/bench/b1/line/p{n}"),
    "zlat": ("zlat: z as cross-attn tokens, planner", "outputs/bench/zlat/planner/p{n}"),
    "vanilla": ("Vanilla Wan2.2 (no fine-tune)", "outputs/bench/vanilla/none/all"),
}


def h(salt, *parts):
    return hashlib.sha256("|".join((salt,) + parts).encode()).hexdigest()[:20]


def faststart_copy(src, dst):
    """Remux (no re-encode) with the moov atom up front and metadata stripped."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-map", "0:v:0", "-c", "copy",
                    "-map_metadata", "-1", "-movflags", "+faststart", dst + ".tmp.mp4"], check=True)
    os.replace(dst + ".tmp.mp4", dst)


def write_video(frames, fps, dst):
    import imageio

    imageio.mimwrite(dst + ".tmp.mp4", list(frames), fps=fps, quality=8, macro_block_size=1,
                     ffmpeg_params=["-movflags", "+faststart"])
    os.replace(dst + ".tmp.mp4", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["main", "b0", "b1", "zlat"])
    ap.add_argument("--ref", default="main", help="method that appears in every trial")
    ap.add_argument("--n_pokes", type=int, default=4)
    ap.add_argument("--n_catch", type=int, default=8, help="clips with GT forward/reversed attention checks")
    args = ap.parse_args()
    assert args.ref in args.methods

    os.makedirs(MEDIA, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    old = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    salt = old.get("salt") or secrets.token_hex(16)

    clips, motion = {}, {}
    files = sorted(f for f in os.listdir(BENCH) if f.endswith(".pt"))
    for f in files:
        cid = f[:-3]
        c = torch.load(os.path.join(BENCH, f), map_location="cpu", weights_only=False)
        g = c["gt_crop"].float()  # 40,64,2 normalized yx in the 320x576 crop
        pokes = torch.cat([g[:args.n_pokes, 0], g[:args.n_pokes, -1]], -1)  # n,4 (y0,x0,y1,x1) = bench_pexels
        motion[cid] = float((g[:args.n_pokes, -1] - g[:args.n_pokes, 0]).norm(dim=-1).mean())

        img = os.path.join(MEDIA, h(salt, "img", cid) + ".jpg")
        if not os.path.exists(img):
            from PIL import Image
            Image.fromarray(c["first_frame"].numpy()).save(img, quality=92)

        videos = {}
        for m in args.methods:
            src = os.path.join(ROOT, METHODS[m][1].format(n=args.n_pokes), cid, "k0", "video.mp4")
            if not os.path.exists(src):
                continue
            dst = os.path.join(MEDIA, h(salt, "vid", m, cid) + ".mp4")
            if not os.path.exists(dst):
                faststart_copy(src, dst)
            videos[m] = os.path.relpath(dst, SITE)
        clips[cid] = {"image": os.path.relpath(img, SITE), "pokes": [[round(v, 5) for v in p] for p in pokes.tolist()],
                      "fps": round(c["fps"]), "videos": videos, "_frames": c["frames"]}

    # attention checks: GT played forward (follows the arrows) vs reversed, on the clips with the most motion
    for cid in sorted(motion, key=motion.get, reverse=True)[:args.n_catch]:
        fr = clips[cid]["_frames"].numpy()
        fwd, rev = (os.path.join(MEDIA, h(salt, "catch", s, cid) + ".mp4") for s in ("fwd", "rev"))
        if not os.path.exists(fwd):
            write_video(fr, clips[cid]["fps"], fwd)
        if not os.path.exists(rev):
            write_video(fr[::-1], clips[cid]["fps"], rev)
        clips[cid]["catch"] = {"fwd": os.path.relpath(fwd, SITE), "rev": os.path.relpath(rev, SITE)}
    for c in clips.values():
        del c["_frames"]

    count = {m: sum(m in c["videos"] for c in clips.values()) for m in args.methods}
    present = [m for m in args.methods if count[m]]
    opponents = [m for m in present if m != args.ref]
    man = {"salt": salt, "ref": args.ref, "n_pokes": args.n_pokes, "opponents": opponents,
           "labels": {m: METHODS[m][0] for m in present}, "clips": clips}
    json.dump(man, open(MANIFEST + ".tmp", "w"), indent=1)
    os.replace(MANIFEST + ".tmp", MANIFEST)
    print(f"{len(clips)} clips; videos per method: {count}; opponents: {opponents}; "
          f"catch clips: {sum('catch' in c for c in clips.values())}")
    missing = [m for m in args.methods if not count[m]]
    if missing:
        print(f"not rendered yet (skipped): {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
