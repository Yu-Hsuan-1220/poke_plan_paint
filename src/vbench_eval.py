"""Eval 3 (video quality) on the Pexels benchmark videos with VBench / VBench-I2V per-video metrics.

FVD on 68 videos is unstable, so we report VBench dimensions, which are scored per video (mean over videos, with a
paired bootstrap across the same 68 clips when comparing methods):
    subject_consistency, background_consistency   DINO / CLIP similarity across frames (identity kept)
    temporal_flickering, motion_smoothness         frame-level flicker; AMT interpolation error (jerky motion)
    dynamic_degree                                  RAFT: does the video move at all (guards against static videos)
    aesthetic_quality, imaging_quality              LAION aesthetic head; MUSIQ (blur, noise, artifacts)
    i2v_subject, i2v_background                     VBench-I2V: frames vs the input first frame (DINO; DreamSim)
The real Pexels clips (64 frames, same 320x576 crop) are scored too, as the reference row ("gt").

VBench lives in envs/vbench_pkgs (pip --target, so nothing leaks into the p2v env); run with
    PYTHONPATH=envs/vbench_pkgs python src/vbench_eval.py outputs/bench/main/planner/p1 --tag main_planner_p1
    PYTHONPATH=envs/vbench_pkgs python src/vbench_eval.py gt --tag gt
-> results/vbench/<tag>.json  {"summary": {dim: mean}, "per_video": {dim: {clip: score}}}
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from common import ROOT, save_video

BENCH = "data/bench_pexels"
VB_DIMS = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness",
           "dynamic_degree", "aesthetic_quality", "imaging_quality"]
I2V_DIMS = ["i2v_subject", "i2v_background"]


def collect(src, work, n=None):
    """-> {clip: video path}, {clip: first-frame png}. src = <out>/<cond>/<p1|all> (uses k0) or 'gt'."""
    from PIL import Image

    vids, imgs = {}, {}
    os.makedirs(os.path.join(work, "videos"), exist_ok=True)
    os.makedirs(os.path.join(work, "images"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(BENCH, "*.pt")))[:n]:
        cid = os.path.basename(f)[:-3]
        v = os.path.join(src, cid, "k0", "video.mp4") if src != "gt" else None
        if src != "gt" and not os.path.exists(v):
            continue
        c = torch.load(f, map_location="cpu")
        img = os.path.join(work, "images", cid + ".png")
        if not os.path.exists(img):
            Image.fromarray(c["first_frame"].numpy()).save(img)
        dst = os.path.join(work, "videos", cid + ".mp4")
        if not os.path.exists(dst):
            if src == "gt":
                save_video(dst, c["frames"], fps=round(c["fps"]))
            else:
                os.symlink(os.path.abspath(v), dst)
        vids[cid], imgs[cid] = dst, img
    return vids, imgs


def run_vbench(vid_dir, work, dims, device):
    # VBench's own checkpoints (e.g. AMT-S for motion_smoothness) are full pickles; torch>=2.6 defaults to
    # weights_only=True and refuses them. They are VBench's official downloads into ~/.cache/vbench.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    # VBench calls clip.load("ViT-B/32"), which always downloads into ~/.cache/clip; send it to the VBench cache
    # (on /project when run remotely, see scripts/remote_run.sh) instead.
    import clip

    _load = clip.load
    clip_root = os.path.join(os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench")), "clip_model")
    clip.load = lambda name, *a, download_root=None, **k: _load(name, *a, download_root=download_root or clip_root, **k)
    import vbench
    from vbench import VBench

    vb = VBench(device, os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json"), work)
    out = {}
    for d in dims:
        vb.evaluate(vid_dir, f"vb_{d}", dimension_list=[d], mode="custom_input")  # writes, returns None
        r = json.load(open(os.path.join(work, f"vb_{d}_eval_results.json")))[d]
        out[d] = {os.path.basename(x["video_path"])[:-4]: float(x["video_results"]) for x in r[1]}
    return out


def run_i2v(vids, imgs, dims, device):
    pairs = [(imgs[c], vids[c]) for c in sorted(vids)]
    out = {}
    if "i2v_subject" in dims:
        from vbench2_beta_i2v.i2v_subject import i2v_subject

        dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()
        _, r = i2v_subject(dino, pairs, device)
        out["i2v_subject"] = {os.path.basename(x["video_path"])[:-4]: float(x["video_results"]) for x in r}
    if "i2v_background" in dims:
        from dreamsim import dreamsim
        from vbench2_beta_i2v.i2v_background import i2v_background

        model, _ = dreamsim(pretrained=True, device=device, cache_dir=os.path.join(os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench")), "dreamsim"))
        _, r = i2v_background(model, pairs, device)
        out["i2v_background"] = {os.path.basename(x["video_path"])[:-4]: float(x["video_results"]) for x in r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="<out>/<cond>/<p1|p4|all> or 'gt'")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", nargs="+", default=VB_DIMS + I2V_DIMS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=None, help="only the first n clips (smoke test)")
    args = ap.parse_args()
    os.chdir(ROOT)
    work = os.path.join("outputs", "vbench", args.tag)
    vids, imgs = collect(args.src, work, args.n)
    print(f"{args.tag}: {len(vids)} videos", flush=True)
    per = run_vbench(os.path.join(work, "videos"), work, [d for d in args.dims if d in VB_DIMS], args.device)
    per |= run_i2v(vids, imgs, [d for d in args.dims if d in I2V_DIMS], args.device)
    summary = {d: float(np.mean(list(v.values()))) for d, v in per.items()} | {"n": len(vids)}
    print(json.dumps(summary, indent=1))
    os.makedirs("results/vbench", exist_ok=True)
    json.dump({"summary": summary, "per_video": per}, open(f"results/vbench/{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
