"""ZipMo official open-domain benchmark (68 Pexels clips, 40 GT tracks each) for the A14B models.

Ported from the teammate's 5B `bench_pexels.py` (pexels_eval_set.zip); same protocol so the numbers are comparable:
  - pokes = the first n of the 40 GT tracks (start -> end), n in --n_pokes
  - K videos per (clip, n): planner -> K ZipMo samples (noise seed 43 like the official sampler); line / oracle /
    none -> the same condition rendered with K render seeds
  - CoTracker3 (offline predictor, = torch.hub cotracker3_offline) tracks the 40 GT query points in every video;
    tracks are mapped back to the original 1280x720 frame; GT is interpolated in time to the video length
    (official rule; 65-frame videos: frame j = step j, the extra last frame is dropped)
  - metrics: the official compute_metrics_atm (Min / Mean / MeanT MSE on [0,128]) + EPE over the poked points
  - "zipmo_only": the planner's own tracks at the 40 GT queries (no video model); "static": every point stays
    at its start (floor reference)
Conditions: planner (main method), oracle (z encoded from the 40 GT tracks = Eval 1), line (Baseline 1),
none (Baseline 0 / vanilla). Data: data/raw/pexels_official (gt_tracks.pt + pexels/original-<name>.mp4).

    python src/bench_pexels.py prep
    python src/bench_pexels.py render --ckpt <high>.pt <low>.pt --cond planner --out outputs/bench/<run>
    python src/bench_pexels.py eval outputs/bench/<run>/planner/p1 outputs/bench/<run>/planner/p4 \
        --out results/bench/<run>__planner.json
    python src/bench_pexels.py static --out results/bench/static.json
"""

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch

from common import H, LAT_H, LAT_W, ROOT, T_STEPS, W, crop_box, crop_to_orig, decode_queries, grid_queries, \
    orig_to_crop, save_video

RAW = "data/raw/pexels_official"
BENCH = "data/bench_pexels"
OFFICIAL_SEED = 43
ZIPMO_REPO = os.environ.get("ZIPMO_REPO", "/project/YuHsuanLee/long-term-motion")  # git clone https://github.com/CompVis/long-term-motion (official eval script)


def make_poke(start, end):
    """[n,2] start / end (normalized yx) -> [n,5] pokes (y0, x0, y1, x1, 1)."""
    return torch.cat([start, end, torch.ones(len(start), 1)], -1)


# ---------------------------------------------------------------- prep
def cmd_prep(args):
    import cv2
    import decord

    gt = torch.load(os.path.join(RAW, "gt_tracks.pt"), weights_only=True, map_location="cpu")
    os.makedirs(BENCH, exist_ok=True)
    for name, d in gt.items():
        vr = decord.VideoReader(os.path.join(RAW, "pexels", f"original-{name}.mp4"))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # 64,h,w,3
        h, w = frames.shape[1:3]
        box = crop_box(h, w)
        y0, x0, th, tw = box
        crop = np.stack([cv2.resize(f[y0:y0 + th, x0:x0 + tw], (W, H), interpolation=cv2.INTER_AREA) for f in frames])
        tracks = d["tracks"].float()  # 40,64,2 original frame
        torch.save({"id": name, "first_frame": torch.from_numpy(crop[0]), "frames": torch.from_numpy(crop),
                    "fps": float(vr.get_avg_fps()), "gt": tracks, "gt_crop": orig_to_crop(tracks, box, (h, w)),
                    "box": box, "hw": (h, w)}, os.path.join(BENCH, name + ".pt"))
    print(f"{len(gt)} clips -> {BENCH}")


# ---------------------------------------------------------------- render
def cmd_render(args):
    from infer import Generator
    from motion_cond import line_tracks
    from common import load_planner

    G = Generator(args.ckpt)
    dev = G.dev
    q = grid_queries(LAT_H, LAT_W)
    planner = load_planner(dev) if args.cond in ("planner", "oracle") else None
    files = sorted(glob.glob(os.path.join(BENCH, "*.pt")))[:args.n]
    pokes_dep = args.cond in ("planner", "line")
    for f in files:
        c = torch.load(f, map_location="cpu")
        first, g = c["first_frame"], c["gt_crop"]
        sf = (first.float() / 127.5 - 1)[None].to(dev).bfloat16()
        for n in (args.n_pokes if pokes_dep else [0]):
            out = os.path.join(args.out, args.cond, f"p{n}" if pokes_dep else "all", c["id"])
            if os.path.exists(os.path.join(out, f"k{args.K - 1}", "video.mp4")):
                continue
            pokes = make_poke(g[:max(n, 1), 0], g[:max(n, 1), -1])
            extra, zs = {}, [None] * args.K
            if args.cond == "planner":
                gen = torch.Generator(device=dev).manual_seed(OFFICIAL_SEED)
                z = torch.randn(args.K, 256, 16, device=dev, dtype=torch.bfloat16, generator=gen)
                sfk = sf.expand(args.K, -1, -1, -1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lat = planner.sample(z, points_per_traj=T_STEPS, query_pos=None,
                                         track_conds=pokes.to(dev).bfloat16()[None].expand(args.K, -1, -1),
                                         start_frame=sfk, sample_steps=50, decode_latent=False)
                    zs = lat.float().cpu()  # normalized z, what a --mode latent model reads
                    lat = planner.denormalize_latents(lat)
                    dec = decode_queries(planner.vae, lat, sfk, q.to(dev)).float().cpu()  # K,2880,64,2
                    at_gt = decode_queries(planner.vae, lat, sfk, g[:, 0].to(dev)).float().cpu()  # K,40,64,2
                extra["zipmo_tracks"] = crop_to_orig(at_gt, c["box"], c["hw"])
                conds = [dec[k] for k in range(args.K)]
            elif args.cond == "line":
                conds = [line_tracks(pokes, q)] * args.K
            elif args.cond == "oracle":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    zr = planner.vae.encode(g[None].to(dev).bfloat16(), start_frame=sf).mean
                    dec = decode_queries(planner.vae, zr, sf, q.to(dev))[0].float().cpu()
                    zs = planner.normalize_latents(zr).float().cpu().expand(args.K, -1, -1)
                conds = [dec] * args.K
            else:
                conds = [None] * args.K
            for k, tracks in enumerate(conds):
                frames = G.generate(first, tracks, args.steps, args.shift, args.motion_cfg, args.text_cfg, seed=k,
                                    z=None if zs[k] is None else zs[k])
                od = os.path.join(out, f"k{k}")
                os.makedirs(od, exist_ok=True)
                save_video(os.path.join(od, "video.mp4"), frames, fps=round(c["fps"]))
                torch.save({"pokes": pokes, "tracks": None if tracks is None else tracks.half(),
                            "z": None if zs[k] is None else zs[k],
                            "args": {**vars(args), "clip": f, "k": k}}, os.path.join(od, "cond.pt"))
            if extra:
                torch.save(extra, os.path.join(out, "extra.pt"))
            print(f"{c['id']} [{args.cond} n={n}] K={args.K} | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)


# ---------------------------------------------------------------- eval
_TRACKER = None


def tracker(device="cuda"):
    """CoTracker3 offline predictor, identical to torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')."""
    global _TRACKER
    if _TRACKER is None:
        import tracking  # noqa: F401  (puts third_party/co-tracker on sys.path)
        from cotracker.predictor import CoTrackerPredictor

        from tracking import COTRACKER3_OFFLINE_URL

        p = CoTrackerPredictor(checkpoint=None, offline=True, window_len=60)
        p.model.load_state_dict(torch.hub.load_state_dict_from_url(COTRACKER3_OFFLINE_URL, map_location="cpu"))
        _TRACKER = p.to(device).eval()
    return _TRACKER


@torch.no_grad()
def track_points(frames_uint8, query_yx, device="cuda"):
    """frames [F,H,W,3] uint8; query_yx [N,2] normalized at frame 0 -> [N,F,2] normalized yx (teammate's eval.py)."""
    model = tracker(device)
    F, h, w, _ = frames_uint8.shape
    v = torch.from_numpy(frames_uint8).to(device).permute(0, 3, 1, 2).float()[None]
    qx = (query_yx[:, 1] + 1) / 2 * (w - 1)
    qy = (query_yx[:, 0] + 1) / 2 * (h - 1)
    qq = torch.stack([torch.zeros_like(qx), qx, qy], dim=-1)[None].to(device).float()
    with torch.autocast("cuda", dtype=torch.float16):
        tr, vis = model(v, queries=qq)
    xy = tr[0].float().permute(1, 0, 2)  # N,F,2
    return torch.stack([xy[..., 1] / (h - 1) * 2 - 1, xy[..., 0] / (w - 1) * 2 - 1], -1).cpu()


def official_metrics():
    import sys
    import types

    sys.modules.setdefault("fire", types.ModuleType("fire"))  # only used by the script's CLI entry point
    path = os.path.join(ZIPMO_REPO, "scripts/eval.py")
    spec = importlib.util.spec_from_file_location("zipmo_official_eval", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.compute_metrics_atm


def score(gen, gt, n_pokes, atm):
    """gen [K,40,T',2], gt [40,64,2] (original frame, [-1,1]) -> official metrics (+ EPE over the poked points)."""
    if gen.shape[2] < gt.shape[1]:  # official: interpolate GT down to the generated length
        gt = torch.nn.functional.interpolate(gt.permute(0, 2, 1), size=gen.shape[2], mode="linear").permute(0, 2, 1)
    elif gen.shape[2] > gt.shape[1]:
        gen = gen[:, :, :gt.shape[1]]  # 65-frame videos: frame j = step j, drop the extra last frame
    r = atm([gen], [gt], return_per_batch=True)
    out = {k: r[k][0] for k in ("Min_MSE", "Mean_MSE", "MeanT_MSE")}
    rn = atm([gen], [gt], normalize_by_track_magnitude=True, return_per_batch=True)
    out.update({f"{k}_norm": rn[k][0] for k in ("Min_MSE", "Mean_MSE")})
    if n_pokes > 0:
        g, s = gen.mul(0.5).add(0.5).mul(128), gt.mul(0.5).add(0.5).mul(128)
        out["EPE"] = float((g[:, :n_pokes, -1] - s[None, :n_pokes, -1]).norm(dim=-1).mean())
    return out


def summarize(rows):
    return {k: {m: float(np.mean([r[m] for r in v.values() if m in r])) for m in next(iter(v.values()))} | {"n": len(v)}
            for k, v in rows.items()}


def cmd_eval(args):
    import imageio.v3 as iio

    atm = official_metrics()
    rows = {}
    for root in args.dirs:
        tag = os.path.basename(root.rstrip("/"))  # p1 / p4 / all
        cond = os.path.basename(os.path.dirname(root.rstrip("/")))  # planner / line / oracle / none
        for clip_dir in sorted(glob.glob(os.path.join(root, "*"))):
            ks = sorted(glob.glob(os.path.join(clip_dir, "k*", "video.mp4")))
            if not os.path.isdir(clip_dir) or not ks:
                continue
            c = torch.load(os.path.join(BENCH, os.path.basename(clip_dir) + ".pt"), map_location="cpu")
            cache = os.path.join(clip_dir, "tracked.pt")
            if os.path.exists(cache):
                gen = torch.load(cache, weights_only=True)
            else:
                gen = torch.stack([crop_to_orig(track_points(iio.imread(v), c["gt_crop"][:, 0]), c["box"], c["hw"])
                                   for v in ks])  # K,40,F,2
                torch.save(gen, cache)
            for n in ([int(tag[1:])] if tag.startswith("p") else args.n_pokes):
                key = f"{cond}_p{n}"
                rows.setdefault(key, {})[c["id"]] = score(gen, c["gt"], n, atm)
                ex = os.path.join(clip_dir, "extra.pt")
                if os.path.exists(ex):
                    zt = torch.load(ex, weights_only=True)["zipmo_tracks"]
                    rows.setdefault(f"zipmo_only_p{n}", {})[c["id"]] = score(zt, c["gt"], n, atm)
            print(clip_dir, {k: round(v, 2) for k, v in rows[key][c["id"]].items()}, flush=True)
    summary = summarize(rows)
    for k, v in summary.items():
        print(k, {m: round(x, 3) for m, x in v.items()})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_clip": rows}, open(args.out, "w"), indent=1)


def cmd_static(args):
    """Floor reference: every GT point stays at its start for all 64 steps."""
    atm = official_metrics()
    rows = {}
    for f in sorted(glob.glob(os.path.join(BENCH, "*.pt"))):
        c = torch.load(f, map_location="cpu")
        gen = c["gt"][None, :, :1].expand(1, -1, T_STEPS, -1).contiguous()
        for n in args.n_pokes:
            rows.setdefault(f"static_p{n}", {})[c["id"]] = score(gen, c["gt"], n, atm)
    summary = summarize(rows)
    for k, v in summary.items():
        print(k, {m: round(x, 3) for m, x in v.items()})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_clip": rows}, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    r = sub.add_parser("render")
    r.add_argument("--ckpt", nargs="*", default=[], help="0-2 checkpoints, one per expert")
    r.add_argument("--cond", choices=["planner", "line", "oracle", "none"], required=True)
    r.add_argument("--n_pokes", type=int, nargs="+", default=[1, 4])
    r.add_argument("--K", type=int, default=1)
    r.add_argument("--n", type=int, default=999)
    r.add_argument("--steps", type=int, default=30)
    r.add_argument("--shift", type=float, default=3.0)
    r.add_argument("--motion_cfg", type=float, default=3.0)
    r.add_argument("--text_cfg", type=float, default=3.5)
    r.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("dirs", nargs="+", help="<out>/<cond>/p1 ... or <out>/<cond>/all")
    e.add_argument("--n_pokes", type=int, nargs="+", default=[1, 4], help="EPE settings for poke-free conds")
    e.add_argument("--out", required=True)
    s = sub.add_parser("static")
    s.add_argument("--n_pokes", type=int, nargs="+", default=[1, 4])
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    {"prep": cmd_prep, "render": cmd_render, "eval": cmd_eval, "static": cmd_static}[args.cmd](args)


if __name__ == "__main__":
    main()
