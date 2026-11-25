"""Post-hoc validation loss of the training checkpoints on held-out OpenVid clips (data/clips_a14b_openvid_test).

    CUDA_VISIBLE_DEVICES=2 python src/val_loss.py --expert high      -> results/val/val_high.jsonl
    CUDA_VISIBLE_DEVICES=4 python src/val_loss.py --expert low       -> results/val/val_low.jsonl
    python3 src/plot_loss.py                                          -> results/figures/val_loss.png

Same flow-matching loss as train.py, but deterministic: the first --n_clips test clips x fixed sigmas x fixed
noise (seeded per clip / sigma), so every checkpoint is scored on exactly the same items (paired across steps
and runs). The condition is fixed too: main = dec_tracks[0], B1 = line from the most-moving track (eval rule),
zlat = z[0], B0 = none. --nocond_steps also scores the conditional runs with the condition dropped (as in
train.py's cond dropout), i.e. how much the model relies on the motion. step 0 = base Wan2.2, no fine-tune.
One 14B load per process; LoRA / adapter weights are swapped per checkpoint. Rerunning skips rows already written.
"""

import argparse
import glob
import json
import os
import time

import torch
from peft import set_peft_model_state_dict

from common import ROOT, i2v_mask
from motion_cond import MotionAdapter, build_condition, clip_poke, line_tracks
from train import GRID, load_transformer

ARCHIVE = "/project2/YuHsuanLee/poke_to_video_ckpt_archive"
RUNS = {"main": ("hybrid", "hybrid"), "B0": ("b0", "none"), "B1": ("b1", "hybrid"), "zlat": ("zlat", "latent")}
SIGMAS = {"high": [0.925, 0.975], "low": [0.3, 0.7]}
STEP_ORDER = [4000, 2000, 1000, 3000, 500, 1500, 2500, 3500]  # coarse curve first


def ckpt_path(run, step):
    for d in (f"outputs/train/{run}", f"{ARCHIVE}/{run}"):
        p = f"{d}/ckpt_{step:05d}.pt"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{run} step {step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=["high", "low"], required=True)
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--clips", default="data/clips_a14b_openvid_test")
    ap.add_argument("--n_clips", type=int, default=20)
    ap.add_argument("--steps", type=int, nargs="+", default=STEP_ORDER)
    ap.add_argument("--nocond_steps", type=int, nargs="+", default=[4000, 2000, 1000, 3000])
    ap.add_argument("--reserve_gb", type=float, default=36)
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("results/val", exist_ok=True)
    out = f"results/val/val_{args.expert}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["run"], r["step"], r["cond"]))

    dev = "cuda"
    torch.empty(int(args.reserve_gb * 2**30), dtype=torch.uint8, device=dev)  # hold the GPU during the slow load
    files = sorted(glob.glob(os.path.join(args.clips, "*.pt")))[:args.n_clips]
    zbank = torch.load(args.clips.rstrip("/") + "_z.pt")
    clips = []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        noise = [torch.randn(d["latents"].shape, generator=torch.Generator().manual_seed(1000 * i + j))
                 for j in range(len(SIGMAS[args.expert]))]
        poke = clip_poke(d["tracks"], d["vis"], d["conf"])
        clips.append({"id": d["id"], "x0": d["latents"][None].float(), "cond_lat": d["cond_lat"][None].float(),
                      "cond": {"hybrid": d["dec_tracks"][0][None].float(), "line": line_tracks(poke, GRID)[None],
                               "latent": zbank[d["id"]][0][None].float()}, "noise": noise})
    print(f"{len(clips)} val clips x {len(SIGMAS[args.expert])} sigmas | expert {args.expert}", flush=True)

    tr = load_transformer(args.expert, 32, dev)  # LoRA B = 0 at init -> identical to the base model until loaded
    tr.eval()
    ads = {m: MotionAdapter(m).to(dev).attach(tr) for m in ("hybrid", "latent")}
    text = torch.load("data/prompt_embeds.pt")["embeds"].to(dev)
    mask = i2v_mask(device=dev)[None]

    @torch.no_grad()
    def score(mode, cond_src):
        """mode in (none, hybrid, latent); cond_src = key into clip['cond'] or None (condition dropped)."""
        losses = []
        for c in clips:
            x0, cl = c["x0"].to(dev), c["cond_lat"].to(dev)
            tracks = c["cond"][cond_src].to(dev) if cond_src else None
            y, ad_in = build_condition(mode if mode != "none" else "y", cl, tracks)
            ad = ads.get(mode)
            if ad is not None:
                ad.set(ad_in)
            for s, n in zip(SIGMAS[args.expert], c["noise"]):
                n = n.to(dev)
                xt = (1 - s) * x0 + s * n
                inp = torch.cat([xt, mask, y.float()], dim=1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = tr(hidden_states=inp.bfloat16(), timestep=torch.tensor([s * 1000.0], device=dev),
                              encoder_hidden_states=text, return_dict=False)[0]
                losses.append(((pred.float() - (n - x0)) ** 2).mean().item())
            if ad is not None:
                ad.set(None)
        return losses

    def write(run, step, cond, losses, t0):
        r = {"run": run, "step": step, "cond": cond, "mean": sum(losses) / len(losses), "losses": losses,
             "sigmas": SIGMAS[args.expert], "clips": [c["id"] for c in clips]}
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        done.add((run, step, cond))
        print(f"[{time.strftime('%H:%M')}] {run:5s} step {step:4d} cond={int(cond)} loss {r['mean']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    if ("base", 0, False) not in done:
        t0 = time.time()
        write("base", 0, False, score("none", None), t0)

    for step in args.steps:
        for label in args.runs:
            key, mode = RUNS[label]
            todo = [c for c in ([True, False] if mode != "none" and step in args.nocond_steps else [mode != "none"])
                    if (label, step, c) not in done]
            if not todo:
                continue
            ck = torch.load(ckpt_path(f"openvid_{key}_{args.expert}", step), map_location="cpu")
            assert ck["step"] == step and ck["expert"] == args.expert, (ck["step"], ck["expert"])
            res = set_peft_model_state_dict(tr, ck["lora"])
            assert not res.unexpected_keys and not [k for k in res.missing_keys if "lora" in k], res
            if mode in ads:
                ads[mode].load_state_dict(ck["adapter"])
            src = {"hybrid": "line" if ck.get("tracks") == "line" else "hybrid", "latent": "latent"}.get(mode)
            for cond in todo:
                t0 = time.time()
                write(label, step, cond, score(mode, src if cond else None), t0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
