"""LoRA (+ motion adapter) fine-tuning of one Wan2.2-I2V-A14B expert on preprocessed Koala clips.

Teacher forcing: the condition is the clip's own motion (ZipMo VAE z_gt of its CoTracker tracks, decoded at the
40x72 latent grid); at inference the planner's z replaces it. One of the 3 z_gt is picked at random each step.

A14B has two experts with disjoint noise ranges, so each is trained on its own (only one 14B model in memory):
    --expert high : `transformer`,   sigma in [0.9, 1]   (default; decides the coarse motion)
    --expert low  : `transformer_2`, sigma in [0, 0.9]
Transformer input (36 ch) = [x_t (16), first-frame mask (4), condition latent y (16)]; see motion_cond.py.

    python src/train.py --run a14b_hybrid_high --mode hybrid --expert high
    python src/train.py --run overfit8 --max_clips 8 --steps 300        # sanity: overfit a few clips
    python src/train.py --run b0_high --mode none                       # Baseline 0: plain fine-tune, no motion
    python src/train.py --run b1_high --mode hybrid --tracks line       # Baseline 1: raw poke as a straight line
    python src/train.py --run zlat_high --mode latent                   # implicit: z itself as cross-attn tokens
--mode latent reads the z of each clip from <clips>_z.pt (src/prep_z.py), the same 3 z that dec_tracks decodes.
"""

import argparse
import errno
import glob
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model_state_dict

from common import BOUNDARY, LAT_H, LAT_W, ROOT, grid_queries, i2v_mask, wan_path
from motion_cond import MODES, MotionAdapter, build_condition, clip_poke, line_tracks

SUBFOLDER = {"high": "transformer", "low": "transformer_2"}


def load_transformer(expert, lora_rank, device="cuda"):
    from diffusers import WanTransformer3DModel

    tr = WanTransformer3DModel.from_pretrained(wan_path(), subfolder=SUBFOLDER[expert], torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    tr.to(device)
    if lora_rank > 0:
        tr.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights=True,
                                  target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    return tr


def sigma_range(expert):
    return (BOUNDARY, 1.0) if expert == "high" else (0.0, BOUNDARY)


GRID = grid_queries(LAT_H, LAT_W)


def load_batch(files, device, mode, cond_drop, track_src="zipmo", zbank=None):
    d = torch.load(random.choice(files), map_location="cpu")
    x0 = d["latents"][None].to(device).float()  # 1,16,17,40,72
    cond_lat = d["cond_lat"][None].to(device).float()
    tracks = None
    if mode != "none" and random.random() >= cond_drop:
        if mode == "latent":  # normalized ZipMo z [1,256,16] instead of tracks
            z = zbank[d["id"]]
            tracks = z[random.randrange(z.shape[0])][None].to(device).float()
        elif track_src == "line":  # Baseline 1: one poke from the clip's own motion, drawn as a straight line
            poke = clip_poke(d["tracks"], d["vis"], d["conf"], top_frac=0.05)
            tracks = line_tracks(poke, GRID)[None].to(device)
        else:
            tracks = d["dec_tracks"][random.randrange(d["dec_tracks"].shape[0])][None].to(device).float()
    y, ad_in = build_condition(mode if mode != "none" else "y", cond_lat, tracks)
    return d["id"], x0, y, ad_in, tracks is not None


def try_enospc(fn):
    """Run a write; on ENOSPC (the NAS quota fills up now and then, 2026-11-22) skip it and return False."""
    try:
        fn()
        return True
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        return False


def retry_enospc(fn, what, wait=120, hours=12):
    """Run a write that must happen; on ENOSPC wait and retry (training state stays on the GPU meanwhile)."""
    for _ in range(int(hours * 3600 / wait)):
        if try_enospc(fn):
            return
        try_enospc(lambda: print(f"ENOSPC writing {what}; retry in {wait}s", flush=True))
        time.sleep(wait)
    raise OSError(errno.ENOSPC, f"no space for {what} after {hours}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--clips", default="data/clips_a14b")
    ap.add_argument("--expert", choices=["high", "low"], default="high")
    ap.add_argument("--mode", choices=list(MODES) + ["none"], default="hybrid",
                    help="y = I1, add = I2, hybrid = I3, none = Baseline 0 (LoRA only)")
    ap.add_argument("--tracks", choices=["zipmo", "line"], default="zipmo",
                    help="zipmo = dense tracks decoded from z_gt (main); line = Baseline 1 (raw poke as a straight line)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adapter_lr", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--cond_drop", type=float, default=0.1)
    ap.add_argument("--max_clips", type=int, default=None, help="use only the first n clips (overfit test)")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reserve_gb", type=float, default=37, help="GPU memory held from the start (peak is ~35.5 GB)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <run>/ckpt_last.pt (LoRA + adapter; AdamW moments are not saved and restart)")
    args = ap.parse_args()

    os.chdir(ROOT)
    out_dir = os.path.join("outputs", "train", args.run)
    os.makedirs(out_dir, exist_ok=True)
    resume = torch.load(os.path.join(out_dir, "ckpt_last.pt"), map_location="cpu") if args.resume else None
    start = resume["step"] + 1 if resume else 1
    random.seed(args.seed + start - 1)  # a resumed run must not replay the clip / sigma sequence of steps 1..
    torch.manual_seed(args.seed + start - 1)
    files = sorted(glob.glob(os.path.join(args.clips, "*.pt")))
    files = files[:args.max_clips] if args.max_clips else files
    assert files, f"no clips in {args.clips}"
    args_path = os.path.join(out_dir, "args.json" if not resume else f"args_resume{start - 1:05d}.json")
    retry_enospc(lambda: json.dump({**vars(args), "train_files": files}, open(args_path, "w"), indent=1), args_path)
    print(f"{len(files)} clips | expert {args.expert} | mode {args.mode}", flush=True)

    dev = "cuda"
    if args.reserve_gb:  # the A14B load takes ~20 min; on a shared GPU another job can take the space meanwhile and
        # we OOM at step 1. Allocate the peak up front and free it: the caching allocator keeps it reserved for us.
        torch.empty(int(args.reserve_gb * 2**30), dtype=torch.uint8, device=dev)
    tr = load_transformer(args.expert, args.lora_rank, dev)
    tr.enable_gradient_checkpointing()
    ad = MotionAdapter(args.mode if args.mode != "none" else "y").to(dev).attach(tr)
    if resume:
        from peft import set_peft_model_state_dict

        assert resume["mode"] == args.mode and resume["expert"] == args.expert, "resume: mode / expert mismatch"
        res = set_peft_model_state_dict(tr, resume["lora"])
        assert not res.unexpected_keys and not [k for k in res.missing_keys if "lora" in k], res
        ad.load_state_dict(resume["adapter"])
        print(f"resumed from step {start - 1}", flush=True)
    zbank = torch.load(args.clips.rstrip("/") + "_z.pt") if args.mode == "latent" else None
    text = torch.load("data/prompt_embeds.pt")["embeds"].to(dev)  # 1,512,4096
    mask = i2v_mask(device=dev)[None]  # 1,4,17,40,72

    lora_params = [p for p in tr.parameters() if p.requires_grad]
    ad_params = list(ad.parameters())
    groups = [{"params": lora_params, "lr": args.lr}]
    if ad_params:
        groups.append({"params": ad_params, "lr": args.adapter_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    print(f"trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f}M, "
          f"adapter {sum(p.numel() for p in ad_params) / 1e6:.2f}M | "
          f"after load {torch.cuda.memory_allocated() / 2**30:.1f}GB", flush=True)

    lo, hi = sigma_range(args.expert)
    log_path = os.path.join(out_dir, "loss.csv")
    if resume and os.path.exists(log_path):  # drop the rows a crashed run logged after its last checkpoint
        rows = [r for r in open(log_path) if r.split(",")[0].isdigit() and int(r.split(",")[0]) < start]

        def rewrite():
            with open(log_path + ".tmp", "w") as f:
                f.writelines(rows)
            os.replace(log_path + ".tmp", log_path)
        retry_enospc(rewrite, "loss.csv")
    pending = []  # loss.csv rows not yet on disk (kept while the NAS quota is full)

    def flush_log():
        def w():
            with open(log_path, "a") as f:
                f.writelines(pending)
        if pending and try_enospc(w):
            pending.clear()
    t0, ema = time.time(), None
    for step in range(start, args.steps + 1):
        cid, x0, y, ad_in, has_cond = load_batch(files, dev, args.mode, args.cond_drop, args.tracks, zbank)
        sigma = lo + (hi - lo) * torch.rand((), device=dev)
        noise = torch.randn_like(x0)
        xt = (1 - sigma) * x0 + sigma * noise
        inp = torch.cat([xt, mask, y.float()], dim=1)
        ad.set(ad_in)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = tr(hidden_states=inp.bfloat16(), timestep=(sigma * 1000)[None], encoder_hidden_states=text,
                      return_dict=False)[0]
        loss = ((pred.float() - (noise - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + ad_params, 1.0)
        opt.step()
        ad.set(None)

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        pending.append(f"{step},{loss.item():.5f},{sigma.item():.4f},{int(has_cond)},{cid}\n")
        if step % 10 == 0 or step <= 3:
            flush_log()
            el, done = time.time() - t0, step - start + 1
            try_enospc(lambda: print(f"step {step} loss {loss.item():.4f} ema {ema:.4f} | {el / done:.1f}s/it | "
                                     f"eta {(args.steps - step) * el / done / 3600:.1f}h | "
                                     f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True))
        if step % args.save_every == 0 or step == args.steps:
            ck = {"lora": get_peft_model_state_dict(tr), "adapter": ad.state_dict(), "mode": args.mode,
                  "tracks": args.tracks,
                  "expert": args.expert, "lora_rank": args.lora_rank, "step": step}
            path, last = os.path.join(out_dir, f"ckpt_{step:05d}.pt"), os.path.join(out_dir, "ckpt_last.pt")

            def save():  # write to .tmp first so a failed write never leaves a truncated ckpt behind
                torch.save(ck, path + ".tmp")
                os.replace(path + ".tmp", path)
                # ckpt_last is a hard link, not a 2nd copy: the NAS quota is tight (ENOSPC on 2026-11-22)
                if os.path.exists(last + ".tmp"):
                    os.remove(last + ".tmp")
                os.link(path, last + ".tmp")
                os.replace(last + ".tmp", last)
            retry_enospc(save, f"ckpt {step}")
            flush_log()
            try_enospc(lambda: print(f"saved ckpt {step}", flush=True))
    retry_enospc(flush_log, "loss.csv")
    retry_enospc(lambda: print("DONE", flush=True), "DONE line")


if __name__ == "__main__":
    main()
