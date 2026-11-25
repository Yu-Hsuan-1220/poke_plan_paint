"""Generate videos with Wan2.2-I2V-A14B (+ trained LoRA/adapter on one expert) from a motion condition.

Conditions (--cond):
    oracle  : the clip's own motion (decoded z_gt[k]) -- teacher-forcing condition, upper bound / sanity check
    planner : ZipMo sparse planner from one poke = the clip's most-moving valid CoTracker track (the real method)
    none    : no motion condition (vanilla I2V condition latent)

--ckpt takes up to two checkpoints, one per expert (e.g. --ckpt <high run>/ckpt_last.pt <low run>/ckpt_last.pt).
Both experts are kept in CPU RAM (bf16, ~28 GB each) and the active one is moved to the GPU.
Guidance: on a trained expert, motion CFG (uncond = same prompt, vanilla condition latent, no adapter; this
matches the training dropout); on an untrained expert, Wan's usual text CFG with the negative prompt.

    python src/infer.py --ckpt outputs/train/<run>/ckpt_last.pt --clips data/clips_a14b --n 2 --cond oracle planner none
"""

import argparse
import glob
import os

import numpy as np
import torch

from common import BOUNDARY, N_LAT, ROOT, T_STEPS, WanVAE, draw_tracks, grid_queries, i2v_mask, load_planner, \
    save_video, wan_path, LAT_H, LAT_W, decode_queries
from motion_cond import MotionAdapter, build_condition
from train import SUBFOLDER


def most_moving_poke(c):
    """[1,5] poke (y0, x0, y1, x1, 1) from the clip's most-moving valid track, crop coords."""
    tr = c["tracks"].float()
    ok = (c["vis"][:, 0] > 0.5) & (c["conf"] > 0.5).all(1) & (tr.abs() <= 0.95).all(-1).all(-1)
    disp = (tr[:, -1] - tr[:, 0]).norm(dim=-1) * ok
    i = int(disp.argmax())
    return torch.cat([tr[i, 0], tr[i, -1], torch.ones(1)])[None]


class Generator:
    def __init__(self, ckpts=(), device="cuda", reserve_gb=34):
        """ckpts: 0-2 checkpoint paths, at most one per expert (high / low); all must use the same --mode.
        reserve_gb: GPU memory held from the start (peak ~32.6 GB). Loading both experts into CPU RAM takes
        10-20 min and a shared GPU can be taken by someone else meanwhile; the caching allocator keeps it for us
        (so model() below must not call empty_cache)."""
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig, set_peft_model_state_dict

        self.dev = device
        if reserve_gb:
            torch.empty(int(reserve_gb * 2**30), dtype=torch.uint8, device=device)
        cks = [torch.load(p, map_location="cpu") for p in ckpts if p]
        self.cks = {c["expert"]: c for c in cks}
        assert len(self.cks) == len(cks), "two checkpoints for the same expert"
        modes = {c["mode"] for c in cks}
        assert len(modes) <= 1, f"checkpoints use different modes: {modes}"
        self.mode = modes.pop() if modes else "y"
        self.models, self.adapters = {}, {}
        for e in ("high", "low"):
            m = WanTransformer3DModel.from_pretrained(wan_path(), subfolder=SUBFOLDER[e], torch_dtype=torch.bfloat16)
            m.requires_grad_(False).eval()
            if e in self.cks:
                ck = self.cks[e]
                r = ck["lora_rank"]
                m.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
                set_peft_model_state_dict(m, ck["lora"])
                ad = MotionAdapter(self.mode if self.mode != "none" else "y")
                ad.load_state_dict(ck["adapter"])
                self.adapters[e] = ad.attach(m).to(device=device, dtype=torch.bfloat16)
            self.models[e] = m
        self.on_gpu = None
        self.vae = WanVAE(device)
        pe = torch.load(os.path.join(ROOT, "data/prompt_embeds.pt"))
        self.text, self.neg = pe["embeds"].to(device), pe["neg_embeds"].to(device)
        self.mask = i2v_mask(device=device)[None]
        self.planner = None

    def model(self, e):
        if self.on_gpu != e:
            if self.on_gpu is not None:
                self.models[self.on_gpu].to("cpu")  # its memory stays cached for the other expert (same size)
            self.models[e].to(self.dev)
            self.on_gpu = e
        return self.models[e]

    @torch.no_grad()
    def plan(self, first_uint8, pokes, k=4, seed=43, steps=50):
        """-> dense tracks [k, 2880, 64, 2] at the 40x72 cell centers (crop coords)."""
        if self.planner is None:
            self.planner = load_planner(self.dev)
        sf = (first_uint8.float() / 127.5 - 1)[None].to(self.dev, torch.bfloat16).expand(k, -1, -1, -1)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        z = torch.randn(k, 256, 16, device=self.dev, dtype=torch.bfloat16, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat = self.planner.sample(z, points_per_traj=T_STEPS, query_pos=None,
                                      track_conds=pokes.to(self.dev, torch.bfloat16)[None].expand(k, -1, -1),
                                      start_frame=sf, sample_steps=steps, decode_latent=False)
            lat = self.planner.denormalize_latents(lat)
            return decode_queries(self.planner.vae, lat, sf, grid_queries(LAT_H, LAT_W)).cpu()

    @torch.no_grad()
    def generate(self, first_uint8, tracks, steps=30, shift=3.0, motion_cfg=3.0, text_cfg=3.5, seed=0, z=None):
        """first [H,W,3] uint8; tracks [2880,64,2] or None -> frames uint8 [65,H,W,3].
        A --mode latent checkpoint uses z [256,16] (normalized ZipMo latent) instead of tracks."""
        from diffusers import UniPCMultistepScheduler

        dev = self.dev
        cond_lat = self.vae.encode_i2v_condition(first_uint8)[None].float()  # 1,16,17,40,72
        if self.mode == "latent":
            tracks = z
        y_c, ad_in = build_condition(self.mode, cond_lat, None if tracks is None else tracks[None].to(dev))
        y_u = cond_lat
        sched = UniPCMultistepScheduler.from_pretrained(wan_path(), subfolder="scheduler", flow_shift=shift)
        sched.set_timesteps(steps, device=dev)
        g = torch.Generator(device=dev).manual_seed(seed)
        lat = torch.randn(1, 16, N_LAT, LAT_H, LAT_W, device=dev, generator=g)

        def run(e, y, text, adx):
            ad = self.adapters.get(e)
            if ad is not None:
                ad.set(adx)
            inp = torch.cat([lat, self.mask, y.float()], 1).bfloat16()
            out = self.model(e)(hidden_states=inp, timestep=t.expand(1), encoder_hidden_states=text,
                                return_dict=False)[0]
            if ad is not None:
                ad.set(None)
            return out.float()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in sched.timesteps:
                e = "high" if t >= BOUNDARY * 1000 else "low"
                if e in self.cks and tracks is not None:  # motion CFG on a trained expert
                    pc, cfg = run(e, y_c, self.text, ad_in), motion_cfg
                    pu = run(e, y_u, self.text, None) if cfg > 1 else None
                else:  # text CFG on a vanilla expert
                    pc, cfg = run(e, y_u, self.text, None), text_cfg
                    pu = run(e, y_u, self.neg, None) if cfg > 1 else None
                pred = pc if pu is None else pu + cfg * (pc - pu)
                lat = sched.step(pred, t, lat, return_dict=False)[0]
        return self.vae.decode(lat)


def panel(first, tracks, frames):
    imgs = [first.numpy() if tracks is None else
            draw_tracks(first.numpy(), tracks.view(LAT_H, LAT_W, T_STEPS, 2)[::4, ::4].reshape(-1, T_STEPS, 2))]
    imgs += [frames[len(frames) // 2].numpy(), frames[-1].numpy()]
    return np.concatenate(imgs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="0-2 checkpoints (one per expert), e.g. high and low")
    ap.add_argument("--clips", default="data/clips_a14b")
    ap.add_argument("--ids", default=None,
                    help="comma-separated clip ids; ids may start with '-', so write --ids=-abc_w0,xyz_w1")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--cond", nargs="+", default=["oracle"], choices=["oracle", "planner", "none"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--motion_cfg", type=float, default=3.0)
    ap.add_argument("--text_cfg", type=float, default=3.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    import cv2

    files = ([os.path.join(args.clips, i + ".pt") for i in args.ids.split(",")] if args.ids
             else sorted(glob.glob(os.path.join(args.clips, "*.pt")))[:args.n])
    G = Generator(args.ckpt)
    for f in files:
        c = torch.load(f, map_location="cpu")
        for cond in args.cond:
            od = os.path.join(args.out, c["id"], cond)
            if os.path.exists(os.path.join(od, "video.mp4")):
                continue
            os.makedirs(od, exist_ok=True)
            pokes = most_moving_poke(c)
            tracks = {"oracle": lambda: c["dec_tracks"][0].float(),
                      "planner": lambda: G.plan(c["first_frame"], pokes, k=1)[0],
                      "none": lambda: None}[cond]()
            frames = G.generate(c["first_frame"], tracks, args.steps, args.shift, args.motion_cfg, args.text_cfg)
            save_video(os.path.join(od, "video.mp4"), frames, fps=round(c["fps"]))
            cv2.imwrite(os.path.join(od, "panel.jpg"), cv2.cvtColor(panel(c["first_frame"], tracks, frames),
                                                                    cv2.COLOR_RGB2BGR))
            torch.save({"tracks": tracks, "pokes": pokes, "args": vars(args)}, os.path.join(od, "cond.pt"))
            print(f"{c['id']} [{cond}] done | peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
