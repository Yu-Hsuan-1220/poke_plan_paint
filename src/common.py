"""Shared paths, constants and helpers for the Koala -> Wan2.2-I2V-A14B poke-to-video pipeline.

Conventions
  - ZipMo tracks: (y, x), normalized to [-1, 1] with 2p/(H-1) - 1 (align-corners style), 64 steps.
  - Video for Wan: 65 frames, 320x576 (center crop of the original frame to 9:16... i.e. 320:576), ~12-15 fps.
    Frame j = ZipMo track step j for j < 64; frame 64 has no track (it is one stride later).
  - Wan2.1 VAE (used by A14B): 16 ch, /8 spatial, /4 temporal -> latents [16, 17, 40, 72].
  - "crop coords": tracks normalized w.r.t. the 320x576 crop; "orig coords": w.r.t. the original frame.
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIPMO_DIR = os.path.join(ROOT, "third_party", "zipmo")
if ZIPMO_DIR not in sys.path:
    sys.path.insert(0, ZIPMO_DIR)

HF_CACHE = os.environ.get("HF_CACHE", "/project/YuHsuanLee/hf_cache/hub")
WAN_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
ZIPMO_REPO = "CompVis/ZipMo"

T_STEPS = 64  # ZipMo track length
N_FRAMES = 65  # video frames (4n+1); frame j = track step j
H, W = 320, 576  # video resolution
VAE_S, VAE_T = 8, 4  # Wan2.1 VAE compression
LAT_H, LAT_W = H // VAE_S, W // VAE_S  # 40, 72
N_LAT = (N_FRAMES - 1) // VAE_T + 1  # 17
BOUNDARY = 0.9  # A14B: `transformer` (high noise) for t >= 900, `transformer_2` below


def wan_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(WAN_ID, cache_dir=HF_CACHE, local_files_only=True)


# ----------------------------------------------------------------------------- ZipMo
def _zipmo_file(name):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(ZIPMO_REPO, name, cache_dir=HF_CACHE)


def load_planner(device="cuda", dtype=torch.bfloat16):
    """Sparse-poke planner (contains its VAE as .vae)."""
    from safetensors.torch import load_file
    from zipmo.planner import ZipMoPlanner_Sparse
    from zipmo.vae import ZipMoVAE

    m = ZipMoPlanner_Sparse(vae=ZipMoVAE())
    m.load_state_dict(load_file(_zipmo_file("zipmo_planner_sparse.safetensors")))
    return m.eval().requires_grad_(False).to(device=device, dtype=dtype)


def denormalize_z(z):
    """Koala z.npy is planner.normalize_latents(vae.encode(...).mean); undo it ONCE before decoding."""
    shift, scale = -0.175, 11.6
    return z.float() * (scale ** 0.5 / 0.5 ** 0.5) + shift


@torch.no_grad()
def decode_queries(vae, z_raw, start_frame, query_pos, chunk=80):
    """z_raw [K,256,16] (denormalized); start_frame [K,h,w,3] in [-1,1]; query_pos [N,2] -> tracks [K,N,64,2].
    The ZipMo decoder was trained with 80 query points per call, so decode in chunks of 80."""
    K = z_raw.shape[0]
    dev, dt = next(vae.parameters()).device, next(vae.parameters()).dtype
    start_emb = vae.img_embedder(start_frame.to(dev, dt))
    N = query_pos.shape[0]
    pad = (-N) % chunk
    q = torch.cat([query_pos, query_pos[:1].expand(pad, -1)], 0) if pad else query_pos
    q = q.to(dev, dt).reshape(-1, chunk, 2)
    out = [vae.decode(z_raw.to(dev, dt), query_pos=qc[None].expand(K, -1, -1), points_per_track=T_STEPS,
                      start_emb=start_emb) for qc in q]
    return torch.cat(out, dim=1)[:, :N].float()


# ----------------------------------------------------------------------------- geometry
def grid_queries(n_h, n_w):
    """Cell centers of an n_h x n_w grid, [n_h*n_w, 2] normalized yx, row-major."""
    ys = torch.linspace(-1, 1, n_h + 1)
    xs = torch.linspace(-1, 1, n_w + 1)
    gy, gx = torch.meshgrid((ys[:-1] + ys[1:]) / 2, (xs[:-1] + xs[1:]) / 2, indexing="ij")
    return torch.stack([gy, gx], -1).reshape(-1, 2)


def crop_box(h, w):
    """Center crop of an h x w frame to the H:W aspect -> (y0, x0, th, tw)."""
    tw, th = min(w, round(h * W / H)), min(h, round(w * H / W))
    return (h - th) // 2, (w - tw) // 2, th, tw


def orig_to_crop(yx, box, hw):
    y0, x0, th, tw = box
    h, w = hw
    py, px = (yx[..., 0] + 1) / 2 * (h - 1), (yx[..., 1] + 1) / 2 * (w - 1)
    cy, cx = (py - y0) * (H - 1) / (th - 1), (px - x0) * (W - 1) / (tw - 1)
    return torch.stack([cy / (H - 1) * 2 - 1, cx / (W - 1) * 2 - 1], -1)


def crop_to_orig(yx, box, hw):
    y0, x0, th, tw = box
    h, w = hw
    cy, cx = (yx[..., 0] + 1) / 2 * (H - 1), (yx[..., 1] + 1) / 2 * (W - 1)
    py, px = y0 + cy * (th - 1) / (H - 1), x0 + cx * (tw - 1) / (W - 1)
    return torch.stack([py / (h - 1) * 2 - 1, px / (w - 1) * 2 - 1], -1)


def crop_resize(frames_uint8, box):
    """[T,h,w,3] uint8 -> [T,H,W,3] uint8 (center crop + antialiased resize)."""
    y0, x0, th, tw = box
    fr = frames_uint8[:, y0:y0 + th, x0:x0 + tw].permute(0, 3, 1, 2).float()
    fr = torch.nn.functional.interpolate(fr, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
    return fr.clamp(0, 255).round().byte().permute(0, 2, 3, 1)


def read_video(path):
    import decord

    vr = decord.VideoReader(path, num_threads=4)
    fr = vr.get_batch(list(range(len(vr))))
    return fr if isinstance(fr, torch.Tensor) else torch.from_numpy(fr.asnumpy())


# ----------------------------------------------------------------------------- Wan2.1 VAE
class WanVAE:
    """Wan2.1 VAE of A14B with the pipeline's latent normalization."""

    def __init__(self, device="cuda", dtype=torch.float32):
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(wan_path(), subfolder="vae", torch_dtype=dtype).to(device).eval()
        c = self.vae.config
        self.mean = torch.tensor(c.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
        self.std = torch.tensor(c.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode(self, video):
        """video [1,3,F,H,W] in [-1,1] -> normalized latents [1,16,F',h,w] (mode, like the pipeline)."""
        lat = self.vae.encode(video.to(self.device, self.dtype)).latent_dist.mode()
        return (lat - self.mean) / self.std

    @torch.no_grad()
    def encode_frames(self, frames_uint8):
        """[T,H,W,3] uint8 -> [16,F',h,w]."""
        x = frames_uint8.permute(3, 0, 1, 2)[None].float().div(127.5).sub(1)
        return self.encode(x)[0]

    @torch.no_grad()
    def encode_i2v_condition(self, first_uint8, n_frames=N_FRAMES):
        """The pipeline's condition latent: VAE([first frame, 0, ..., 0]) -> [16,F',h,w]."""
        x = first_uint8.permute(2, 0, 1)[None, :, None].float().div(127.5).sub(1)  # 1,3,1,H,W
        x = torch.cat([x, torch.zeros(1, 3, n_frames - 1, *x.shape[-2:])], dim=2)
        return self.encode(x)[0]

    @torch.no_grad()
    def decode(self, lat):
        """normalized [1,16,F',h,w] -> uint8 [F,H,W,3]."""
        v = self.vae.decode((lat.to(self.device, self.dtype) * self.std + self.mean)).sample
        return ((v[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu()


def i2v_mask(n_lat=N_LAT, h=LAT_H, w=LAT_W, device="cpu"):
    """A14B's 4-channel first-frame mask: 1 on latent frame 0, 0 elsewhere -> [4,F',h,w]."""
    m = torch.zeros(4, n_lat, h, w, device=device)
    m[:, 0] = 1
    return m


# ----------------------------------------------------------------------------- text
@torch.no_grad()
def encode_prompt(prompts, device="cuda", max_len=512):
    """UMT5 embeddings exactly like WanImageToVideoPipeline._get_t5_prompt_embeds -> [B,512,4096] bf16."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    p = wan_path()
    tok = AutoTokenizer.from_pretrained(p, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(p, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    t = tok(prompts, padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    mask = t.attention_mask.to(device)
    emb = te(t.input_ids.to(device), mask).last_hidden_state
    seq = mask.gt(0).sum(1)
    out = torch.zeros_like(emb)
    for i, n in enumerate(seq):
        out[i, :n] = emb[i, :n]
    del te
    torch.cuda.empty_cache()
    return out.cpu()


# ----------------------------------------------------------------------------- io / viz
def save_video(path, frames_uint8, fps=12):
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = frames_uint8.numpy() if isinstance(frames_uint8, torch.Tensor) else frames_uint8
    imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)


def draw_tracks(img_uint8, tracks, every=1, color_by_time=True):
    """img [H,W,3] uint8 numpy; tracks [N,T,2] normalized yx (crop coords) -> image with polylines."""
    import cv2

    img = np.ascontiguousarray(img_uint8.copy())
    h, w = img.shape[:2]
    pts = ((tracks.float().cpu().numpy() + 1) / 2 * np.array([h - 1, w - 1]))[..., ::-1]  # -> (x, y)
    T = pts.shape[1]
    for n in range(0, pts.shape[0], every):
        for t in range(T - 1):
            c = cv2.applyColorMap(np.uint8([[int(255 * t / (T - 1))]]), cv2.COLORMAP_TURBO)[0, 0]
            cv2.line(img, tuple(int(v) for v in pts[n, t]), tuple(int(v) for v in pts[n, t + 1]),
                     (int(c[2]), int(c[1]), int(c[0])), 1, cv2.LINE_AA)
    return img
