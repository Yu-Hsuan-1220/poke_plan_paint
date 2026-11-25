"""Dense-track motion conditioning for Wan2.2-I2V-A14B.

The ZipMo tracks (decoded at the 40x72 latent-grid cell centers) carry the first-frame latent features along
their paths (Wan-Move-style splat; adapted from the teammate's 5B `build_track_guide`). Each latent frame f gets
    feat [16]  the splatted first-frame latent features
    disp [2]   displacement from the source cell (x4, so typical values are O(1))
    occ  [1]   occupancy (bilinear coverage, clamped to 1)
and is injected in one of three ways (`mode`):
    "y"      (I1) write feat into A14B's own 16-ch I2V condition latent: y_f = occ*feat + (1-occ)*cond_lat_f
             for f >= 1 (frame 0 untouched; the 4-ch mask is unchanged). No new parameters.
    "add"    (I2) the teammate's 5B approach: feat+disp+occ (19 ch) -> zero-init Conv3d(1x2x2) added to the
             patch-embedding output. y stays the vanilla condition latent.
    "hybrid" (I3) "y" for feat, plus disp+occ (3 ch) through the zero-init Conv3d.
    "latent" (implicit; the teammate's 5B method B) no tracks: the ZipMo latent z [256,16] itself -> MLP ->
             256 extra text tokens read by every cross-attention layer (see LatentAdapter). y stays vanilla.
Unconditional (condition dropout / CFG): vanilla condition latent, no adapter.

Latent frame f >= 1 covers video frames 4f-3..4f; with 65 frames video frame j = track step j, so latent frame f
represents track step 4f - 1.5 (the center of its 4 frames).
"""

import torch
from torch import nn

from common import N_LAT

HIDDEN = 5120  # A14B transformer width
MODES = ("y", "add", "hybrid", "latent")


def latent_frame_steps(n_lat=N_LAT):
    return torch.tensor([0.0] + [4 * f - 1.5 for f in range(1, n_lat)])


def sample_tracks_at(tracks, steps):
    """tracks [B,N,T,2]; steps [F] float -> [B,N,F,2] (linear interpolation in time)."""
    T = tracks.shape[2]
    s = steps.clamp(0, T - 1).to(tracks.device)
    i0 = s.floor().long()
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (s - i0.float()).view(1, 1, -1, 1).to(tracks.dtype)
    return tracks[:, :, i0] * (1 - w) + tracks[:, :, i1] * w


def splat_tracks(first_latent, tracks, n_lat=N_LAT):
    """first_latent [B,C,h,w]; tracks [B,h*w,64,2] normalized yx of the h x w cell centers (row-major)
    -> feat [B,C,F,h,w], disp [B,2,F,h,w], occ [B,1,F,h,w] (float32). Frame 0 = first_latent, occ 1, disp 0."""
    B, C, h, w = first_latent.shape
    assert tracks.shape[1] == h * w, (tracks.shape, h, w)
    F, dev = n_lat, first_latent.device
    feats = first_latent.float().flatten(2).transpose(1, 2)  # B,N,C
    pos = sample_tracks_at(tracks.float().to(dev), latent_frame_steps(F))  # B,N,F,2
    disp = (pos - pos[:, :, :1]) * 4.0
    py = (pos[..., 0] + 1) / 2 * h - 0.5  # continuous cell coords, centers at integers
    px = (pos[..., 1] + 1) / 2 * w - 0.5
    y0, x0 = py.floor(), px.floor()
    vals = torch.cat([feats[:, :, None].expand(-1, -1, F, -1), disp], -1)  # B,N,F,C+2
    D = vals.shape[-1]
    acc = torch.zeros(B, F, h * w, D, device=dev)
    wsum = torch.zeros(B, F, h * w, device=dev)
    vals_t = vals.permute(0, 2, 1, 3)  # B,F,N,D
    for dy in (0, 1):
        for dx in (0, 1):
            yy, xx = y0 + dy, x0 + dx
            wt = (1 - (py - yy).abs()) * (1 - (px - xx).abs())
            wt = wt * ((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w))
            lin = (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)).long().permute(0, 2, 1)  # B,F,N
            wt_t = wt.permute(0, 2, 1)
            acc.scatter_add_(2, lin[..., None].expand(-1, -1, -1, D), vals_t * wt_t[..., None])
            wsum.scatter_add_(2, lin, wt_t)
    g = acc / wsum.clamp(min=1e-6)[..., None] * (wsum > 1e-6)[..., None]
    g = g.view(B, F, h, w, D).permute(0, 4, 1, 2, 3)  # B,D,F,h,w
    feat, disp = g[:, :C].contiguous(), g[:, C:].contiguous()
    occ = wsum.clamp(max=1.0).view(B, 1, F, h, w)
    feat[:, :, 0], disp[:, :, 0], occ[:, :, 0] = first_latent.float(), 0, 1
    return feat, disp, occ


def build_condition(mode, cond_lat, tracks):
    """cond_lat [B,16,F,h,w] (vanilla I2V condition latent); tracks [B,h*w,64,2] or None
    -> (y [B,16,F,h,w], adapter input [B,c,F,h,w] or None). tracks=None -> unconditional.
    mode "latent": `tracks` is the normalized ZipMo z [B,256,16] instead, passed through to the adapter."""
    if tracks is None:
        return cond_lat, None
    if mode == "latent":
        return cond_lat, tracks
    feat, disp, occ = splat_tracks(cond_lat[:, :, 0], tracks, cond_lat.shape[2])
    cl = cond_lat.float()
    if mode == "add":
        return cond_lat, torch.cat([feat, disp, occ], 1)
    y = occ * feat + (1 - occ) * cl
    y[:, :, 0] = cl[:, :, 0]
    return y.to(cond_lat.dtype), (torch.cat([disp, occ], 1) if mode == "hybrid" else None)


def adapter_channels(mode):
    return {"y": 0, "add": 16 + 3, "hybrid": 3, "latent": 0}[mode]


def zipmo_grid_pos(h=16, w=16):
    """[h*w, 2] (y, x) centers of ZipMo's latent-token grid, row-major, in [-1, 1] (= zipmo.rope.make_axial_pos_2d
    for a square grid): z token i*w + j sits at image position (y_i, x_j)."""
    c = lambda n: (torch.arange(n) + 0.5) / n * 2 - 1
    return torch.stack(torch.meshgrid(c(h), c(w), indexing="ij"), -1).view(-1, 2)


def sincos_2d(pos, dim, max_freq=16.0):
    """[N, 2] positions in [-1, 1] -> [N, dim] sin/cos features (dim/4 frequencies per axis, pi .. max_freq*pi)."""
    f = torch.pi * max_freq ** torch.linspace(0, 1, dim // 4)
    a = pos[:, :, None] * f  # N,2,dim/4
    return torch.cat([a.sin(), a.cos()], -1).flatten(1)


class LatentAdapter(nn.Module):
    """ZipMo z [B,256,16] (normalized, as the planner samples it) -> [B,256,4096] tokens appended to the UMT5 text
    tokens (before Wan's own frozen text projection 4096 -> 5120). Port of the teammate's 5B `LatentAdapter`,
    with one change: the learned token positions start from a 2D sin/cos code of each z token's place on ZipMo's
    16x16 image grid (theirs: random), so cross-attention gets a spatial prior instead of learning it from scratch.
    Last layer zero-init: the 256 tokens start as identical constants after the text projection."""

    def __init__(self, z_dim=16, n_tokens=256, text_dim=4096, width=1024):
        super().__init__()
        self.pos = nn.Parameter(sincos_2d(zipmo_grid_pos(), width) * 0.5)  # std ~0.35, ~ in_proj(z) scale
        self.in_proj = nn.Linear(z_dim, width)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, text_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z):
        return self.mlp(self.in_proj(z.to(self.in_proj.weight.dtype)) + self.pos)


class MotionAdapter(nn.Module):
    """Zero-init Conv3d with the patch-embedding kernel; its output is added to the patch-embedding output."""

    def __init__(self, mode, hidden=HIDDEN, patch=(1, 2, 2)):
        super().__init__()
        self.mode = mode
        c = adapter_channels(mode)
        self.proj = nn.Conv3d(c, hidden, kernel_size=patch, stride=patch) if c else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.latent = LatentAdapter() if mode == "latent" else None
        self._x = None

    def attach(self, transformer):
        if self.proj is not None:
            transformer.patch_embedding.register_forward_hook(self._hook)
        if self.latent is not None:
            transformer.register_forward_pre_hook(self._text_hook, with_kwargs=True)
        return self

    def set(self, x):
        self._x = x

    def _hook(self, module, inp, out):
        if self._x is None:
            return out
        return out + self.proj(self._x.to(out.dtype))

    def _text_hook(self, module, args, kwargs):
        if self._x is None:
            return args, kwargs
        ehs = kwargs["encoder_hidden_states"]
        tok = self.latent(self._x).to(ehs.dtype).expand(ehs.shape[0], -1, -1)
        kwargs["encoder_hidden_states"] = torch.cat([ehs, tok], 1)
        return args, kwargs


# ----------------------------------------------------------------------------- Baseline 1: raw poke, no ZipMo
def clip_poke(tracks, vis, conf, top_frac=None, generator=None):
    """Poke [1,5] (y0, x0, y1, x1, 1) from a clip's CoTracker tracks (crop coords; [N,64,2], probs [N,64]):
    the most-moving valid track (evaluation) or, with top_frac, a random one among the top `top_frac` most-moving
    valid tracks (training). Same rule as the teammate's 5B Baseline 1."""
    tr = tracks.float()
    valid = (vis[:, 0] > 0.5) & (vis[:, -1] > 0.5) & (conf > 0.5).all(1) & (tr.abs() <= 0.95).all(-1).all(-1)
    disp = (tr[:, -1] - tr[:, 0]).norm(dim=-1) * valid
    if top_frac is None:
        i = int(disp.argmax())
    else:
        k = max(1, int(valid.sum().item() * top_frac))
        top = disp.topk(k).indices
        i = int(top[torch.randint(len(top), (1,), generator=generator)])
    return torch.cat([tr[i, 0], tr[i, -1], torch.ones(1)])[None]


def line_tracks(pokes, query, radius=0.12, n_steps=64):
    """Raw poke without ZipMo: query points within `radius` (normalized units) of a poke start move on a straight
    line at constant speed by (end - start); every other point stays. pokes [P,5], query [N,2] -> [N,64,2]."""
    tr = query[:, None, :].repeat(1, n_steps, 1).clone()
    alpha = torch.linspace(0, 1, n_steps)[None, :, None]
    for p in pokes:
        s, e = p[:2], p[2:4]
        near = (query - s).norm(dim=-1) < radius
        tr[near] = query[near][:, None] + alpha * (e - s)
    return tr
