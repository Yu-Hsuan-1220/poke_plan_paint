"""CoTracker3 tracking + ZipMo track filters + camera-motion metrics, matching how koala_train_v2 was built.

Ported from the team's Koala preprocessing (spencer, ZipMo-Poke2Video/scripts/preprocess/{zipmo_utils,
camera_motion}.py) so that OpenVid windows are tracked and filtered exactly like the Koala ones:
  - raw CoTracker3 offline (keeps visibility/confidence probabilities), 1,024 random queries on frame 0,
    6x6 support grid, 64 frames
  - ZipMo filters: static-camera fraction (TrackerDataModule.get_static_tracks), track mask
    (visible at t=0, confidence > 0.5 for all 64 steps, stays in frame)
  - camera rules R1 pan+zoom / R2 too far / R3 cut (doc "Koala 影片固定鏡頭篩選規則")
Tracks are (y, x) in [-1, 1] w.r.t. the original frame, [T=64, N, 2].
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT

sys.path.insert(0, os.path.join(ROOT, "third_party", "co-tracker"))

COTRACKER3_OFFLINE_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
T_ZIPMO = 64
CAMERA_THRESHOLDS = dict(pan_zoom_motion=5.0, pan_zoom_zoom=7.0, too_far_motion=25.0, cut_ratio=8.0)


# ----------------------------------------------------------------------------- CoTracker3
def load_cotracker(device="cuda"):
    from cotracker.models.build_cotracker import build_cotracker

    model = build_cotracker(checkpoint=None, offline=True, window_len=60, v2=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(COTRACKER3_OFFLINE_URL, map_location="cpu"))
    return model.eval().to(device)


@torch.no_grad()
def track_points(model, video, queries_xy, support_grid=6):
    """video uint8 [T,H,W,3] (on device); queries_xy [N,2] pixel (x, y) at frame 0.
    -> tracks_xy [T,N,2] pixels, vis [T,N], conf [T,N] (probabilities)."""
    from cotracker.models.core.model_utils import get_points_on_a_grid

    T, H, W, _ = video.shape
    ih, iw = model.model_resolution
    v = F.interpolate(video.permute(0, 3, 1, 2).float(), (ih, iw), mode="bilinear", align_corners=True)[None]
    q = queries_xy.float().clone() * queries_xy.new_tensor([(iw - 1) / (W - 1), (ih - 1) / (H - 1)]).float()
    N = q.shape[0]
    q = torch.cat([torch.zeros_like(q[:, :1]), q], 1)  # (t=0, x, y)
    if support_grid > 0:
        g = get_points_on_a_grid(support_grid, (ih, iw), device=q.device)[0]
        q = torch.cat([q, torch.cat([torch.zeros_like(g[:, :1]), g], 1)], 0)
    tracks, vis, conf, _ = model(video=v, queries=q[None], iters=6)
    tracks, vis, conf = tracks[0, :, :N], vis[0, :, :N], conf[0, :, :N]
    tracks[0] = q[:N, 1:]  # CoTrackerPredictor's query-point correction
    vis[0] = 1.0
    return tracks * tracks.new_tensor([(W - 1) / (iw - 1), (H - 1) / (ih - 1)]), vis, conf


def xy_to_yx_norm(tracks_xy, H, W):
    yx = tracks_xy.flip(-1)
    return yx / yx.new_tensor([H - 1, W - 1]) * 2 - 1


# ----------------------------------------------------------------------------- ZipMo filters
def static_fraction(tracks_yx):
    """TrackerDataModule.get_camera_static keeps a clip iff this is > 0.5."""
    flow = torch.diff(tracks_yx, dim=0) / 2.0
    return ((flow.norm(dim=-1) < 0.01).float().mean(0) > 0.95).float().mean().item()


def track_mask(vis, conf, tracks_yx=None):
    """visible at t=0, confident for all steps, (optionally) inside the frame for all steps -> bool [N]."""
    m = (conf > 0.5).all(0) & (vis[0] > 0.5)
    if tracks_yx is not None:
        m = m & ((tracks_yx >= -1) & (tracks_yx <= 1)).all(-1).all(0)
    return m


# ----------------------------------------------------------------------------- camera rules
def _corners(H, W):
    return np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [(W - 1) / 2, (H - 1) / 2]], np.float32)


def camera_track_metrics(tracks_yx, vis, conf, H, W):
    """cam_motion: max_t displacement of corners+centre under the RANSAC similarity 0 -> t (% of diagonal);
    zoom_total: |chained frame-to-frame zoom| (%)."""
    xy = ((tracks_yx.astype(np.float32) + 1) / 2 * np.array([H - 1, W - 1], np.float32))[..., ::-1]
    ok = (conf > 0.5).all(0) & (vis[0] > 0.5)
    xy = np.ascontiguousarray(xy[:, ok])
    diag, ref = float(np.hypot(H, W)), _corners(H, W)
    motion, log_zoom = [], 0.0
    if xy.shape[1] < 3:  # (almost) no confident track: camera unknown -> rejected by R2 below
        return dict(cam_motion=100.0, zoom_total=0.0)
    for t in range(1, xy.shape[0]):
        M, _ = cv2.estimateAffinePartial2D(xy[0], xy[t], method=cv2.RANSAC, ransacReprojThreshold=0.005 * diag)
        if M is not None:
            motion.append(float(np.linalg.norm(ref @ M[:, :2].T + M[:, 2] - ref, axis=1).mean()) / diag * 100)
        M, _ = cv2.estimateAffinePartial2D(xy[t - 1], xy[t], method=cv2.RANSAC, ransacReprojThreshold=0.003 * diag)
        if M is not None:
            log_zoom += float(np.log(np.sqrt(abs(np.linalg.det(M[:, :2])))))
    return dict(cam_motion=round(max(motion) if motion else 100.0, 3), zoom_total=round(abs(log_zoom) * 100, 2))


def cut_ratio(frames):
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32) for f in frames[:T_ZIPMO]]
    d = np.array([np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))])
    return round(float(d.max() / (np.median(d) + 0.5)), 2)


def camera_rules(m, th=CAMERA_THRESHOLDS):
    r = []
    if m["cam_motion"] >= th["pan_zoom_motion"] and m["zoom_total"] >= th["pan_zoom_zoom"]:
        r.append("R1_pan_zoom")
    if m["cam_motion"] >= th["too_far_motion"]:
        r.append("R2_too_far")
    if m["cut_ratio"] >= th["cut_ratio"]:
        r.append("R3_cut")
    return r


# ----------------------------------------------------------------------------- content
_FACE = None


def face_area(frames, at=(0, 32, 64)):
    """Largest frontal face (OpenCV Haar cascade) as a fraction of the frame, max over frames `at`.
    Large frontal faces = someone talking to the camera, the dominant content of static-camera YouTube clips."""
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(ROOT, "third_party", "opencv", "haarcascade_frontalface_default.xml"))
    best = 0.0
    for i in at:
        if i >= len(frames):
            continue
        g = cv2.cvtColor(cv2.resize(frames[i], (640, 360)), cv2.COLOR_RGB2GRAY)
        fs = _FACE.detectMultiScale(g, 1.1, 6, minSize=(30, 30))
        if len(fs):
            best = max(best, max(w * h for _, _, w, h in fs) / (640 * 360))
    return round(float(best), 4)
