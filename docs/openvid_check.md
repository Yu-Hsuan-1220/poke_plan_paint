# OpenVid-1M vs Koala v2: track quality and ZipMo reconstruction (2026-11-18)

Question: is OpenVid-1M (static-camera clips) better training data than koala_train_v2, and does ZipMo describe
its motion better?

## Setup

- **OpenVid**: 150 clips with `camera motion == static`, at least 65 frames at about 12 fps, and not
  celebv/magictime. They come from 10 zip parts spread across the dataset (20, 35, …, 170), 30 clips in each of
  5 OpenVid motion-score buckets. One window per clip, taken from the start: 65 frames every `round(fps/12)`
  frames. Resolution is mostly 1080p (108 clips) or 720p (35).
- **Koala**: all 781 koala_train_v2 windows (these already passed ZipMo's filter and the three camera rules).
- **Tracks**: the same CoTracker3 setup on both datasets (1,024 random queries, 64 steps; `src/tracking.py` is
  ported from the Koala preprocessing). The same filters apply to both: ZipMo static fraction > 0.5,
  ≥ 80 valid tracks, and camera rules R1–R3. 121 of the 150 OpenVid windows pass.
- **ZipMo check** (`src/zipmo_recon_check.py`): per window, take 80 valid tracks. Encode 40 of them plus the
  first frame into z, decode at all 80 start points, and measure the error on the **40 held-out tracks**.
  Units are ZipMo's [0,128] scale.
- **Motion statistics** (`src/motion_stats.py`): a RANSAC similarity transform from frame 0 to frame t gives
  the camera motion. The residual left after removing it is the object motion.

## Results (medians over windows that pass all filters)

| | Koala (781) | OpenVid (121) |
|---|---|---|
| camera motion, % of diagonal (`cam_motion`) | 2.92 | **0.43** |
| camera displacement of the tracks (`cam_disp`) | 4.49 | **0.56** |
| ZipMo static fraction | 0.77 | **0.93** |
| how far held-out tracks move by the end (GT motion) | **6.29** | 2.22 |
| fraction of tracks with object motion > 2 | **0.44** | 0.29 |
| windows where almost nothing moves (object-moving fraction < 0.1) | **15%** | 21% |
| windows where the camera moves but nothing else does | 6.4% | 5.0% |
| ZipMo held-out end-point error | 0.87 | **0.53** |

ZipMo reconstruction at **matched motion** (held-out end-point error binned by GT motion):

| GT motion | 0–1 | 1–2 | 2–4 | 4–8 | 8–16 | > 16 |
|---|---|---|---|---|---|---|
| Koala | 0.35 (75) | 0.75 (69) | 0.84 (114) | 1.10 (225) | 1.10 (176) | 0.86 (122) |
| OpenVid | 0.27 (25) | 0.49 (31) | 0.52 (26) | 0.89 (26) | 0.86 (13) | none |

Regression `log EPE ~ log motion + dataset`: on OpenVid the error is a factor **0.80 [95% CI 0.70, 0.92]** of
Koala's at the same amount of motion.

## Conclusions

1. **ZipMo does reconstruct OpenVid better**, by about 20% at matched motion, and the error is lower in every
   motion bin. This holds even though ZipMo was *trained on Koala* (`koala-tapnext-subset` in ZipMo's
   train.py), so the gain comes from cleaner data: a much steadier camera and mostly 1080p.
2. **OpenVid's camera really is static**: camera motion is about 7x smaller than in Koala v2.
3. **But OpenVid has less motion.** Tracks move about 1/3 as far, fewer tracks move, and there are more
   "almost nothing moves" windows. No window reaches GT motion > 16.
4. **OpenVid's motion score does not select dynamic windows.** The highest bucket (≥ 25) has the *least*
   track motion (median 1.1; object-moving fraction 0.16). Any dynamics filter has to come from our own tracks
   (`obj_frac_moving`, `obj_p90`).
   A second check that does not use CoTracker gives the same answer. Take the mean |frame difference| inside
   the window at 320x180. Its Spearman correlation is 0.63 with track motion, but −0.31 with motion score
   (−0.03 for motion score < 25). Clips with motion score 40–64 are people sitting and talking or cooking, with
   a frame difference of 0.3–1.1. Per-clip numbers: `outputs/openvid_yt_motionscore.csv`.
5. **Content**: the eligible static pool has 307k clips, but 42% of their captions describe talking,
   interviews or news. Hands and objects being pushed or poured are rare (e.g. "pour" 1.6%).

Requiring `obj_frac_moving ≥ 0.2 and obj_p90 ≥ 4` keeps 60% of the passing OpenVid windows (Koala: 72%).
With the camera rules, about 49% of tracked OpenVid clips survive.

## Caveats

- OpenVid sample: 150 clips (121 after the filters). The bins with > 8 motion have only 13 OpenVid windows.
- Only the VAE was tested (z from 40 real tracks). The planner (poke → z) was not.
- Like Koala, the OpenVid static clips are YouTube-derived (Panda-70M style ids). Same rule: class use only.
- Pexels / pixabay / mixkit clips in OpenVid are 64 frames at 24 fps, too short for 65 frames at ~12 fps.

Files: `outputs/koala_vs_openvid.csv` (per window), `outputs/zipmo_recon_{koala,openvid_yt}/vis/*.jpg`
(green = CoTracker3, red = ZipMo, held-out tracks).

## Training set selection (2026-11-20)

The first 2000-window set (`data/openvid_train`, motion filters only) was ~60% people talking to the camera:
OpenVid's long clips all come from YouTube (Panda-70M), and "static camera" there mostly means interviews and
vlogs. Two alternatives were checked:

- **Pexels stock footage** (Pexels-400k metadata, full-length originals): the content is right (animals, people
  walking, hands and objects), but only 3.5% of a 258-clip pilot has a static camera (cam_disp < 2), and none of
  the people-walking clips do. Stock footage is shot with camera moves. Rejected.
- **OpenVid with content filters**: caption prefilter (`CLEAN_NEG`), frontal-face area ≤ 2%, cam_disp < 2.5.
  Of the caption-clean candidates, 9% pass; adopted.

| median | Koala (781) | OpenVid v1 (2000) | OpenVid clean (2000) |
|---|---|---|---|
| obj_frac_moving | 0.44 | 0.38 | 0.32 |
| obj_p90 | 9.2 | 9.3 | **11.8** |
| cam_disp | 4.5 | 2.1 | **0.25** |
| share with cam_disp < 1 | 29% | 42% | **77%** |

Content of the clean set: mostly hands manipulating objects (cooking, crafts, toys, tools). Caption keywords:
object interaction 30%, people walking / sport 6%, animals 6%. In a 48-clip preview a few people still face the
camera (~10%) and some are in-car shots. Static camera + large subject motion (walking, running animals) is rare
in real footage, so those categories stay small.
