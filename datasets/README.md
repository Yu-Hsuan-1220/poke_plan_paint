# Datasets

Every clip used in this project is listed in [`clips.csv`](clips.csv). The CSV contains **only identifiers and
statistics**: no frames, tracks or captions. To get the actual data, re-download it from the original sources
with the steps below.

| split | clips | source | used for |
|---|---|---|---|
| `train` | 1950 | [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | training all four models (Main / B0 / B1 / zlat) |
| `heldout` | 50 | OpenVid-1M | not trained on: validation loss and held-out generation |
| `bench_pexels` | 68 | Pexels, ZipMo's official open-domain benchmark ([Google Drive](https://drive.google.com/drive/folders/1ddt4gPIbfnvnARRjL1YdWYpGS0J3tASN), see [CompVis/long-term-motion](https://github.com/CompVis/long-term-motion)) | all reported numbers (EPE / MSE / VBench) |
| `koala_pilot` | 781 | Koala-36M (the teammate's `koala_train_v2`) | only the early overfit test and data comparison; not used by the final models |

[`qualitative_images.csv`](qualitative_images.csv) lists 270 Wikimedia Commons photos: the inputs for the
qualitative results, i.e. the images on the annotation website. Only CC0 / PD / CC BY / CC BY-SA licenses are
accepted; the title, author, license and source URL of each photo are in the CSV. Credit must be given wherever
one of these photos, or a video generated from it, appears (slides, report, web pages).

## Columns

| column | meaning |
|---|---|
| `clip_id` | clip name: `<source video>_w<k>` for OpenVid / Koala, the benchmark file name for Pexels |
| `source_dataset`, `source_video` | source dataset and file name in it (OpenVid: the `video` column of `OpenVid-1M.csv`) |
| `source_part` | the OpenVid zip that holds the file (`OpenVid_partN.zip`). Empty for 168 clips from the first download batch, whose list was deleted; `--from_csv` scans all parts and does not need this column |
| `youtube_id` | YouTube source of an OpenVid / Koala clip |
| `start_frame`, `frame_skip`, `n_frames` | clip = `n_frames` frames (65, about 12–15 fps) taken every `frame_skip` frames from frame `start_frame` of the source video |
| `fps`, `height`, `width` | frame rate after subsampling, original resolution |
| `motion_score` | OpenVid's own motion score |
| `cam_disp`, `obj_p90`, `face_area` | our filter statistics: camera displacement, 90th percentile of object motion, largest frontal-face area (see [docs/data_collection.md](../docs/data_collection.md)) |

## Rebuilding the data

```bash
# OpenVid: fetch only the 2000 videos in the CSV (range reads from the remote HF zips, no full zip downloads)
python src/openvid_fetch_train.py --from_csv datasets/clips.csv --out data/openvid_rebuild
python src/openvid_windows.py --root data/openvid_rebuild --shard 0 --n_shards 1   # CoTracker3 + 3 ZipMo z
python src/dataset_csv.py windows --root data/openvid_rebuild                        # -> windows/train_windows.csv
python src/prep_koala.py --src data/openvid_rebuild/windows --out data/clips_a14b_openvid
python src/dataset_csv.py split --clips data/clips_a14b_openvid                     # 50 heldout clips -> *_test/
python src/prep_z.py --clips data/clips_a14b_openvid data/clips_a14b_openvid_test --src data/openvid_rebuild/windows

# Pexels benchmark (needs pip install gdown; every file is checked against an md5 list)
python src/download_pexels.py               # -> data/raw/pexels_official/
python src/bench_pexels.py prep              # -> data/bench_pexels/
```

The CoTracker3 query points are determined by the clip name and a seed, so rerunning gives almost exactly our
tracks and z; only GPU floating-point differences cause tiny changes.

On the lab NAS the data is ready to use at `/project/YuHsuanLee/poke_to_video/data/`.

## Usage restrictions

The OpenVid-1M and Koala-36M videos come from YouTube and are **for class use only: do not redistribute or
publish them**. Public web pages (user study, annotation site) may only use Pexels clips and Wikimedia photos.

`clips.csv` is generated from the local `data/` by `python src/dataset_csv.py export`.
