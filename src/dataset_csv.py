"""One CSV that lists every clip used in the project, with its source, plus the helpers that rebuild a split from it.

    python src/dataset_csv.py export                 # (needs the local data/) -> datasets/clips.csv,
                                                     #    datasets/qualitative_images.csv
    python src/dataset_csv.py windows --root data/openvid_rebuild    # after openvid_fetch_train.py --from_csv and
                                                     #    openvid_windows.py: <root>/windows/train_windows.csv
    python src/dataset_csv.py split --clips data/clips_a14b_openvid  # move the 50 held-out clips to <clips>_test/

Splits in datasets/clips.csv:
    train         OpenVid-1M windows the four models were trained on (1950)
    heldout       OpenVid-1M windows held out from training (50; validation loss, held-out generation)
    bench_pexels  ZipMo's official open-domain benchmark, 68 Pexels clips with 40 GT tracks (all reported numbers)
    koala_pilot   Koala-36M windows (koala_train_v2) used only for the early overfit / data checks, not the final models
A window is n_frames frames starting at start_frame, taking every frame_skip-th frame of the source video.
Only identifiers and statistics are listed; no frames, tracks or captions are redistributed.
"""

import argparse
import csv
import glob
import json
import os
import shutil

from common import ROOT

OUT = os.path.join(ROOT, "datasets")
OPENVID_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M"
PEXELS_URL = "https://drive.google.com/drive/folders/1ddt4gPIbfnvnARRjL1YdWYpGS0J3tASN"
COLS = ["split", "clip_id", "source_dataset", "source_video", "source_part", "youtube_id", "start_frame",
        "frame_skip", "n_frames", "fps", "height", "width", "motion_score", "cam_disp", "obj_p90", "face_area",
        "source_url"]


def rnd(v, k=3):
    return "" if v in ("", None) else round(float(v), k)


def cmd_export(_):
    import cv2
    import torch

    rows = []
    # OpenVid clean set (2000 windows): train / heldout
    win = list(csv.DictReader(open(os.path.join(ROOT, "data/openvid_clean_set/windows/train_windows.csv"))))
    part = {r["video"]: r["part"] for r in
            csv.DictReader(open(os.path.join(ROOT, "data/openvid_clean_set/candidates/openvid_clean_selected.csv")))}
    test = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(ROOT, "data/clips_a14b_openvid_test/*.pt"))}
    train = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(ROOT, "data/clips_a14b_openvid/*.pt"))}
    for r in win:
        w = r["window"]
        split = "heldout" if w in test else "train" if w in train else "unused"
        vid = r["videoID"] + ".mp4"
        rows.append(dict(split=split, clip_id=w, source_dataset="OpenVid-1M", source_video=vid,
                         source_part=part.get(vid, ""), youtube_id=r["youtube_id"], start_frame=0,
                         frame_skip=int(float(r["frame_skip"])), n_frames=65, fps=rnd(r["fps"]), height=r["H"],
                         width=r["W"], motion_score=rnd(r["motion_score"]), cam_disp=rnd(r["cam_disp"]),
                         obj_p90=rnd(r["obj_p90"]), face_area=rnd(r["face_area"], 4), source_url=OPENVID_URL))
    # Pexels benchmark (ZipMo official)
    raw = os.path.join(ROOT, "data/raw/pexels_official")
    gt = torch.load(os.path.join(raw, "gt_tracks.pt"), weights_only=True, map_location="cpu")
    for name in sorted(gt):
        cap = cv2.VideoCapture(os.path.join(raw, "pexels", f"original-{name}.mp4"))
        fps, h, w, n = (cap.get(p) for p in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH,
                                             cv2.CAP_PROP_FRAME_COUNT))
        rows.append(dict(split="bench_pexels", clip_id=name, source_dataset="Pexels (ZipMo open-domain benchmark)",
                         source_video=f"pexels/original-{name}.mp4", source_part="", youtube_id="", start_frame=0,
                         frame_skip=1, n_frames=int(n), fps=rnd(fps), height=int(h), width=int(w), motion_score="",
                         cam_disp="", obj_p90="", face_area="", source_url=PEXELS_URL))
    # Koala pilot (koala_train_v2)
    kroot = os.path.join(ROOT, "data/koala_train_v2")
    for r in csv.DictReader(open(os.path.join(kroot, "train_windows.csv"))):
        m = json.load(open(os.path.join(kroot, r["window"], "meta.json")))
        rows.append(dict(split="koala_pilot", clip_id=r["window"], source_dataset="Koala-36M",
                         source_video=r["videoID"], source_part="", youtube_id=r["youtube_id"],
                         start_frame=m.get("raw_start", ""), frame_skip=int(float(r["frame_skip"])), n_frames=65,
                         fps=rnd(r["fps"]), height=r["H"], width=r["W"], motion_score="", cam_disp="", obj_p90="",
                         face_area="", source_url=""))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "clips.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, COLS)
        wr.writeheader()
        wr.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print("datasets/clips.csv:", counts)

    # qualitative input photos (annotation site), with credits
    ims = json.load(open(os.path.join(ROOT, "annotate/data/images.json")))
    cols = ["id", "query", "title", "author", "license", "license_url", "source", "orig_width", "orig_height"]
    with open(os.path.join(OUT, "qualitative_images.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        for im in ims:
            wr.writerow({k: im.get(k, "") for k in cols[:7]} | dict(orig_width=im["orig_size"][0],
                                                                   orig_height=im["orig_size"][1]))
    print(f"datasets/qualitative_images.csv: {len(ims)} images")


def rows_of(splits):
    return [r for r in csv.DictReader(open(os.path.join(OUT, "clips.csv"))) if r["split"] in splits]


def cmd_windows(args):
    """train_windows.csv (the list prep_koala.py reads) for the train + heldout OpenVid windows found under root."""
    wroot = os.path.join(args.root, "windows")
    names = [r["clip_id"] for r in rows_of({"train", "heldout"})]
    have = [w for w in names if os.path.exists(os.path.join(wroot, w, "z.npy"))]
    with open(os.path.join(wroot, "train_windows.csv"), "w", newline="") as f:
        f.write("window\n" + "".join(w + "\n" for w in have))
    print(f"{len(have)} / {len(names)} windows -> {wroot}/train_windows.csv")


def cmd_split(args):
    test_dir = args.clips.rstrip("/") + "_test"
    os.makedirs(test_dir, exist_ok=True)
    n = 0
    for r in rows_of({"heldout"}):
        src = os.path.join(args.clips, r["clip_id"] + ".pt")
        if os.path.exists(src):
            shutil.move(src, os.path.join(test_dir, r["clip_id"] + ".pt"))
            n += 1
    print(f"moved {n} held-out clips -> {test_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    w = sub.add_parser("windows")
    w.add_argument("--root", required=True)
    s = sub.add_parser("split")
    s.add_argument("--clips", default="data/clips_a14b_openvid")
    a = ap.parse_args()
    {"export": cmd_export, "windows": cmd_windows, "split": cmd_split}[a.cmd](a)


if __name__ == "__main__":
    main()
