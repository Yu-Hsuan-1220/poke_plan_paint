"""Download the ZipMo official open-domain benchmark (68 Pexels clips + 40 GT tracks each) from the Google Drive
folder linked in the ZipMo README (https://github.com/CompVis/long-term-motion, "open-domain evaluation"),
then verify every file against pexels_official.md5 (next to this script).

    pip install gdown                       # once
    python src/download_pexels.py           # -> data/raw/pexels_official/{gt_tracks.pt, pexels/*.mp4}
"""

import argparse
import hashlib
import os
import sys

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1ddt4gPIbfnvnARRjL1YdWYpGS0J3tASN"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(out, manifest):
    """manifest lines: '<md5>  ./<relative path>'. -> list of problems (empty = ok)."""
    problems = []
    for line in open(manifest):
        digest, rel = line.split(maxsplit=1)
        path = os.path.join(out, rel.strip())
        if not os.path.exists(path):
            problems.append(f"missing {rel.strip()}")
        elif md5(path) != digest:
            problems.append(f"md5 mismatch {rel.strip()}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/raw/pexels_official"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "pexels_official.md5"))
    args = ap.parse_args()

    if os.path.exists(args.out) and not verify(args.out, args.manifest):
        print(f"already complete: {args.out}")
        return
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not found: pip install gdown")
    os.makedirs(args.out, exist_ok=True)
    gdown.download_folder(DRIVE_FOLDER, output=args.out, quiet=False, resume=True)

    problems = verify(args.out, args.manifest)
    if problems:
        sys.exit("download incomplete or changed:\n  " + "\n  ".join(problems))
    n = sum(1 for _ in open(args.manifest))
    print(f"OK: {n} files verified -> {args.out}")


if __name__ == "__main__":
    main()
