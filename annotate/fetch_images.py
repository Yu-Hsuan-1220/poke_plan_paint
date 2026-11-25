"""Collect openly licensed input photos for the qualitative poke results (Wikimedia Commons, no API key).

    python3 annotate/fetch_images.py                    # ~6 per query -> annotate/site/img/*.jpg + annotate/data/images.json
    python3 annotate/fetch_images.py --per_query 10     # more

Input spec (what the A14B model was trained on / runs at):
  - real photographs (no illustrations, text, collages, heavy filters), sharp, no watermark or border
  - landscape; the model sees a center crop at 576:320 (1.8:1), so the subject must sit in the central region
  - >= 1280x720 source; stored at 1152x640 (2x the model resolution, for annotation), rendered at 576x320
  - a clearly movable subject (people, animals, vehicles, water, flags, plants, ...) covering ~10-50% of the frame
Only licenses that allow derivatives (CC0 / public domain / CC BY / CC BY-SA); author + license + source page are
kept per image and must be credited wherever the image or a video generated from it is shown.
"""

import argparse
import hashlib
import html
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "site", "img")
MANIFEST = os.path.join(HERE, "data", "images.json")
API = "https://commons.wikimedia.org/w/api.php"
UA = "poke-to-video-class-project/0.1 (academic, non-commercial; python-urllib)"
OUT_W, OUT_H = 1152, 640  # 2x the model's 576x320
OK_LICENSE = re.compile(r"^(cc0|public domain|pd\b|pdm|cc by(-sa)? \d)", re.I)

QUERIES = [
    "dog running", "dog playing park", "cat walking", "horse running field", "cows pasture", "sheep flock",
    "bird perched branch", "ducks pond", "swan lake", "goat", "deer forest", "squirrel", "elephant", "giraffe",
    "person walking street", "cyclist road", "skateboarder", "runner park", "children playing playground",
    "street musician", "dancer", "tennis player", "soccer match", "chef cooking", "people crossing street",
    "car city street", "bus street", "tram", "train station platform", "motorcycle road", "airplane runway",
    "boat harbor", "sailboat sea", "kayak river", "ferry", "waterfall", "ocean waves beach", "river rapids",
    "fountain square", "flag wind", "windmill", "wind turbine", "trees wind", "flower field", "laundry clothesline",
    "hot air balloon", "kite beach", "swing playground", "ferris wheel", "carousel", "market stall",
    "construction crane", "smoke chimney", "candle flame", "fish aquarium", "horse carriage", "tractor field",
]


def get(url, params=None, retries=4):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def search(q, n):
    """Commons file search restricted to large landscape bitmaps in Quality images; -> imageinfo dicts."""
    params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": n * 5,
              "gsrsearch": f'{q} filetype:bitmap filew:>1500 fileh:>840 incategory:"Quality_images"',
              "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1600}
    d = json.loads(get(API, params))
    pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    return [(p["title"], p["imageinfo"][0]) for p in pages if p.get("imageinfo")]


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def crop_resize(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    tw, th = min(w, round(h * OUT_W / OUT_H)), min(h, round(w * OUT_H / OUT_W))
    x0, y0 = (w - tw) // 2, (h - th) // 2
    return im.crop((x0, y0, x0 + tw, y0 + th)).resize((OUT_W, OUT_H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_query", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    man = {m["id"]: m for m in json.load(open(MANIFEST))} if os.path.exists(MANIFEST) else {}
    seen = {m["source"] for m in man.values()}
    for q in QUERIES:
        got = sum(m["query"] == q for m in man.values())
        try:
            results = search(q, args.per_query)
        except Exception as e:  # noqa: BLE001
            print(f"{q}: search failed ({e})")
            continue
        for title, ii in results:
            if got >= args.per_query:
                break
            meta = ii.get("extmetadata", {})
            lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
            w, h = ii["width"], ii["height"]
            if ii["descriptionurl"] in seen or not OK_LICENSE.match(lic) or ii.get("mime") != "image/jpeg":
                continue
            if not 1.3 <= w / h <= 2.4:  # landscape, the 1.8:1 crop keeps most of the photo
                continue
            if strip_html(meta.get("Restrictions", {}).get("value")):  # e.g. personality rights, trademarks
                continue
            try:
                im = crop_resize(Image.open(io.BytesIO(get(ii.get("thumburl") or ii["url"]))))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {title}: {e}")
                continue
            iid = hashlib.sha1(ii["descriptionurl"].encode()).hexdigest()[:10]
            im.save(os.path.join(IMG, iid + ".jpg"), quality=90)
            man[iid] = {"id": iid, "file": f"img/{iid}.jpg", "query": q,
                        "title": re.sub(r"^File:|\.\w+$", "", title), "author": strip_html(
                            meta.get("Artist", {}).get("value"))[:120] or "unknown", "license": lic,
                        "license_url": meta.get("LicenseUrl", {}).get("value", ""), "source": ii["descriptionurl"],
                        "orig_size": [w, h]}
            seen.add(ii["descriptionurl"])
            got += 1
            time.sleep(0.3)
        print(f"{q}: {got}", flush=True)
        json.dump(list(man.values()), open(MANIFEST + ".tmp", "w"), indent=1, ensure_ascii=False)
        os.replace(MANIFEST + ".tmp", MANIFEST)
    print(f"{len(man)} images -> {IMG}")


if __name__ == "__main__":
    main()
