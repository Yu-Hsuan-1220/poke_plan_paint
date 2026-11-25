# Poke annotation for the qualitative results

The Pexels benchmark pokes (first n GT tracks) are fine for numbers but often poor as demos, so the qualitative
figures use our own photos with hand-drawn pokes. Each photo gets up to two poke sets with different counts (1-4),
e.g. a 1-poke and a 4-poke version; photos that don't work are skipped.

## Input images

`fetch_images.py` collects openly licensed photos from Wikimedia Commons (Quality images; CC0 / PD / CC BY /
CC BY-SA only) and stores them center-cropped to 1152x640 (2x the model's 576x320). The spec, for anything added by hand:

- a real photograph (no illustration, text, collage, heavy filter), sharp, no watermark / border
- landscape; the model sees a 576:320 (1.8:1) center crop, so the subject must be in the central region
- source >= 1280x720
- a clearly movable subject covering roughly 10-50% of the frame (people, animals, vehicles, boats, water, flags,
  plants...), not hidden behind other things
- credit (title, author, license, source) is kept in `data/images.json` and must be shown wherever the image or
  a video made from it appears (slides, report, web)

Never use OpenVid / Koala frames here: that data is for class use only and must not be shared or published.

## Running

```bash
python3 annotate/fetch_images.py                 # -> site/img/*.jpg + data/images.json (rerun to add more)
python3 annotate/server.py --new-key && cat annotate/data/team_key.txt    # give this key to teammates
nohup python3 annotate/server.py --port 8766 > annotate/data/server.log 2>&1 &
# nginx: annotate/nginx.conf  ->  https://<host>/annotate/   (review page: /annotate/review.html)
```

Local test: `python3 annotate/server.py --port 8766 --serve-static`, open http://localhost:8766/.

## Annotating

Enter a name and the team key once. Per image: pick the poke count, drag from a point on the object to where it
should be after ~4 s (a click without dragging = keep this point still), save; then optionally a second set with
another count, optionally with a short motion description in Chinese (turned into a full English prompt
later for the text experiments), or skip the image (removes it for everyone), or pass (only for you, this time). Keys: 1-4, Z undo,
Enter save, N pass. An image shown to one person is held for 10 min so two people don't annotate the same one.
`review.html` shows every set with delete / unskip and downloads the JSON.

`GET /api/export` (with the `X-Team-Key` header) returns
`{"sets": [{"image", "file", "n", "pokes": [[y0, x0, y1, x1], ...], "annotator", "prompt", "credit"}]}`
(`prompt` = the annotator's Chinese motion description, may be empty), pokes normalized to
[-1, 1] on the 576x320 frame, the same convention as `src/bench_pexels.py`.

`data/` (images.json, annotations.jsonl, team_key.txt) and `site/img/` are git-ignored.
