#!/bin/bash
# Overfit sanity test: train the high-noise expert (hybrid / I3) on 8 clips, then generate on 2 of them with
# their own motion (oracle). If the videos follow the tracks, the pipeline can learn.
set -e
cd "$(dirname "$0")/.."
P=${PYTHON:-/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=overfit8_hybrid_high
$P src/train.py --run $RUN --mode hybrid --expert high --max_clips 8 --steps 400 --save_every 200 --cond_drop 0.1
IDS=$($P -c "import glob,os;f=sorted(glob.glob('data/clips_a14b/*.pt'))[:2];print(','.join(os.path.basename(x)[:-3] for x in f))")
$P src/infer.py --ckpt outputs/train/$RUN/ckpt_last.pt --ids=$IDS --cond oracle none --motion_cfg 1 --out outputs/infer/${RUN}_cfg1
$P src/infer.py --ckpt outputs/train/$RUN/ckpt_last.pt --ids=$IDS --cond oracle planner --motion_cfg 3 --out outputs/infer/${RUN}_cfg3
echo OVERFIT_TEST_DONE
