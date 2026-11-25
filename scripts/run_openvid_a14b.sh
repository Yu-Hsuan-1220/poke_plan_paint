#!/bin/bash
# Full A14B fine-tuning on the OpenVid clean set with at most 2 GPUs (2026-11-21):
#   1. wait for 2 idle GPUs (scripts/wait_free_gpus.sh: no process, < 1000 MiB, >= 60 s)
#   2. prep_koala.py on both (2 shards)                     -> data/clips_a14b_openvid/*.pt
#   3. hold out 50 clips (seed 0)                           -> data/clips_a14b_openvid_test/
#   4. train both experts in parallel, hybrid mode, 4000 steps each (high on GPU A, low on GPU B)
#   5. held-out generation: high-only vs high+low, oracle and planner condition, 12 clips
# Every step is resumable (prep skips existing clips; training is not, so it is skipped when ckpt_last exists).
#     nohup scripts/run_openvid_a14b.sh > logs/run_openvid_a14b.log 2>&1 &
set -e
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
SRC=data/openvid_clean_set/windows
CLIPS=data/clips_a14b_openvid
TEST=data/clips_a14b_openvid_test
STEPS=4000
HIGH=openvid_hybrid_high
LOW=openvid_hybrid_low
mkdir -p logs

stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }

read -r GA GB <<<"$(scripts/wait_free_gpus.sh 2)"
stamp "prep on GPUs $GA $GB"
CUDA_VISIBLE_DEVICES=$GA $P src/prep_koala.py --src $SRC --out $CLIPS --shard 0 --n_shards 2 > logs/prep_openvid_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GB $P src/prep_koala.py --src $SRC --out $CLIPS --shard 1 --n_shards 2 > logs/prep_openvid_1.log 2>&1 &
wait
stamp "prep done: $(ls $CLIPS/*.pt | wc -l) clips"

$P - <<EOF
import glob, os, random
files = sorted(glob.glob("$CLIPS/*.pt"))
os.makedirs("$TEST", exist_ok=True)
if not glob.glob("$TEST/*.pt"):
    for f in random.Random(0).sample(files, 50):
        os.replace(f, os.path.join("$TEST", os.path.basename(f)))
print("train", len(glob.glob("$CLIPS/*.pt")), "test", len(glob.glob("$TEST/*.pt")))
EOF

read -r GA GB <<<"$(scripts/wait_free_gpus.sh 2)"
stamp "train high on $GA, low on $GB"
for spec in "$GA high $HIGH" "$GB low $LOW"; do
  read -r g e run <<<"$spec"
  if [ ! -f outputs/train/$run/ckpt_last.pt ]; then
    CUDA_VISIBLE_DEVICES=$g $P src/train.py --run $run --clips $CLIPS --mode hybrid --expert $e --steps $STEPS \
      --save_every 500 > logs/train_$run.log 2>&1 &
  fi
done
wait
stamp "training done"

IDS=$(ls $TEST/*.pt | head -12 | xargs -n1 basename | sed 's/\.pt$//' | paste -sd,)
read -r GA GB <<<"$(scripts/wait_free_gpus.sh 2)"
stamp "held-out generation on $GA $GB"
CUDA_VISIBLE_DEVICES=$GA $P src/infer.py --ckpt outputs/train/$HIGH/ckpt_last.pt --clips $TEST --ids="$IDS" \
  --cond oracle planner --out outputs/infer/openvid_high_only > logs/infer_openvid_high_only.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GB $P src/infer.py --ckpt outputs/train/$HIGH/ckpt_last.pt outputs/train/$LOW/ckpt_last.pt \
  --clips $TEST --ids="$IDS" --cond oracle planner --out outputs/infer/openvid_high_low \
  > logs/infer_openvid_high_low.log 2>&1 &
wait
stamp "ALL DONE"
