#!/bin/bash
# Takes over steps 4-5 of scripts/run_openvid_a14b.sh after the high expert OOMed at step 1 (2026-11-21: another
# user's job took 14 GB of its GPU during the 20-min model load). The low expert kept running; this script restarts
# high on an idle GPU (train.py now reserves its peak memory up front), waits for both, then runs the held-out
# generation and writes "ALL DONE" to logs/run_openvid_a14b.log, which scripts/run_baselines_and_bench.sh waits on.
#     nohup scripts/resume_openvid_a14b.sh >> logs/run_openvid_a14b.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
CLIPS=data/clips_a14b_openvid
TEST=data/clips_a14b_openvid_test
HIGH=openvid_hybrid_high
LOW=openvid_hybrid_low
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }

if [ ! -f outputs/train/$HIGH/ckpt_04000.pt ]; then
  G=$(scripts/wait_free_gpus.sh 1)
  stamp "restart train high on $G (low keeps running)"
  mv logs/train_$HIGH.log logs/train_${HIGH}_oom.log 2>/dev/null
  CUDA_VISIBLE_DEVICES=$G $P src/train.py --run $HIGH --clips $CLIPS --mode hybrid --expert high --steps 4000 \
    --save_every 500 > logs/train_$HIGH.log 2>&1 &
fi
for run in $HIGH $LOW; do
  until grep -qx DONE logs/train_$run.log; do
    if grep -q Traceback logs/train_$run.log; then stamp "TRAIN FAILED: $run (see logs/train_$run.log)"; exit 1; fi
    sleep 300
  done
done
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
