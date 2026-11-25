#!/bin/bash
# Baseline 1 (raw straight-line poke) on a 3rd GPU while the main run trains (user OK'd a 3rd card, 2026-11-21).
# Same recipe as the main run and Baseline 0; the two experts run one after the other on one idle GPU.
# scripts/run_baselines_and_bench.sh waits for the "DONE" lines of both logs instead of training b1 itself.
#     nohup scripts/run_b1.sh > logs/run_b1.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for e in high low; do
  run=openvid_b1_$e
  [ -f outputs/train/$run/ckpt_04000.pt ] && continue
  G=$(scripts/wait_free_gpus.sh 1)
  stamp "train $run on $G"
  CUDA_VISIBLE_DEVICES=$G $P src/train.py --run $run --clips data/clips_a14b_openvid --expert $e --steps 4000 \
    --save_every 500 --mode hybrid --tracks line > logs/train_$run.log 2>&1 || { stamp "FAILED $run"; exit 1; }
  stamp "$run done"
done
stamp "ALL DONE"
