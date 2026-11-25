#!/bin/bash
# After scripts/run_openvid_a14b.sh (main method, hybrid + ZipMo tracks, both experts): train the two baselines with
# the SAME recipe (OpenVid clean set minus the 50 held-out clips, both experts, 4000 steps each, LoRA r32, lr 1e-4,
# cond dropout 0.1, save every 500), then render + score the ZipMo Pexels benchmark (68 clips, K=1, 1 and 4 pokes).
# At most 2 GPUs, always taken with scripts/wait_free_gpus.sh.
#     Baseline 0 (plain fine-tune):  --mode none
#     Baseline 1 (raw poke):         --mode hybrid --tracks line
#     nohup scripts/run_baselines_and_bench.sh > logs/run_baselines_and_bench.log 2>&1 &
set -e
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
CLIPS=data/clips_a14b_openvid
STEPS=4000
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }

MAIN="outputs/train/openvid_hybrid_high/ckpt_last.pt outputs/train/openvid_hybrid_low/ckpt_last.pt"
train_one() {  # name, expert, extra train args -- one expert on one idle GPU
  local name=$1 e=$2; shift 2
  local run=openvid_${name}_$e
  [ -f outputs/train/$run/ckpt_$(printf %05d $STEPS).pt ] && return
  local g; g=$(scripts/wait_free_gpus.sh 1)
  stamp "train $run on $g"
  CUDA_VISIBLE_DEVICES=$g $P src/train.py --run $run --clips $CLIPS --expert $e --steps $STEPS --save_every 500 \
    "$@" > logs/train_$run.log 2>&1 && stamp "train $run done" || stamp "TRAIN FAILED: $run"
}
# Baseline 0, max 3 GPUs in total with B1 and the main preview (2026-11-22):
#   high: once the main method's first 8 Pexels preview videos exist (user: start B0 after one version is out)
#   low:  once scripts/run_main_preview.sh is done and its GPU is free
until grep -q "held-out generation" logs/run_openvid_a14b.log; do sleep 120; done
train_one b0 high --mode none &
until grep -q "ALL DONE" logs/run_openvid_a14b.log; do sleep 300; done
sleep 120  # let the preview's GPU count as idle (>= 60 s) before picking
train_one b0 low --mode none &

# Baseline 1 trains in scripts/run_b1.sh; when it is done, its GPU runs the bench smoke test (1 clip, the
# oracle / line / none conditions; planner was already exercised by the preview) while B0 low finishes.
until grep -q "ALL DONE" logs/run_b1.log 2>/dev/null; do
  grep -q FAILED logs/run_b1.log 2>/dev/null && { stamp "b1 FAILED (see logs/run_b1.log)"; exit 1; }
  sleep 300
done
stamp "train b1 done"
GA=$(scripts/wait_free_gpus.sh 1)
stamp "bench smoke test on $GA"
for cond in oracle line none; do
  CUDA_VISIBLE_DEVICES=$GA $P src/bench_pexels.py render --ckpt $MAIN --cond $cond --n 1 --n_pokes 1 \
    --out outputs/bench_smoke >> logs/bench_smoke.log 2>&1 || stamp "SMOKE FAILED: render $cond (see logs/bench_smoke.log)"
done
stamp "bench smoke finished"
wait  # B0 high + low
grep -q "TRAIN FAILED" logs/run_baselines_and_bench.log && { stamp "stopping: a B0 run failed"; exit 1; }
stamp "train b0 done"

ck() { echo outputs/train/openvid_$1_high/ckpt_last.pt outputs/train/openvid_$1_low/ckpt_last.pt; }
# each finished render set leaves outputs/bench/<run>/<cond>/.done; scripts/run_eval_remote.sh scores it right away
# on the gpu4/5/6 RTX 4090s (EPE/MSE + VBench), so no scoring happens here.
render() {  # gpu, run, cond, ckpts...
  local g=$1 run=$2 cond=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $P src/bench_pexels.py render --ckpt "$@" --cond $cond --out outputs/bench/$run \
    && touch outputs/bench/$run/$cond/.done || stamp "RENDER FAILED: $run $cond"
}
read -r GA GB <<<"$(scripts/wait_free_gpus.sh 2)"
stamp "bench render on $GA $GB"
(
  render $GA main planner $MAIN
  render $GA b1 line $(ck b1)
) > logs/bench_render_A.log 2>&1 &
(
  render $GB main oracle $MAIN
  render $GB b0 none $(ck b0)
  render $GB vanilla none
) > logs/bench_render_B.log 2>&1 &
wait
stamp "ALL DONE"
