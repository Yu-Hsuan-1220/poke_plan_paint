#!/bin/bash
# 2026-11-22 22:22 the NAS quota ran out (ENOSPC) and B0 high (step 3845), B1 low (2895) and B0 low (1680) crashed
# while flushing loss.csv; scripts/run_baselines_and_bench.sh and scripts/run_b1.sh died with them. This script
# takes over the rest of both, on at most 3 gpu7 GPUs at any time (scripts/run_zlat.sh takes slot 2's GPU later):
#   slot 1: resume B0 high -> bench smoke test -> render main planner, main oracle -> (after B1) render b1 line
#   slot 2: resume B1 low  -> "ALL DONE" in logs/run_b1.log, which starts scripts/run_zlat.sh
#   slot 3: resume B0 low  -> render b0 none, vanilla none
# --resume restarts from <run>/ckpt_last.pt (LoRA + adapter; AdamW moments restart, which we note in the report).
# Each finished render set leaves outputs/bench/<run>/<cond>/.done for scripts/run_eval_remote.sh.
#     nohup scripts/continue_after_enospc.sh > logs/continue_after_enospc.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
CLIPS=data/clips_a14b_openvid
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }
MAIN="outputs/train/openvid_hybrid_high/ckpt_last.pt outputs/train/openvid_hybrid_low/ckpt_last.pt"
ck() { echo outputs/train/openvid_$1_high/ckpt_last.pt outputs/train/openvid_$1_low/ckpt_last.pt; }

resume() {  # run, train args -- picks its own idle GPU (sets RES_GPU); returns train.py's status
  local run=$1 g; shift
  [ -f outputs/train/$run/ckpt_04000.pt ] && return 0
  for try in 1 2 3 4 5; do
    g=$(scripts/wait_free_gpus.sh 1)
    stamp "resume $run on $g (try $try)"
    echo "=== $(date) resume on GPU $g" >> logs/train_$run.log
    CUDA_VISIBLE_DEVICES=$g $P src/train.py --run $run --clips $CLIPS --steps 4000 --save_every 500 --resume "$@" \
      >> logs/train_$run.log 2>&1 && { stamp "train $run done"; RES_GPU=$g; return 0; }
    # somebody took the GPU between our idle check and our 37 GB reservation: pick another one
    tail -n 5 logs/train_$run.log | grep -q "OutOfMemoryError: CUDA out of memory. Tried to allocate 37.00 GiB" \
      || break
    sleep 120
  done
  stamp "TRAIN FAILED: $run"; return 1
}
keep_or_new() {  # keep GPU $1 if it is still idle, else wait for another idle one
  local used; used=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && echo $1 || scripts/wait_free_gpus.sh 1
}
render() {  # gpu, run, cond, ckpts...
  local g=$1 run=$2 cond=$3; shift 3
  g=$(keep_or_new $g)
  stamp "render $run $cond on $g"
  CUDA_VISIBLE_DEVICES=$g $P src/bench_pexels.py render --ckpt "$@" --cond $cond --out outputs/bench/$run \
    >> logs/bench_render_$run.log 2>&1 && touch outputs/bench/$run/$cond/.done && stamp "render $run $cond done" \
    || stamp "RENDER FAILED: $run $cond"
  LAST_GPU=$g
}

(
  resume openvid_b0_high --expert high --mode none || exit 1
  g=$(keep_or_new $RES_GPU)
  stamp "bench smoke test on $g"
  for cond in oracle line none; do
    CUDA_VISIBLE_DEVICES=$g $P src/bench_pexels.py render --ckpt $MAIN --cond $cond --n 1 --n_pokes 1 \
      --out outputs/bench_smoke >> logs/bench_smoke.log 2>&1 || stamp "SMOKE FAILED: render $cond (see logs/bench_smoke.log)"
  done
  render $g main planner $MAIN; g=$LAST_GPU
  render $g main oracle $MAIN; g=$LAST_GPU
  until [ -f outputs/train/openvid_b1_low/ckpt_04000.pt ]; do sleep 300; done
  render $g b1 line $(ck b1)
) &
sleep 90  # let wait_free_gpus see the GPU slot 1 took as busy
(
  resume openvid_b1_low --expert low --mode hybrid --tracks line || { echo "[$(date '+%m-%d %H:%M')] FAILED openvid_b1_low" >> logs/run_b1.log; exit 1; }
  echo "[$(date '+%m-%d %H:%M')] ALL DONE" >> logs/run_b1.log
) &
sleep 90
(
  resume openvid_b0_low --expert low --mode none || exit 1
  render $RES_GPU b0 none $(ck b0); g=$LAST_GPU
  render $g vanilla none
) &
wait
stamp "ALL DONE"
