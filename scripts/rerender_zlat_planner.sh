#!/bin/bash
# 11-25 00:26 the zlat planner render (scripts/run_zlat.sh) OOMed when another user's job landed on its GPU.
# Retry on idle GPUs (team rule via wait_free_gpus.sh), p1 and p4 in parallel on separate GPUs (with the zlat
# oracle render that is <= 3 GPUs); bench_pexels.py skips clips already rendered.
#     nohup scripts/rerender_zlat_planner.sh >> logs/run_zlat.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }
CK="outputs/train/openvid_zlat_high/ckpt_last.pt outputs/train/openvid_zlat_low/ckpt_last.pt"
worker() {  # n_pokes
  local n=$1 g log=logs/bench_render_zlat_planner_p$1.log
  for try in 1 2 3 4 5 6 7 8; do
    g=$(scripts/wait_free_gpus.sh 1)
    stamp "rerender zlat planner p$n on $g (try $try)"
    echo "=== $(date) GPU $g try $try" >> $log
    CUDA_VISIBLE_DEVICES=$g $P src/bench_pexels.py render --ckpt $CK --cond planner --n_pokes $n --out outputs/bench/zlat \
      >> $log 2>&1 && { stamp "zlat planner p$n done"; return 0; }
    tail -n 3 $log | grep -q OutOfMemoryError || break
    sleep 120
  done
  stamp "RENDER FAILED: zlat planner p$n (gave up)"; return 1
}
worker 1 & a=$!
sleep 90  # let wait_free_gpus see the first worker's GPU as busy
worker 4 & b=$!
wait $a && wait $b && touch outputs/bench/zlat/planner/.done && stamp "zlat planner done (.done)"
