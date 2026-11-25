#!/bin/bash
# Implicit condition (z itself as cross-attention tokens, --mode latent; port of the teammate's 5B method B with
# ZipMo-grid position init), queued last (user, 2026-11-21): starts when Baseline 1 is done, on one idle GPU,
# high then low expert (same recipe as the other three: 4000 steps, LoRA r32, lr 1e-4, cond dropout 0.1).
# Alongside the bench renders this keeps the total at <= 3 GPUs. Then Pexels render (planner p1/p4 + oracle);
# scoring is done by scripts/run_eval_remote.sh on the gpu4/5/6 4090s.
#     nohup scripts/run_zlat.sh > logs/run_zlat.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }

until grep -q "ALL DONE" logs/run_b1.log 2>/dev/null; do sleep 300; done
for e in high low; do
  run=openvid_zlat_$e
  [ -f outputs/train/$run/ckpt_04000.pt ] && continue
  G=$(scripts/wait_free_gpus.sh 1)
  stamp "train $run on $G"
  CUDA_VISIBLE_DEVICES=$G $P src/train.py --run $run --clips data/clips_a14b_openvid --expert $e --steps 4000 \
    --save_every 500 --mode latent > logs/train_$run.log 2>&1 || { stamp "FAILED $run"; exit 1; }
  stamp "$run done"
done

CK="outputs/train/openvid_zlat_high/ckpt_last.pt outputs/train/openvid_zlat_low/ckpt_last.pt"
read -r GA GB <<<"$(scripts/wait_free_gpus.sh 2)"
stamp "bench render on $GA $GB (scored by scripts/run_eval_remote.sh on the gpu4/5/6 4090s)"
for spec in "$GA planner" "$GB oracle"; do
  read -r g cond <<<"$spec"
  ( CUDA_VISIBLE_DEVICES=$g $P src/bench_pexels.py render --ckpt $CK --cond $cond --out outputs/bench/zlat \
      && touch outputs/bench/zlat/$cond/.done || stamp "RENDER FAILED: zlat $cond" ) > logs/bench_render_zlat_$cond.log 2>&1 &
done
wait
stamp "ALL DONE"
