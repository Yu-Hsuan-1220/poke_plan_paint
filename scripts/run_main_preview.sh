#!/bin/bash
# Replaces the 2-GPU held-out step of scripts/resume_openvid_a14b.sh with 1 GPU (user wants to see the main
# method's videos early, 2026-11-22; gpu7 had only 1 idle card): Pexels planner p1 on 8 clips first, then the
# held-out OpenVid clips (high+low, oracle + planner), then Pexels planner p1 on the rest. p4 and oracle are
# rendered later by scripts/run_baselines_and_bench.sh (it skips clips that already exist). Finally writes
# "ALL DONE" to logs/run_openvid_a14b.log so that script starts the smoke test + Baseline 0.
#     nohup scripts/run_main_preview.sh >> logs/run_openvid_a14b.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }
MAIN="outputs/train/openvid_hybrid_high/ckpt_last.pt outputs/train/openvid_hybrid_low/ckpt_last.pt"
TEST=data/clips_a14b_openvid_test

G=$(scripts/wait_free_gpus.sh 1)
export CUDA_VISIBLE_DEVICES=$G
stamp "preview on GPU $G: Pexels planner p1, first 8 clips"
$P src/bench_pexels.py render --ckpt $MAIN --cond planner --n_pokes 1 --n 8 --out outputs/bench/main \
  > logs/bench_render_main_preview.log 2>&1 || stamp "FAILED Pexels preview"
stamp "held-out generation (high+low)"
IDS=$(ls $TEST/*.pt | head -12 | xargs -n1 basename | sed 's/\.pt$//' | paste -sd,)
$P src/infer.py --ckpt $MAIN --clips $TEST --ids="$IDS" --cond oracle planner --out outputs/infer/openvid_high_low \
  > logs/infer_openvid_high_low.log 2>&1 || stamp "FAILED held-out"
stamp "Pexels planner p1, all 68 clips"
$P src/bench_pexels.py render --ckpt $MAIN --cond planner --n_pokes 1 --out outputs/bench/main \
  >> logs/bench_render_main_preview.log 2>&1 || stamp "FAILED Pexels p1"
stamp "ALL DONE"
