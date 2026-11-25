#!/bin/bash
# All benchmark scoring (EPE/MSE with CoTracker3 + VBench) on idle RTX 4090s of gpu4/5/6 via scripts/remote_run.sh,
# at most 2 at a time (user's limit, 2026-11-21). A job starts as soon as its render set is finished (marker
# outputs/bench/<run>/<cond>/.done, written by the render scripts) and is skipped if its result file exists.
#     nohup scripts/run_eval_remote.sh > logs/run_eval_remote.log 2>&1 &
P=/home_nfs/YuHsuanLee/miniconda3/envs/p2v/bin/python
cd /project/YuHsuanLee/poke_to_video
stamp() { echo "[$(date '+%m-%d %H:%M')] $*"; }
VB="PYTHONPATH=envs/vbench_pkgs $P src/vbench_eval.py"
EV="$P src/bench_pexels.py eval"
B=outputs/bench
# name | marker (empty = ready now) | result file | command (run from the repo root)
JOBS=(
"vb_gt||results/vbench/gt.json|$VB gt --tag gt"
"static||results/bench/static.json|$P src/bench_pexels.py static --out results/bench/static.json"
"ev_main_planner|$B/main/planner/.done|results/bench/main__planner.json|$EV $B/main/planner/p1 $B/main/planner/p4 --out results/bench/main__planner.json"
"ev_main_oracle|$B/main/oracle/.done|results/bench/main__oracle.json|$EV $B/main/oracle/all --out results/bench/main__oracle.json"
"ev_b1_line|$B/b1/line/.done|results/bench/b1__line.json|$EV $B/b1/line/p1 $B/b1/line/p4 --out results/bench/b1__line.json"
"ev_b0_none|$B/b0/none/.done|results/bench/b0__none.json|$EV $B/b0/none/all --out results/bench/b0__none.json"
"ev_vanilla_none|$B/vanilla/none/.done|results/bench/vanilla__none.json|$EV $B/vanilla/none/all --out results/bench/vanilla__none.json"
"vb_main_planner_p1|$B/main/planner/.done|results/vbench/main_planner_p1.json|$VB $B/main/planner/p1 --tag main_planner_p1"
"vb_main_planner_p4|$B/main/planner/.done|results/vbench/main_planner_p4.json|$VB $B/main/planner/p4 --tag main_planner_p4"
"vb_main_oracle|$B/main/oracle/.done|results/vbench/main_oracle.json|$VB $B/main/oracle/all --tag main_oracle"
"vb_b1_line_p1|$B/b1/line/.done|results/vbench/b1_line_p1.json|$VB $B/b1/line/p1 --tag b1_line_p1"
"vb_b1_line_p4|$B/b1/line/.done|results/vbench/b1_line_p4.json|$VB $B/b1/line/p4 --tag b1_line_p4"
"vb_b0_none|$B/b0/none/.done|results/vbench/b0_none.json|$VB $B/b0/none/all --tag b0_none"
"vb_vanilla_none|$B/vanilla/none/.done|results/vbench/vanilla_none.json|$VB $B/vanilla/none/all --tag vanilla_none"
"ev_zlat_planner|$B/zlat/planner/.done|results/bench/zlat__planner.json|$EV $B/zlat/planner/p1 $B/zlat/planner/p4 --out results/bench/zlat__planner.json"
"ev_zlat_oracle|$B/zlat/oracle/.done|results/bench/zlat__oracle.json|$EV $B/zlat/oracle/all --out results/bench/zlat__oracle.json"
"vb_zlat_planner_p1|$B/zlat/planner/.done|results/vbench/zlat_planner_p1.json|$VB $B/zlat/planner/p1 --tag zlat_planner_p1"
"vb_zlat_planner_p4|$B/zlat/planner/.done|results/vbench/zlat_planner_p4.json|$VB $B/zlat/planner/p4 --tag zlat_planner_p4"
"vb_zlat_oracle|$B/zlat/oracle/.done|results/vbench/zlat_oracle.json|$VB $B/zlat/oracle/all --tag zlat_oracle"
)
CL=logs/eval_claims
rm -rf $CL  # stale claims from an interrupted run; finished jobs are skipped by their result file
mkdir -p $CL results/bench results/vbench

worker() {
  while true; do
    left=0
    for j in "${JOBS[@]}"; do
      IFS='|' read -r name marker out cmd <<<"$j"
      [ -f "$out" ] && continue
      [ -e $CL/$name.failed ] && continue
      left=1
      [ -n "$marker" ] && [ ! -f "$marker" ] && continue
      mkdir $CL/$name 2>/dev/null || continue  # claimed by the other worker
      stamp "worker $1: start $name"
      scripts/remote_run.sh "$cmd" > logs/eval_$name.log 2>&1  # judged by the result file: python can exit non-zero at teardown after writing it
      if [ -f "$out" ]; then
        stamp "worker $1: done $name"
      else
        touch $CL/$name.failed; stamp "worker $1: FAILED $name (see logs/eval_$name.log)"
      fi
    done
    [ $left = 0 ] && return
    sleep 120
  done
}
worker 1 & sleep 5; worker 2 &
wait
stamp "ALL DONE"
