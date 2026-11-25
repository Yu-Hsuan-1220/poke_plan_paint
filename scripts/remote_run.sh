#!/bin/bash
# usage: remote_run.sh '<command run from the repo root>'
# Runs one GPU job on an idle RTX 4090 of gpu4/5/6 (shared /project), at most 2 such GPUs at a time across all
# callers (user's limit, 2026-11-21): slot locks in logs/remote_slots/{1,2}, each holding the "host idx" it uses.
# Caches point at envs/cache on /project (the remote /home disks are local and nearly full).
REPO=/project/YuHsuanLee/poke_to_video
SL=$REPO/logs/remote_slots
mkdir -p $SL
while true; do
  for s in 1 2; do mkdir $SL/$s 2>/dev/null && break 2; done
  sleep 30
done
trap 'rm -rf $SL/$s' EXIT
until mkdir $SL/pick.lock 2>/dev/null; do sleep 5; done  # one picker at a time, so two jobs never pick one GPU
read -r host idx <<<"$($REPO/scripts/remote_free_gpu.sh $SL/[12]/gpu)"
echo "$host $idx" > $SL/$s/gpu
rmdir $SL/pick.lock
echo "[$(date '+%m-%d %H:%M')] slot $s -> $host GPU $idx: $1"
C=$REPO/envs/cache
ssh -o BatchMode=yes -o ServerAliveInterval=60 $host.cplab.cs.nycu.edu.tw \
  "cd $REPO && export CUDA_VISIBLE_DEVICES=$idx VBENCH_CACHE_DIR=$C/vbench TORCH_HOME=$C/torch XDG_CACHE_HOME=$C/xdg \
   HF_HOME=$C/hf && $1"
