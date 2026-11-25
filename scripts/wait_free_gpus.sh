#!/bin/bash
# usage: wait_free_gpus.sh N  -> prints N GPU indices that have had no compute process and < 1000 MiB used
# for >= 60 s (team rule). Polls every 10 s until enough GPUs qualify.
N=$1
declare -A since
while true; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  now=$(date +%s); ok=()
  while IFS=', ' read -r idx uuid mem; do
    if [ "${mem%% *}" -lt 1000 ] && ! grep -q "$uuid" <<<"$busy"; then
      [ -z "${since[$idx]}" ] && since[$idx]=$now
      [ $((now - since[$idx])) -ge 60 ] && ok+=("$idx")
    else
      unset "since[$idx]"
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  if [ ${#ok[@]} -ge "$N" ]; then echo "${ok[@]:0:$N}"; exit 0; fi
  sleep 10
done
