#!/bin/bash
# usage: remote_free_gpu.sh [exclude-file...] -> prints "host idx" of an RTX 4090 on gpu4/5/6 that has had no
# compute process and < 1000 MiB used for >= 60 s (same team rule as wait_free_gpus.sh). "host idx" lines in the
# exclude files (GPUs we already hold) are skipped. Polls every 15 s.
HOSTS="gpu4 gpu5 gpu6"
declare -A since
while true; do
  now=$(date +%s)
  held=$(cat "$@" 2>/dev/null)
  for h in $HOSTS; do
    out=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 $h.cplab.cs.nycu.edu.tw \
      'nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | tr "\n" " "; echo;
       nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits' 2>/dev/null) || continue
    busy=$(head -1 <<<"$out")
    while IFS=', ' read -r idx uuid mem; do
      [ -z "$idx" ] && continue
      k="$h $idx"
      if [ "$mem" -lt 1000 ] && ! grep -q "$uuid" <<<"$busy" && ! grep -qx "$k" <<<"$held"; then
        [ -z "${since[$k]}" ] && since[$k]=$now
        if [ $((now - since[$k])) -ge 60 ]; then echo "$k"; exit 0; fi
      else
        unset "since[$k]"
      fi
    done < <(tail -n +2 <<<"$out")
  done
  sleep 15
done
