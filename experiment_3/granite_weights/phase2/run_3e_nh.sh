#!/bin/bash
# 3e's commands on the non-holdout means of ws_pass3.py (their NOTES.md, "Phase 2", item 0). Runs next to the round 2
# sampler: the GPU share of each script is capped, so an overflow stops the script and not the sampler.
# Usage: ./run_3e_nh.sh STEP...   with steps nh, nhL, nhW, compare, lensL, lensW.   Logs: /content/ws/3e/ws_<step>.log
cd /content/ws_3e_code
ln -sf /dev/shm/ws_nh/nh_* /content/ws/
N=$(python3 -c "import json; c = json.load(open('/dev/shm/ws_nh/nh_counts.json')); print(str(c['nh_mean_imperial_h0']) + ',' + str(c['nh_mean_imperial_h1']) + ',' + str(c['nh_mean_modern']))")
run() { python3 -c "import torch, runpy, sys; torch.cuda.set_per_process_memory_fraction(${GPU_FRAC:-0.18}); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" "$@"; }
for step in "$@"; do
  log=/content/ws/3e/ws_$step.log
  case $step in
    nh)      run mean_structure.py --prefix nh_mean --tag nh --n_half $N --hold_from 60 > $log 2>&1 ;;
    nhL)     run mean_structure.py --prefix nh_mean_L --tag nhL --n_half $N --hold_from 60 --energy e_parts.npy:1 > $log 2>&1 ;;
    nhW)     run mean_structure.py --prefix nh_mean --minus nh_mean_L --tag nhW --n_half $N --hold_from 60 --energy e_parts.npy:0 > $log 2>&1 ;;
    compare) run compare_parts.py --tags dW=nh,W=nhW,L=nhL > $log 2>&1 ;;
    lensL)   run lens.py --tag nhL > $log 2>&1 ;;
    lensW)   run lens.py --tag nhW > $log 2>&1 ;;
  esac
  echo "$(date -u +%T) $step exit $? : $(tail -c 200 $log | tr '\n' ' ')" >> /content/ws/3e/ws_nh_steps.txt
done
